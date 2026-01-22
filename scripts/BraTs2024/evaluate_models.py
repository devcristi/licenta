
import os
import warnings
import torch
import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, Compose
from monai.utils import set_determinism
from pathlib import Path
import random

# Reduce noisy logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

from data_loader import build_brats2024_list_from_json, val_transforms
from model import get_brats_model

# --- Configurație ---
CONFIG = {
    "json_path": "../../dataset/BRATS/brats_metadata_splits.json",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_dir": "D:/study/licenta/creier/checkpoints", # Cale absolută
}

def get_brats_regions(y):
    """
    Convertește cele 3 canale de tumoră în regiunile standard BraTS:
    WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)
    """
    wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0
    tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > 0
    et = y[:, [3]] > 0
    return torch.cat([wt, tc, et], dim=1).float()

def evaluate_model(model_path: str, model_name: str, post_processing_transform: Compose):
    """
    Rulează evaluarea pe un model salvat, cu o strategie de post-procesare specifică.
    """
    print(f"--- Evaluare pentru modelul: {model_name} ---")
    print(f"Folosind post-procesare: {post_processing_transform.transforms[1]}")

    # 1. Data Loader
    base_dir = Path(__file__).parent.parent.parent
    json_path = base_dir / "dataset" / "BRATS" / "brats_metadata_splits.json"
    val_files = build_brats2024_list_from_json(str(json_path), split="val_from_train")
    
    # Folosim același subset de 20 pentru consistență
    random.seed(42)
    random.shuffle(val_files)
    val_files_subset = val_files[:20]

    val_ds = Dataset(data=val_files_subset, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # 2. Model
    model = get_brats_model(in_channels=4, out_channels=4).to(CONFIG["device"])
    
    # Încărcare checkpoint
    checkpoint = torch.load(model_path, map_location=CONFIG["device"])
    if "model_state_dict" in checkpoint:
        # Format din 'latest_checkpoint.pth'
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint incarcat din dictionar (epoca: {checkpoint.get('epoch', 'N/A')})")
    else:
        # Format din 'best_model.pth' (doar state_dict)
        model.load_state_dict(checkpoint)
        print("Checkpoint incarcat direct (format state_dict).")

    model.eval()

    # 3. Metrici
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", percentile=95)
    
    post_label = AsDiscrete(to_onehot=4)
    
    sens_tp = [0, 0, 0]
    sens_fp = [0, 0, 0]
    sens_tn = [0, 0, 0]
    sens_fn = [0, 0, 0]

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(CONFIG["device"]), val_data["label"].to(CONFIG["device"])
            
            # --- TEST-TIME AUGMENTATION (TTA) ---
            from monai.inferers import sliding_window_inference
            def get_pred(x):
                return sliding_window_inference(x, (96, 96, 96), sw_batch_size=4, predictor=model, overlap=0.25)

            val_outputs = get_pred(val_inputs)
            val_outputs += torch.flip(get_pred(torch.flip(val_inputs, dims=[2])), dims=[2])
            val_outputs += torch.flip(get_pred(torch.flip(val_inputs, dims=[3])), dims=[3])
            val_outputs += torch.flip(get_pred(torch.flip(val_inputs, dims=[4])), dims=[4])
            val_outputs /= 4.0
            
            # Aplică post-procesarea specificată (LCC)
            val_outputs = [post_processing_transform(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            
            # Convertim în regiuni WT, TC, ET
            val_outputs_regions = [get_brats_regions(i.unsqueeze(0)) for i in val_outputs]
            val_labels_regions = [get_brats_regions(i.unsqueeze(0)) for i in val_labels]
            
            for p, l in zip(val_outputs_regions, val_labels_regions):
                dice_metric(y_pred=p, y=l)
                hd95_metric(y_pred=p, y=l)

                # Calcul TP/FP/TN/FN pentru Sensibilitate/Specificitate
                p_arr, l_arr = p.squeeze(0), l.squeeze(0)
                for ci in range(3): # WT, TC, ET
                    pred_mask, label_mask = p_arr[ci] > 0, l_arr[ci] > 0
                    sens_tp[ci] += int((pred_mask & label_mask).sum().item())
                    sens_fp[ci] += int((pred_mask & (~label_mask)).sum().item())
                    sens_fn[ci] += int(((~pred_mask) & label_mask).sum().item())
                    sens_tn[ci] += int(((~pred_mask) & (~label_mask)).sum().item())

    # Agregare și calcul metrici
    metric_dice = dice_metric.aggregate()
    metric_hd95 = hd95_metric.aggregate()
    
    metric_sens, metric_spec = [], []
    for ci in range(3):
        sens = (sens_tp[ci] / (sens_tp[ci] + sens_fn[ci])) if (sens_tp[ci] + sens_fn[ci]) > 0 else 0.0
        spec = (sens_tn[ci] / (sens_tn[ci] + sens_fp[ci])) if (sens_tn[ci] + sens_fp[ci]) > 0 else 0.0
        metric_sens.append(sens)
        metric_spec.append(spec)

    results = {
        "Model": model_name,
        "Dice_WT": metric_dice[0].item(), "Dice_TC": metric_dice[1].item(), "Dice_ET": metric_dice[2].item(),
        "Dice_Mean": metric_dice.mean().item(),
        "HD95_WT": metric_hd95[0].item(), "HD95_TC": metric_hd95[1].item(), "HD95_ET": metric_hd95[2].item(),
        "Sens_WT": metric_sens[0], "Sens_TC": metric_sens[1], "Sens_ET": metric_sens[2],
        "Spec_WT": metric_spec[0], "Spec_TC": metric_spec[1], "Spec_ET": metric_spec[2],
    }
    
    print(f"Evaluare finalizata pentru {model_name}.\n")
    return results

if __name__ == "__main__":
    set_determinism(seed=42)
    torch.backends.cudnn.benchmark = True

    # --- Definire modele și post-procesare ---
    model_dir = Path(CONFIG["model_dir"])
    model_paths = {
        "Model Epoca 14 (Best)": model_dir / "best_model.pth",
        "Model Epoca 44 (Latest)": model_dir / "latest_checkpoint.pth",
    }
    
    # Post-procesare cu LCC pe toate clasele de tumoare (1:Necrotic, 2:Edema, 3:Enhancing)
    post_proc_lcc_all = Compose([
        AsDiscrete(argmax=True, to_onehot=4),
        KeepLargestConnectedComponent(applied_labels=[1, 2, 3])
    ])
    
    all_results = []
    for name, path in model_paths.items():
        if not path.exists():
            print(f"Atentie: Fisierul model '{path}' nu a fost gasit. Se omite evaluarea.")
            continue
        
        results = evaluate_model(
            model_path=str(path), 
            model_name=name, 
            post_processing_transform=post_proc_lcc_all
        )
        all_results.append(results)

    # --- Afișare tabel comparativ ---
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.set_index("Model")
        
        # Formatare pentru lizibilitate
        pd.set_option('display.precision', 4)
        
        # Organizare pe categorii de metrici
        dice_cols = ["Dice_WT", "Dice_TC", "Dice_ET", "Dice_Mean"]
        hd95_cols = ["HD95_WT", "HD95_TC", "HD95_ET"]
        sens_cols = ["Sens_WT", "Sens_TC", "Sens_ET"]
        spec_cols = ["Spec_WT", "Spec_TC", "Spec_ET"]
        
        print("\n" + "="*80)
        print(" " * 25 + "REZULTATE COMPARATIVE (LCC pe toate clasele)")
        print("="*80)
        
        print("\n--- DICE SCORE ---")
        print(df[dice_cols].to_markdown(index=True))
        
        print("\n--- HAUSDORFF95 DISTANCE (mai mic = mai bun) ---")
        print(df[hd95_cols].to_markdown(index=True))
        
        print("\n--- SENSITIVITY (True Positive Rate) ---")
        print(df[sens_cols].to_markdown(index=True))
        
        print("\n--- SPECIFICITY (True Negative Rate) ---")
        print(df[spec_cols].to_markdown(index=True))
        print("\n" + "="*80)

    else:
        print("Nu s-a putut rula nicio evaluare. Verificati caile catre modele.")
