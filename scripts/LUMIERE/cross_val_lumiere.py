import os
import warnings
import gc
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose, EnsureTyped, KeepLargestConnectedComponent
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import KFold

# Configurare mediu pentru Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- CONFIGURARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRATS_PRETRAINED = os.environ.get('BRATS_PRETRAINED', r"D:/study/licenta/creier/checkpoints/model_epoch70.pth")
LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"
CHECKPOINT_DIR = r"D:/study/licenta/creier/checkpoints/lumiere_cv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hiperparametri (Simulăm Batch 4 prin Gradient Accumulation)
LR = 1e-5
EPOCHS = 12
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
FOLDS = 3

# Importuri module locale
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model  # type: ignore
from data_loader import train_transforms, val_transforms  # type: ignore

def get_brats_regions(y):
    """Extrage regiunile WT, TC, ET cu protecție la dimensiuni."""
    if y.shape[1] < 4:
        wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0.5
        tc = torch.zeros_like(wt)
        et = torch.zeros_like(wt)
    else:
        wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0.5
        tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > 0.5
        et = y[:, [3]] > 0.5
    return torch.cat([wt, tc, et], dim=1).float()

def evaluate_with_tta(model, val_loader, fixed_threshold=None):
    """Evaluare cu TTA, LCC și fix pentru indexarea Hausdorff."""
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
    
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)
    lcc = KeepLargestConnectedComponent(applied_labels=[1, 2, 3])

    sens_tp, sens_fp, sens_tn, sens_fn = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    
    with torch.no_grad():
        for val_data in val_loader:
            inputs, labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
            
            # TTA: Flip pe axele 2 și 3
            out = sliding_window_inference(inputs, (96, 96, 96), 4, model).softmax(dim=1)
            out += sliding_window_inference(inputs.flip(2), (96, 96, 96), 4, model).flip(2).softmax(dim=1)
            out += sliding_window_inference(inputs.flip(3), (96, 96, 96), 4, model).flip(3).softmax(dim=1)
            out /= 3.0

            arg = torch.stack([post_pred(i) for i in decollate_batch(out)])
            arg = torch.stack([lcc(i) for i in decollate_batch(arg)])
            
            pred_regs = get_brats_regions(arg)
            gt_regs = get_brats_regions(torch.stack([post_label(i) for i in decollate_batch(labels)]))

            dice_metric(y_pred=pred_regs, y=gt_regs)
            hd95_metric(y_pred=pred_regs, y=gt_regs)

            for ci in range(3):
                p_m, l_m = pred_regs[0, ci] > 0.5, gt_regs[0, ci] > 0.5
                sens_tp[ci] += (p_m & l_m).sum().item()
                sens_fn[ci] += (~p_m & l_m).sum().item()
                sens_tn[ci] += (~p_m & ~l_m).sum().item()
                sens_fp[ci] += (p_m & ~l_m).sum().item()
            
            del inputs, out, arg; torch.cuda.empty_cache(); gc.collect()

    # Agregare Dice
    metric_dice = dice_metric.aggregate()
    
    # --- FIX PENTRU IndexError Hausdorff ---
    hd95_raw = hd95_metric.aggregate()
    num_classes_in_buffer = hd95_raw.shape[1]
    metric_hd95 = []
    
    for i in range(3):
        if i < num_classes_in_buffer:
            val = torch.nanmean(hd95_raw[:, i]).item()
            metric_hd95.append(val if not np.isnan(val) else 0.0)
        else:
            metric_hd95.append(0.0) # Clasa lipsește din buffer (ex: niciun ET detectat)
    
    sens = [sens_tp[i]/(sens_tp[i]+sens_fn[i]) if (sens_tp[i]+sens_fn[i])>0 else 0 for i in range(3)]
    spec = [sens_tn[i]/(sens_tn[i]+sens_fp[i]) if (sens_tn[i]+sens_fp[i])>0 else 0 for i in range(3)]

    return {
        "WT": metric_dice[0].item(), "TC": metric_dice[1].item(), "ET": metric_dice[2].item(),
        "Mean": torch.mean(metric_dice).item(), "Threshold": fixed_threshold or 0.75,
        "HD95_WT": metric_hd95[0], "HD95_TC": metric_hd95[1], "HD95_ET": metric_hd95[2],
        "Sens": sens, "Spec": spec
    }

def main():
    with open(LUMIERE_JSON, "r") as f: data = json.load(f)
    patients = sorted(list(set([d["patient_id"] for d in data])))
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    all_fold_metrics = []
    pilot_threshold = 0.75

    for fold_idx, (_, val_p_idx) in enumerate(kf.split(patients)):
        val_files = [d for d in data if d["patient_id"] in [patients[i] for i in val_p_idx]]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold_idx+1}.pth")
        
        model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
        # Dacă fold-ul e antrenat, îl încărcăm, altfel folosim pre-trained
        model.load_state_dict(torch.load(checkpoint_path if os.path.exists(checkpoint_path) else BRATS_PRETRAINED))
        
        val_loader = DataLoader(Dataset(data=val_files, transform=val_transforms), batch_size=1)
        res = evaluate_with_tta(model, val_loader, fixed_threshold=pilot_threshold)
        all_fold_metrics.append(res)
        print(f"Fold {fold_idx+1} Evaluat: Mean Dice {res['Mean']:.4f}")

    print("\n" + "="*30 + "\nREZULTATE FINALE CV (Fixed)\n" + "="*30)
    for m in ["WT", "TC", "ET", "Mean"]:
        v = [f[m] for f in all_fold_metrics]
        print(f"{m}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
    # Afișare HD95
    hd_wt = [f['HD95_WT'] for f in all_fold_metrics if f['HD95_WT'] > 0]
    if hd_wt:
        print(f"HD95 WT (Mean): {np.mean(hd_wt):.2f} mm")

if __name__ == "__main__":
    main()