import os
import warnings
import gc

# Silențiem warning-urile enervante
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silențiem TensorFlow

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureTyped
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import KFold

# Adăugăm calea către modulele BraTS
import sys
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model  # type: ignore
from data_loader import train_transforms, val_transforms  # type: ignore

# CONFIGURARE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRATS_PRETRAINED = r"D:/study/licenta/creier/checkpoints/best_model.pth"
LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"
CHECKPOINT_DIR = r"D:/study/licenta/creier/checkpoints/lumiere_cv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hiperparametri
LR = 1e-5
EPOCHS = 12 # Reducem puțin pentru CV, dar userul poate crește
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
FOLDS = 3

def get_brats_regions(y):
    # y: [B, 4, D, H, W] (Softmax/One-hot)
    wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0.5
    tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > 0.5
    et = y[:, [3]] > 0.5
    return torch.cat([wt, tc, et], dim=1).float()

def run_fold(fold_idx, train_files, val_files):
    print(f"\n=== START FOLD {fold_idx+1}/{FOLDS} ===")
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    
    # Încărcăm pre-antrenarea BraTS pentru fiecare fold
    checkpoint = torch.load(BRATS_PRETRAINED, map_location=DEVICE)
    if all(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, weight=torch.tensor([0.5, 2.0, 1.0, 2.0]).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_seg = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    best_dice = 0.0
    fold_results = {}

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader):
            inputs, labels = batch_data["image"].to(DEVICE), batch_data["label"].to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item()
        
        # Validare
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                
                val_outputs = [post_seg(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                val_outputs_reg = [get_brats_regions(i.unsqueeze(0)).squeeze(0) for i in val_outputs]
                val_labels_reg = [get_brats_regions(i.unsqueeze(0)).squeeze(0) for i in val_labels]
                dice_metric(y_pred=val_outputs_reg, y=val_labels_reg)
            
            metric_batch = dice_metric.aggregate()
            mean_dice = torch.mean(metric_batch).item()
            dice_metric.reset()
            
            scheduler.step(mean_dice)
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold_idx+1}.pth"))
                fold_results = {
                    "WT": metric_batch[0].item(),
                    "TC": metric_batch[1].item(),
                    "ET": metric_batch[2].item(),
                    "Mean": mean_dice
                }
        print(f"Fold {fold_idx+1} Ep {epoch+1} - Mean Dice: {mean_dice:.4f}")

    return fold_results

def evaluate_with_tta(model, val_loader, fixed_threshold=None):
    """Evaluare finală cu Test Time Augmentation (TTA) și Threshold Tuning."""
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    
    # Eliberăm memoria GPU/RAM înainte de evaluare
    torch.cuda.empty_cache()
    gc.collect()

    all_outputs_reg = [] # Stocăm doar probabilitățile regiunilor (3 canale în loc de 4)
    all_labels_reg = []  # Stocăm GT-ul ca boolean (ocupă mult mai puțin spațiu)

    print("Rulăm TTA pe setul de validare...")
    with torch.no_grad():
        for val_data in val_loader:
            inputs = val_data["image"].to(DEVICE)
            label = val_data["label"].to(DEVICE)
            
            # TTA: Original + Flip X + Flip Y
            out = sliding_window_inference(inputs, (96, 96, 96), 4, model).softmax(dim=1)
            out += sliding_window_inference(inputs.flip(2), (96, 96, 96), 4, model).flip(2).softmax(dim=1)
            out += sliding_window_inference(inputs.flip(3), (96, 96, 96), 4, model).flip(3).softmax(dim=1)
            out /= 3.0
            
            # Extragem probabilitățile pentru regiuni (WT, TC, ET) pentru a economisi RAM
            wt_p = torch.sum(out[:, 1:], dim=1, keepdim=True)
            tc_p = torch.sum(out[:, [1, 3]], dim=1, keepdim=True)
            et_p = out[:, [3]]
            reg_p = torch.cat([wt_p, tc_p, et_p], dim=1).cpu().half() # half() reduce memoria la jumătate (float16)
            
            # Extragem GT regions ca bool
            wt_g = (label > 0)
            tc_g = torch.logical_or(label == 1, label == 3)
            et_g = (label == 3)
            reg_g = torch.cat([wt_g, tc_g, et_g], dim=1).cpu().bool()
            
            all_outputs_reg.append(reg_p)
            all_labels_reg.append(reg_g)
            
            # Curățăm memoria GPU după fiecare pacient
            del inputs, label, out, wt_p, tc_p, et_p, wt_g, tc_g, et_g
            torch.cuda.empty_cache()

    best_thr = fixed_threshold
    
    if fixed_threshold is None:
        # Threshold Tuning (doar pentru Fold-ul Pilot)
        best_wt = 0.0
        print("Căutăm pragul optim (Threshold Tuning - Pilot Fold)...")
        for thr in np.arange(0.3, 0.8, 0.05):
            dice_metric.reset()
            for out_reg, label_reg in zip(all_outputs_reg, all_labels_reg):
                pred_reg = (out_reg > thr).float()
                dice_metric(y_pred=pred_reg, y=label_reg.float())
            
            current_wt = dice_metric.aggregate()[0].item()
            if current_wt > best_wt:
                best_wt = current_wt
                best_thr = thr
        print(f"Prag optim găsit pe Pilot Fold: {best_thr:.2f}")
    else:
        print(f"Folosim pragul fixat anterior: {best_thr:.2f}")
    
    # Rezultate finale cu pragul ales
    dice_metric.reset()
    for out_reg, label_reg in zip(all_outputs_reg, all_labels_reg):
        pred_reg = (out_reg > best_thr).float()
        dice_metric(y_pred=pred_reg, y=label_reg.float())
        
    final_metrics = dice_metric.aggregate()
    
    # Curățăm listele mari înainte de a ieși
    del all_outputs_reg, all_labels_reg
    gc.collect()

    return {
        "WT": final_metrics[0].item(),
        "TC": final_metrics[1].item(),
        "ET": final_metrics[2].item(),
        "Mean": torch.mean(final_metrics).item(),
        "Threshold": best_thr
    }
        
    final_metrics = dice_metric.aggregate()
    return {
        "WT": final_metrics[0].item(),
        "TC": final_metrics[1].item(),
        "ET": final_metrics[2].item(),
        "Mean": torch.mean(final_metrics).item()
    }

def main():
    with open(LUMIERE_JSON, "r") as f:
        data = json.load(f)
    
    # Grupăm datele pe pacienți
    patients = sorted(list(set([d["patient_id"] for d in data])))
    print(f"Total pacienți: {len(patients)}")
    
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    all_fold_metrics = []
    pilot_threshold = None
    
    for fold_idx, (train_p_idx, val_p_idx) in enumerate(kf.split(patients)):
        train_patients = [patients[i] for i in train_p_idx]
        val_patients = [patients[i] for i in val_p_idx]
        
        train_files = [d for d in data if d["patient_id"] in train_patients]
        val_files = [d for d in data if d["patient_id"] in val_patients]
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold_idx+1}.pth")
        
        # 1. Antrenare Fold (doar dacă nu există deja checkpoint-ul)
        if os.path.exists(checkpoint_path):
            print(f"Fold {fold_idx+1} deja antrenat. Sărim peste training.")
        else:
            fold_res_train = run_fold(fold_idx, train_files, val_files)
        
        # 2. Evaluare finală cu TTA pe acest fold folosind cel mai bun model salvat
        model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path))
        
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        # Dacă e primul fold, găsim pragul. Pentru restul, îl folosim pe cel găsit.
        fold_res_tta = evaluate_with_tta(model, val_loader, fixed_threshold=pilot_threshold)
        
        if pilot_threshold is None:
            pilot_threshold = fold_res_tta["Threshold"]
            
        all_fold_metrics.append(fold_res_tta)
        
        print(f"Rezultate Fold {fold_idx+1} (cu TTA, Thr={fold_res_tta['Threshold']}): {fold_res_tta}")
        
    # Raportare finală
    print("\n" + "="*30)
    print(f"REZULTATE FINALE {FOLDS}-FOLD CV (cu TTA și Threshold Tuning)")
    print("="*30)
    
    for m_name in ["WT", "TC", "ET", "Mean"]:
        vals = [m[m_name] for m in all_fold_metrics]
        print(f"{m_name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

if __name__ == "__main__":
    main()
