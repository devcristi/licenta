import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import json
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from pathlib import Path
import sys

# Adăugăm calea către modulele BraTS
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model  # type: ignore
from data_loader import val_transforms  # type: ignore

# CONFIGURARE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:/study/licenta/creier/checkpoints/lumiere_finetune/best_lumiere_model.pth"
LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"

def get_brats_regions(y, threshold=0.5):
    # y: [B, 4, D, H, W] (Proba/Softmax)
    wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > threshold
    tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > threshold
    et = y[:, [3]] > threshold
    return torch.cat([wt, tc, et], dim=1).float()

def evaluate():
    # 1. Date
    with open(LUMIERE_JSON, "r") as f:
        data = json.load(f)
    val_files = [d for d in data if d["split"] == "val"][:20] # Folosim același subset de validare
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 2. Model
    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Evaluăm modelul: {MODEL_PATH}")
    print("Rulăm inferență cu TTA (Flips)...")

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # TTA: Original + Flip X + Flip Y
            out = sliding_window_inference(inputs, (96, 96, 96), 4, model).softmax(dim=1)
            out += sliding_window_inference(inputs.flip(2), (96, 96, 96), 4, model).flip(2).softmax(dim=1)
            out += sliding_window_inference(inputs.flip(3), (96, 96, 96), 4, model).flip(3).softmax(dim=1)
            out /= 3.0
            
            all_outputs.append(out.cpu())
            all_labels.append(labels.cpu())
            if (i+1) % 5 == 0: print(f" Progres: {i+1}/{len(val_loader)}")

    # 3. Threshold Tuning
    print("\nCăutăm pragul optim (Threshold Tuning)...")
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_label = AsDiscrete(to_onehot=4)
    
    best_wt = 0
    best_thr = 0.5
    
    for thr in np.arange(0.3, 0.75, 0.05):
        dice_metric.reset()
        for out, label in zip(all_outputs, all_labels):
            # out is [1, 4, D, H, W] (probabilities)
            # label is [1, 1, D, H, W] (indices)
            pred_reg = get_brats_regions(out, threshold=thr)
            
            # Convert label to one-hot [1, 4, D, H, W]
            label_onehot = torch.stack([post_label(l) for l in decollate_batch(label)])
            gt_reg = get_brats_regions(label_onehot, threshold=0.5) # GT e fix
            
            dice_metric(y_pred=pred_reg, y=gt_reg)
        
        metrics = dice_metric.aggregate()
        current_wt = metrics[0].item()
        if current_wt > best_wt:
            best_wt = current_wt
            best_thr = thr
        print(f" Prag {thr:.2f} -> WT Dice: {current_wt:.4f}")

    # 4. Rezultate Finale
    print("\n" + "="*30)
    print(f"REZULTATE FINALE (TTA + Threshold {best_thr:.2f})")
    print("="*30)
    dice_metric.reset()
    for out, label in zip(all_outputs, all_labels):
        pred_reg = get_brats_regions(out, threshold=best_thr)
        label_onehot = torch.stack([post_label(l) for l in decollate_batch(label)])
        gt_reg = get_brats_regions(label_onehot, threshold=0.5)
        dice_metric(y_pred=pred_reg, y=gt_reg)
    
    final_metrics = dice_metric.aggregate()
    print(f"WT (Whole Tumor): {final_metrics[0].item():.4f}")
    print(f"TC (Tumor Core):  {final_metrics[1].item():.4f}")
    print(f"ET (Enhancing):   {final_metrics[2].item():.4f}")
    print(f"Mean Dice:        {torch.mean(final_metrics).item():.4f}")

if __name__ == "__main__":
    evaluate()
