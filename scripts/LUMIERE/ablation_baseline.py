import os
import torch
import json
import numpy as np
import warnings
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Orientationd, Spacingd, EnsureTyped
from pathlib import Path
import sys

# Silențiem warning-urile
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adăugăm calea către modulele BraTS
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model

# CONFIGURARE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRATS_PRETRAINED = r"D:/study/licenta/creier/checkpoints/best_model.pth"
LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"

# Transformări standard de validare
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

def run_baseline_evaluation(num_samples=30):
    print("=== EVALUARE BASELINE (Model BraTS Original pe LUMIERE) ===")
    print(f"Se evaluează pe un subset de {num_samples} vizite pentru viteză.")

    # 1. Încărcare Date
    with open(LUMIERE_JSON, "r") as f:
        data = json.load(f)
    
    # Luăm un subset reprezentativ (primele num_samples)
    subset_data = data[:num_samples]
    ds = Dataset(data=subset_data, transform=val_transforms)
    loader = DataLoader(ds, batch_size=1)

    # 2. Încărcare Model Original (fără fine-tuning)
    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    checkpoint = torch.load(BRATS_PRETRAINED, map_location=DEVICE)
    if all(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    print("Începem inferența...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            
            # Predicție simplă (fără TTA, fără post-proc, doar modelul brut)
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
            
            # Convertim în regiuni BraTS (WT, TC, ET)
            # Folosim pragul standard de 0.5 pentru baseline
            probs = torch.softmax(outputs, dim=1)
            wt_p = (torch.sum(probs[:, 1:], dim=1, keepdim=True) > 0.5).float()
            tc_p = (torch.sum(probs[:, [1, 3]], dim=1, keepdim=True) > 0.5).float()
            et_p = (probs[:, [3]] > 0.5).float()
            pred_reg = torch.cat([wt_p, tc_p, et_p], dim=1)
            
            # Ground Truth regions
            wt_g = (label > 0).float()
            tc_g = torch.logical_or(label == 1, label == 3).float()
            et_g = (label == 3).float()
            gt_reg = torch.cat([wt_g, tc_g, et_g], dim=1)
            
            dice_metric(y_pred=pred_reg, y=gt_reg)
            
            if (i+1) % 5 == 0:
                print(f" Progres: {i+1}/{num_samples}")

    results = dice_metric.aggregate()
    print("\n" + "="*40)
    print("REZULTATE BASELINE (FĂRĂ FINE-TUNING)")
    print("="*40)
    print(f"WT (Whole Tumor): {results[0].item():.4f}")
    print(f"TC (Tumor Core):  {results[1].item():.4f}")
    print(f"ET (Enhancing):   {results[2].item():.4f}")
    print(f"Mean Dice:        {torch.mean(results).item():.4f}")
    print("="*40)
    print("\nAceste rezultate demonstrează performanța modelului 'out-of-the-box'.")
    print("Compară-le cu rezultatele tale finale pentru a arăta valoarea fine-tuning-ului!")

if __name__ == "__main__":
    run_baseline_evaluation()
