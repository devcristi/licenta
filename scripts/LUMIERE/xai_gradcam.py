import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from monai.visualize import GradCAM
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Orientationd, Spacingd, EnsureTyped, DivisiblePadd
from pathlib import Path
import sys
import json

# Adăugăm calea către modulele BraTS
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Folosim Fold 1 ca referință dacă e gata, altfel căutăm orice fold
CHECKPOINT_DIR = r"D:/study/licenta/creier/checkpoints/lumiere_cv"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_fold1.pth")
if not os.path.exists(MODEL_PATH):
    # Căutăm orice alt fold disponibil
    available = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if available:
        MODEL_PATH = os.path.join(CHECKPOINT_DIR, available[0])

LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"
OUTPUT_DIR = r"D:/study/licenta/creier/scripts/LUMIERE/xai_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Transformări pentru un singur pacient (fără crop, vrem imaginea întreagă)
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    DivisiblePadd(keys=["image", "label"], k=16), # Asigură dimensiuni divizibile cu 16
    EnsureTyped(keys=["image", "label"]),
])

def run_gradcam(patient_idx=0):
    print(f"Încărcăm modelul din: {MODEL_PATH}")
    
    # 1. Model
    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Curățăm state_dict dacă a fost salvat cu DataParallel
    if all(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Data
    with open(LUMIERE_JSON, "r") as f:
        metadata = json.load(f)
    
    # Alegem un pacient care are tumoră (pentru vizualizare relevantă)
    test_data = [metadata[patient_idx]]
    print(f"Analizăm pacientul: {test_data[0]['subject_id']}")
    
    ds = Dataset(data=test_data, transform=val_transforms)
    loader = DataLoader(ds, batch_size=1)
    
    batch = next(iter(loader))
    inputs = batch["image"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    # 3. Grad-CAM
    try:
        # În MONAI UNet, model.model este un Sequential.
        # model.model[0] este primul strat (convoluția de intrare)
        # model.model[-1] este ultimul strat (convoluția de ieșire)
        # Folosim ultimul strat de trăsături (înainte de final)
        cam = GradCAM(nn_module=model, target_layers="model.1") 
        
        # Generăm heatmap
        # class_idx=1 (WT), class_idx=2 (TC), class_idx=3 (ET)
        heatmap_wt = cam(x=inputs, class_idx=1)
        
        # 4. Vizualizare
        img_flair = inputs[0, 3].cpu().numpy() # FLAIR e cel mai bun pentru WT
        mask = labels[0, 0].cpu().numpy()
        cam_result = heatmap_wt[0, 0].cpu().numpy()
        
        # Găsim slice-ul cu cea mai mare arie de tumoră
        mask_sums = np.sum(mask, axis=(1, 2))
        slice_idx = np.argmax(mask_sums)
        
        if mask_sums[slice_idx] == 0:
            slice_idx = img_flair.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MRI
        axes[0].imshow(np.rot90(img_flair[slice_idx]), cmap="gray")
        axes[0].set_title("MRI (FLAIR)")
        axes[0].axis("off")
        
        # Ground Truth
        axes[1].imshow(np.rot90(img_flair[slice_idx]), cmap="gray")
        axes[1].imshow(np.rot90(mask[slice_idx]), cmap="jet", alpha=0.4)
        axes[1].set_title("Ground Truth (Tumor)")
        axes[1].axis("off")
        
        # Grad-CAM
        axes[2].imshow(np.rot90(img_flair[slice_idx]), cmap="gray")
        axes[2].imshow(np.rot90(cam_result[slice_idx]), cmap="jet", alpha=0.5)
        axes[2].set_title("Grad-CAM (Model Attention)")
        axes[2].axis("off")
        
        out_path = os.path.join(OUTPUT_DIR, f"gradcam_{test_data[0]['subject_id']}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        
        print(f"✅ Succes! Imaginea a fost salvată în: {out_path}")
        
    except Exception as e:
        print(f"❌ Eroare la Grad-CAM: {e}")
        print("Sfat: Verifică dacă 'target_layers' corespunde structurii modelului.")

if __name__ == "__main__":
    # Rulăm pentru primii 3 pacienți ca să avem de unde alege
    for i in range(3):
        run_gradcam(patient_idx=i)
