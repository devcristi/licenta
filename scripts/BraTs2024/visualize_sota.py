import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import Dataset, DataLoader, decollate_batch
from pathlib import Path
import scipy.ndimage as ndimage

# Importăm componentele noastre
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import build_brats2024_list_from_json, val_transforms
from model import get_brats_model

# CONFIGURARE - Putem rula pe GPU acum ca antrenarea e oprita
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
MODEL_PATH = r"D:/study/licenta/creier/checkpoints/best_model.pth"
DATA_JSON = r"D:/study/licenta/creier/dataset/BRATS/brats_metadata_splits.json"
OUTPUT_DIR = r"D:/study/licenta/creier/visualizations_sota"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_brats_regions(y):
    # Canal 0: BG, 1: Necrotic, 2: Edema, 3: Enhancing
    wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0
    tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > 0
    et = y[:, [3]] > 0
    return torch.cat([wt, tc, et], dim=1).float()

def compute_saliency(model, input_tensor, target_class=3):
    """
    Calculează o hartă de saliență îmbunătățită (Smoothed Saliency).
    """
    input_tensor.requires_grad = True
    
    # Forward pass
    output = model(input_tensor)
    
    # Target: suma activărilor pentru clasa dorită
    target_output = torch.sum(output[:, target_class, ...])
    
    model.zero_grad()
    target_output.backward()
    
    # Luăm valoarea absolută a gradientului și facem media pe canalele MRI
    saliency = torch.abs(input_tensor.grad).sum(dim=1)[0].cpu().numpy()
    
    # ÎMBUNĂTĂȚIRE: Aplicăm un filtru Gaussian pentru a netezi zgomotul
    # Asta transformă punctele izolate în "nori" de importanță (Heatmap)
    saliency = ndimage.gaussian_filter(saliency, sigma=2.0)
    
    # Normalizare 0-1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency

def visualize_sota(num_samples=1, target_id=None):
    print(f"Initializam vizualizarea SOTA + XAI pentru {'pacientul ' + target_id if target_id else 'mostre aleatorii'}...")
    
    # 1. Model
    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Date
    val_files = build_brats2024_list_from_json(DATA_JSON, split="val_from_train")
    
    if target_id:
        val_files = [f for f in val_files if f["subject_id"] == target_id]
        if not val_files:
            print(f"EROARE: Nu am gasit pacientul {target_id}")
            return
    
    val_ds = Dataset(data=val_files[:10], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False if target_id else True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            subject_id = batch["subject_id"][0] if "subject_id" in batch else f"sample_{i}"
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # Crop central manual pentru stabilitate XAI
            d, h, w = inputs.shape[2:]
            sd, sh, sw = 128, 128, 128
            z1, y1, x1 = (d-sd)//2, (h-sh)//2, (w-sw)//2
            inputs = inputs[:, :, z1:z1+sd, y1:y1+sh, x1:x1+sw]
            labels = labels[:, :, z1:z1+sd, y1:y1+sh, x1:x1+sw]
            
            print(f"Procesam {subject_id}...")
            
            # 1. Predicție
            outputs = model(inputs)
            seg_argmax = torch.argmax(outputs[0], dim=0).cpu().numpy()
            gt_argmax = labels[0, 0].cpu().numpy()
            
            # 2. XAI: Saliency
            with torch.enable_grad():
                saliency_et = compute_saliency(model, inputs, target_class=3)
            
            # Alegem slice-ul cu cea mai multă tumoră (sau centrul dacă e goală)
            tumor_mask = gt_argmax > 0
            if np.any(tumor_mask):
                z_slice = np.argmax(np.sum(tumor_mask, axis=(0, 1)))
            else:
                z_slice = sd // 2
            
            plt.figure(figsize=(20, 12))
            
            # 1. MRI T1c
            plt.subplot(2, 3, 1)
            plt.title(f"1. MRI (T1c) - {subject_id}")
            plt.imshow(inputs[0, 0, :, :, z_slice].detach().cpu(), cmap="gray")
            plt.axis("off")

            # 2. MRI FLAIR
            plt.subplot(2, 3, 2)
            plt.title("2. MRI (FLAIR)")
            plt.imshow(inputs[0, 3, :, :, z_slice].detach().cpu(), cmap="gray")
            plt.axis("off")

            # 3. Ground Truth
            from matplotlib.colors import ListedColormap
            cmap_custom = ListedColormap(["none", "blue", "green", "red"])
            plt.subplot(2, 3, 3)
            plt.title("3. Ground Truth (B=Nec, G=Ede, R=ET)")
            plt.imshow(inputs[0, 0, :, :, z_slice].detach().cpu(), cmap="gray")
            plt.imshow(gt_argmax[:, :, z_slice], cmap=cmap_custom, alpha=0.7)
            plt.axis("off")

            # 4. AI Prediction
            plt.subplot(2, 3, 4)
            plt.title("4. AI Prediction")
            plt.imshow(inputs[0, 0, :, :, z_slice].detach().cpu(), cmap="gray")
            plt.imshow(seg_argmax[:, :, z_slice], cmap=cmap_custom, alpha=0.7)
            plt.axis("off")
            
            # 5. Erori (AI vs GT)
            plt.subplot(2, 3, 5)
            plt.title("5. Erori (Rosu = Diferente)")
            error_map = (seg_argmax[:, :, z_slice] != gt_argmax[:, :, z_slice]) & ((gt_argmax[:, :, z_slice] > 0) | (seg_argmax[:, :, z_slice] > 0))
            plt.imshow(inputs[0, 0, :, :, z_slice].detach().cpu(), cmap="gray")
            plt.imshow(error_map, cmap="Reds", alpha=0.6)
            plt.axis("off")

            # 6. XAI Heatmap (Enhancing)
            plt.subplot(2, 3, 6)
            plt.title("6. XAI Heatmap (Importanta ET)")
            plt.imshow(inputs[0, 0, :, :, z_slice].detach().cpu(), cmap="gray")
            plt.imshow(saliency_et[:, :, z_slice], cmap="jet", alpha=0.5)
            plt.axis("off")

            plt.tight_layout()
            save_name = f"hardest_case_{subject_id}.png" if target_id else f"final_report_{i}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Salvat raport in: {save_path}")

if __name__ == "__main__":
    # Rulăm pentru cel mai greu pacient identificat
    visualize_sota(num_samples=1, target_id="BraTS-GLI-02878-100")
