import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.inferers import sliding_window_inference
from data_loader import build_brats2024_list_from_json, val_transforms
from model import get_brats_model

# Configurații
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"d:\study\licenta\creier\checkpoints\best_model.pth"
DATA_JSON = r"d:\study\licenta\creier\dataset\BRATS\brats_metadata_splits.json"
OUTPUT_DIR = r"d:\study\licenta\creier\visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_prediction(num_samples=3):
    # 1. Încarcă modelul
    model = get_brats_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Ia datele de validare
    val_files = build_brats2024_list_from_json(DATA_JSON, split="val_from_train")
    from monai.data import Dataset, DataLoader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    print(f"Începem vizualizarea pe {DEVICE}...")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE) # [1, 1, D, H, W]
            
            # Infernță (Sliding Window)
            outputs = sliding_window_inference(
                inputs, (96, 96, 96), 4, model, overlap=0.5
            )
            
            # Convertim în probabilități și apoi în clase (0, 1) pentru fiecare canal
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Pregătim Ground Truth pentru vizualizare (convertim din index în 3 canale)
            # 1: Necrotic, 2: Edema, 3: Enhancing
            gt_channels = torch.zeros((3, *labels.shape[2:]), device=DEVICE)
            gt_channels[0] = (labels[0, 0] == 1).float()
            gt_channels[1] = (labels[0, 0] == 2).float()
            gt_channels[2] = (labels[0, 0] == 3).float()

            # Mutăm pe CPU pentru plot
            img = inputs[0, 0].cpu().numpy() # T1c
            gt = gt_channels.cpu().numpy()
            pred = preds[0].cpu().numpy()

            # Găsim slice-ul cu cea mai multă tumoră în GT
            z_slice = np.argmax(np.sum(gt, axis=(0, 1, 2)))
            
            # Dacă nu e tumoră în GT (rar în BraTS), luăm mijlocul
            if np.sum(gt[..., z_slice]) == 0:
                z_slice = gt.shape[3] // 2

            plt.figure(figsize=(18, 6))
            
            # Imaginea RMN
            plt.subplot(1, 3, 1)
            plt.title(f"MRI (T1c) - Slice {z_slice}")
            plt.imshow(img[:, :, z_slice], cmap="gray")
            plt.axis("off")

            # Ground Truth
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth (Doctor)")
            gt_vis = np.zeros((*gt.shape[1:3], 3))
            gt_vis[..., 0] = gt[0, ..., z_slice] # Roșu: Necrotic
            gt_vis[..., 1] = gt[1, ..., z_slice] # Verde: Edema
            gt_vis[..., 2] = gt[2, ..., z_slice] # Albastru: Enhancing
            plt.imshow(gt_vis)
            plt.axis("off")

            # Predicția AI
            plt.subplot(1, 3, 3)
            plt.title("AI Prediction")
            pred_vis = np.zeros((*pred.shape[1:3], 3))
            pred_vis[..., 0] = pred[0, ..., z_slice]
            pred_vis[..., 1] = pred[1, ..., z_slice]
            pred_vis[..., 2] = pred[2, ..., z_slice]
            plt.imshow(pred_vis)
            plt.axis("off")

            save_path = os.path.join(OUTPUT_DIR, f"result_patient_{i}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Salvat: {save_path}")

if __name__ == "__main__":
    visualize_prediction()
