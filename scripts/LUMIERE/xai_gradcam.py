import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Orientationd, Spacingd, EnsureTyped, DivisiblePadd
from pathlib import Path
import sys
import json

# Implementare Manuală Grad-CAM pentru a evita erorile de import MONAI/Tensorboard
class ManualGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Create a one-hot vector for the target class
        one_hot = torch.zeros_like(output)
        one_hot[:, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Global Average Pooling of gradients
        # Dimensionalitate: [Batch, Channel, D, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalizare 0-1
        cam -= torch.min(cam)
        cam /= (torch.max(cam) + 1e-10)
        
        return cam.detach()

# Adăugăm calea către modulele BraTS
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = r"D:/study/licenta/creier/checkpoints/lumiere_cv"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_fold1.pth")
if not os.path.exists(MODEL_PATH):
    available = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if available:
        MODEL_PATH = os.path.join(CHECKPOINT_DIR, available[0])

LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"
OUTPUT_DIR = r"D:/study/licenta/creier/scripts/LUMIERE/xai_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    DivisiblePadd(keys=["image", "label"], k=16),
    EnsureTyped(keys=["image", "label"]),
])

def run_gradcam(patient_idx=0):
    print(f"Incarcam modelul din: {MODEL_PATH}")
    
    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if all(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Data
    with open(LUMIERE_JSON, "r") as f:
        metadata = json.load(f)
    
    test_data = [metadata[patient_idx]]
    subject_id = test_data[0].get('subject_id', f"Patient-{patient_idx:03d}")
    week = test_data[0].get('week', 'unknown')
    print(f"Analizam pacientul: {subject_id} - {week}")
    
    ds = Dataset(data=test_data, transform=val_transforms)
    loader = DataLoader(ds, batch_size=1)
    
    batch = next(iter(loader))
    inputs = batch["image"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    # 3. Grad-CAM (Manual)
    try:
        # Din lista de module, găsim cel mai adânc strat convoluțional (Bottleneck)
        # sau unul care are rezoluție mică și trăsături semantice bogate.
        # Pentru MONAI UNet-ul nostru, acesta este de obicei la model.model[0].submodule...
        
        target_layer = None
        # Căutăm stratul convoluțional din centrul rețelei
        for name, module in model.named_modules():
            if "1.submodule.1.submodule.1.submodule.1.submodule.conv.unit1.conv" in name:
                target_layer = module
                break
        
        if target_layer is None:
            # Fallback: ia ultimul strat de trăsături înainte de stratul final 1x1
            # care este de obicei model.model[0] -> SkipConnection
            # Încercăm ceva mai sigur
            layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
            target_layer = layers[len(layers)//2] # Stratul din mijloc (mai mult sau mai puțin bottleneck)

        print(f"Target Layer identificat: {target_layer}")
        
        cam_viz = ManualGradCAM(model, target_layer)
        heatmap = cam_viz.generate(inputs, class_idx=1) # Class 1 = WT
        
        # Redimensionăm heatmap-ul la mărimea inputului
        heatmap_resized = torch.nn.functional.interpolate(
            heatmap, size=inputs.shape[2:], mode="trilinear", align_corners=False
        )
        cam_result = heatmap_resized[0, 0].cpu().numpy()
        
        # --- ÎMBUNĂTĂȚIRI STIL EXEMPLU ---
        # 1. Mascare: Eliminăm heatmap-ul de pe fundalul negru (unde nu e creier)
        brain_mask = (inputs[0, 3].cpu().numpy() > 0.05) # Mască bazată pe FLAIR
        cam_result = cam_result * brain_mask
        
        # 2. Thresholding: Eliminăm "zgomotul" (activările sub 30% devin zero)
        # Atenție: În exemplul tău, zonele de sub prag sunt transparente/negre.
        cam_result[cam_result < 0.3] = 0 
        
        # 3. Normalizare finală pentru contrast maxim
        if np.max(cam_result) > 0:
            cam_result = (cam_result - np.min(cam_result)) / (np.max(cam_result) - np.min(cam_result))
        
        # 4. Vizualizare
        img_flair = inputs[0, 3].cpu().numpy()
        mask = labels[0, 0].cpu().numpy()
        
        mask_sums = np.sum(mask, axis=(1, 2))
        slice_idx = np.argmax(mask_sums)
        if mask_sums[slice_idx] == 0: slice_idx = img_flair.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
        
        # Stilizare axe (fără chenare albe)
        for ax in axes:
            ax.set_facecolor('black')
            ax.axis("off")
        
        # MRI
        axes[0].imshow(np.rot90(img_flair[slice_idx]), cmap="gray")
        axes[0].set_title("Input MRI", color='white', fontsize=14, pad=10)
        
        # Ground Truth (Stil medical)
        axes[1].imshow(np.rot90(img_flair[slice_idx]), cmap="gray")
        # Folosim o culoare solidă pentru GT (ex: verde neon sau albastru)
        axes[1].imshow(np.rot90(mask[slice_idx]), cmap="winter", alpha=0.4)
        axes[1].set_title("Target Region", color='white', fontsize=14, pad=10)
        
        # Grad-CAM (Stilul din poza ta)
        axes[2].imshow(np.rot90(img_flair[slice_idx]), cmap="gray")
        # Interpolare 'Gaussian' pentru acel efect de "glow" din poza ta
        im = axes[2].imshow(np.rot90(cam_result[slice_idx]), cmap="jet", alpha=0.7, interpolation="gaussian")
        axes[2].set_title("AI Grad-CAM Focus", color='yellow', fontsize=16, fontweight='bold', pad=10)
        
        # Layout ajustat să nu aibă margini albe
        plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95)
        
        out_path = os.path.join(OUTPUT_DIR, f"gradcam_{subject_id}_{week}.png")
        plt.savefig(out_path, dpi=250, facecolor='black', bbox_inches='tight')
        plt.close()
        
        print(f"Imagine salvata: {out_path}")
        
    except Exception as e:
        print(f"Eroare: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Rulăm pentru Patient-001 (primele câteva vizite)
    with open(LUMIERE_JSON, "r") as f:
        metadata = json.load(f)
    
    p001_indices = [i for i, d in enumerate(metadata) if d['patient_id'] == "Patient-001"]
    for i in p001_indices[:3]: # Primele 3 vizite
        run_gradcam(patient_idx=i)

