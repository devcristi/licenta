import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Orientationd, Spacingd, EnsureTyped
from monai.inferers import sliding_window_inference
from skimage import morphology

# Adăugăm calea către modulele BraTS
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = r"D:/study/licenta/creier/checkpoints/lumiere_cv"
LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"
OUTPUT_DIR = r"D:/study/licenta/creier/scripts/LUMIERE/longitudinal_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pacientul ales pentru analiză (Patient-001 are multe vizite)
TARGET_PATIENT = "Patient-001"

# Transformări
val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"]),
])

def get_tta_flips():
    return [
        None,
        [2], [3], [4],
        [2, 3], [2, 4], [3, 4],
        [2, 3, 4]
    ]

def load_ensemble_models():
    models = []
    for i in range(1, 4):
        path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{i}.pth")
        if os.path.exists(path):
            model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            if all(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            model.eval()
            models.append(model)
            print(f"Model Fold {i} încărcat.")
    return models

def predict_with_ensemble_tta(models, image_tensor):
    all_probs = []
    flips = get_tta_flips()
    
    with torch.no_grad():
        for model in models:
            model_probs = torch.zeros((1, 4, *image_tensor.shape[2:]), device=DEVICE)
            for flip_dims in flips:
                inputs = image_tensor
                if flip_dims:
                    inputs = torch.flip(inputs, dims=flip_dims)
                
                outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model, overlap=0.5)
                probs = torch.softmax(outputs, dim=1)
                
                if flip_dims:
                    probs = torch.flip(probs, dims=flip_dims)
                
                model_probs += probs
            
            all_probs.append(model_probs / len(flips))
            
    # Media modelelor
    ensemble_probs = torch.mean(torch.stack(all_probs), dim=0)
    return ensemble_probs

def run_analysis():
    print(f"=== Analiză Longitudinală pentru {TARGET_PATIENT} ===")
    
    # 1. Încărcăm modelele
    models = load_ensemble_models()
    if not models:
        print("Nu s-au găsit modele salvate!")
        return

    # 2. Filtrăm vizitele pacientului
    with open(LUMIERE_JSON, "r") as f:
        metadata = json.load(f)
    
    patient_data = [d for d in metadata if d['patient_id'] == TARGET_PATIENT]
    # Sortăm după săptămână (week-000, week-004, etc.)
    patient_data.sort(key=lambda x: x['week'])
    
    print(f"S-au găsit {len(patient_data)} vizite.")

    volumes = []
    weeks = []

    for visit in patient_data:
        week_label = visit['week']
        print(f"Procesăm {week_label}...")
        
        ds = Dataset(data=[visit], transform=val_transforms)
        loader = DataLoader(ds, batch_size=1)
        batch = next(iter(loader))
        inputs = batch["image"].to(DEVICE)
        
        # Predicție Ensemble + TTA
        probs = predict_with_ensemble_tta(models, inputs)
        
        # Whole Tumor (WT) = Clasele 1, 2, 3
        wt_prob = torch.sum(probs[0, 1:], dim=0).cpu().numpy()
        wt_mask = (wt_prob > 0.75).astype(np.uint8) # Pragul fixat de noi
        
        # Post-procesare (eliminăm obiecte mici)
        wt_mask = morphology.remove_small_objects(wt_mask.astype(bool), min_size=500).astype(np.uint8)
        
        # Calcul volum (în mm^3, deoarece avem spacing 1x1x1)
        volume_mm3 = np.sum(wt_mask)
        volumes.append(volume_mm3)
        weeks.append(week_label)

    # 3. Generăm graficul
    plt.figure(figsize=(10, 6))
    plt.plot(weeks, volumes, marker='o', linestyle='-', color='crimson', linewidth=2, markersize=8)
    plt.title(f"Evoluția Volumului Tumoral - {TARGET_PATIENT}", fontsize=14)
    plt.xlabel("Vizită (Săptămână)", fontsize=12)
    plt.ylabel("Volum Whole Tumor (mm³)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Adăugăm etichete cu valorile pe puncte
    for i, vol in enumerate(volumes):
        plt.annotate(f"{vol}", (weeks[i], volumes[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"longitudinal_{TARGET_PATIENT}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Graficul a fost salvat în: {plot_path}")
    
    # Salvăm și datele brute în JSON
    results = {"patient_id": TARGET_PATIENT, "weeks": weeks, "volumes_mm3": volumes}
    with open(os.path.join(OUTPUT_DIR, f"data_{TARGET_PATIENT}.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_analysis()
