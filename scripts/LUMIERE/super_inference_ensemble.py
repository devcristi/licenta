import os
import warnings

# Silențiem TOATE warning-urile înainte de orice import
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import json
import numpy as np
import gc
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureTyped
from monai.utils import set_determinism
from pathlib import Path
import sys
from skimage import morphology
from sklearn.model_selection import KFold

# Silențiem warning-urile
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adăugăm calea către modulele BraTS
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model  # type: ignore
from data_loader import val_transforms  # type: ignore

# CONFIGURARE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = r"D:/study/licenta/creier/checkpoints/lumiere_cv"
LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"
THRESHOLD = 0.75 # Pragul optim găsit la Fold 1

def remove_small_objects(seg, min_size=500):
    """Elimină componentele conectate mai mici de min_size voxeli."""
    # seg: [3, D, H, W]
    seg_np = seg.cpu().numpy().astype(bool)
    for i in range(seg_np.shape[0]):
        seg_np[i] = morphology.remove_small_objects(seg_np[i], min_size=min_size)
    return torch.from_numpy(seg_np).float().to(DEVICE)

def get_tta_flips(inputs):
    """Generează toate cele 8 combinații de flip-uri (Full TTA)."""
    flips = []
    for fz in [0, 1]:
        for fy in [0, 1]:
            for fx in [0, 1]:
                out = inputs
                if fz: out = out.flip(2)
                if fy: out = out.flip(3)
                if fx: out = out.flip(4)
                flips.append((out, (fz, fy, fx)))
    return flips

def undo_tta_flip(output, flip_config):
    """Anulează flip-ul pentru a reveni la orientarea originală."""
    fz, fy, fx = flip_config
    if fx: output = output.flip(4)
    if fy: output = output.flip(3)
    if fz: output = output.flip(2)
    return output

def run_super_inference():
    set_determinism(seed=42)
    
    # 1. Încărcare Date și Recreare Split-uri (pentru a ști cine e în ce fold)
    with open(LUMIERE_JSON, "r") as f:
        data = json.load(f)
    
    # Identificăm pacienții unici (la fel ca în cross_val_lumiere.py)
    patient_ids = sorted(list(set([d['patient_id'] for d in data])))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Mapăm fiecare pacient la fold-ul său de VALIDARE
    patient_to_val_fold = {}
    for fold_idx, (_, val_idx) in enumerate(kf.split(patient_ids)):
        for i in val_idx:
            patient_to_val_fold[patient_ids[i]] = fold_idx + 1

    # 2. Încărcare Modele
    models = {}
    for i in range(1, 4):
        path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{i}.pth")
        if os.path.exists(path):
            model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            if all(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            model.eval()
            models[i] = model
            print(f"Model Fold {i} încărcat.")
    
    if len(models) < 3:
        print(f"Atenție: S-au găsit doar {len(models)} modele din 3!")

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    
    print(f"\nÎncepem Out-of-Fold (OOF) Inference pe {len(data)} vizite...")
    print(f"Metodă: Fiecare vizită este evaluată DOAR de modelul care NU a văzut pacientul la antrenare.")
    print(f"Configurație: Single-Fold Model + Full TTA (8 flips) + Post-processing")

    with torch.no_grad():
        for i, item in enumerate(data):
            # Determinăm ce model trebuie să folosească acest pacient
            p_id = item['patient_id']
            fold_to_use = patient_to_val_fold.get(p_id)
            
            if fold_to_use not in models:
                continue # Sărim dacă modelul nu e gata

            model = models[fold_to_use]
            
            # Încărcăm imaginea și label-ul
            batch = val_transforms(item)
            inputs = batch["image"].unsqueeze(0).to(DEVICE)
            label = batch["label"].unsqueeze(0).to(DEVICE)
            
            # Full TTA pentru modelul respectiv
            final_probs = torch.zeros((1, 4, *inputs.shape[2:]), device=DEVICE)
            tta_configs = get_tta_flips(inputs)
            
            for flipped_inputs, config in tta_configs:
                out = sliding_window_inference(flipped_inputs, (96, 96, 96), 4, model).softmax(dim=1)
                final_probs += undo_tta_flip(out, config)
            
            final_probs /= len(tta_configs)
            
            # Convertim în regiuni BraTS cu pragul optim
            wt_p = (torch.sum(final_probs[:, 1:], dim=1, keepdim=True) > THRESHOLD).float()
            tc_p = (torch.sum(final_probs[:, [1, 3]], dim=1, keepdim=True) > THRESHOLD).float()
            et_p = (final_probs[:, [3]] > THRESHOLD).float()
            pred_reg = torch.cat([wt_p, tc_p, et_p], dim=1).squeeze(0)
            
            # Post-procesare (eliminare zgomot)
            pred_reg = remove_small_objects(pred_reg)
            
            # Ground Truth regions
            wt_g = (label > 0).float()
            tc_g = torch.logical_or(label == 1, label == 3).float()
            et_g = (label == 3).float()
            gt_reg = torch.cat([wt_g, tc_g, et_g], dim=1).squeeze(0)
            
            dice_metric(y_pred=pred_reg.unsqueeze(0), y=gt_reg.unsqueeze(0))
            
            if (i+1) % 5 == 0:
                print(f" Progres: {i+1}/{len(data)}")
            
            # Curățenie RAM
            del inputs, label, final_probs, pred_reg, gt_reg
            torch.cuda.empty_cache()

    results = dice_metric.aggregate()
    print("\n" + "="*40)
    print("REZULTATE SUPREME (ENSEMBLE + FULL TTA + POST-PROC)")
    print("="*40)
    print(f"WT (Whole Tumor): {results[0].item():.4f}")
    print(f"TC (Tumor Core):  {results[1].item():.4f}")
    print(f"ET (Enhancing):   {results[2].item():.4f}")
    print(f"Mean Dice:        {torch.mean(results).item():.4f}")
    print("="*40)

if __name__ == "__main__":
    run_super_inference()
