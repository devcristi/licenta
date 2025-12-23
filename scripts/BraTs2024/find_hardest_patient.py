import os
import torch
import numpy as np
import json
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import Dataset, DataLoader, decollate_batch
from pathlib import Path

# Importăm componentele noastre
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import build_brats2024_list_from_json, val_transforms
from model import get_brats_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:/study/licenta/creier/checkpoints/best_model.pth"
DATA_JSON = r"D:/study/licenta/creier/dataset/BRATS/brats_metadata_splits.json"

def get_brats_regions(y):
    wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0
    tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > 0
    et = y[:, [3]] > 0
    return torch.cat([wt, tc, et], dim=1).float()

def find_hardest_patient():
    print(f"Evaluăm tot setul de validare pe {DEVICE} pentru a găsi cel mai greu pacient...")
    
    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    val_files = build_brats2024_list_from_json(DATA_JSON, split="val_from_train")
    val_ds = Dataset(data=val_files[:50], transform=val_transforms) # Evaluăm primii 50 pentru viteză
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    dice_metric = DiceMetric(include_background=True, reduction="none")
    post_seg = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    results = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # Extragem ID-ul pacientului din path
            patient_path = batch["label_meta_dict"]["filename_or_obj"][0]
            patient_id = Path(patient_path).parent.name
            
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model, overlap=0.5)
            
            seg = post_seg(outputs[0]).unsqueeze(0)
            gt = post_label(labels[0]).unsqueeze(0)
            
            seg_regions = get_brats_regions(seg)
            gt_regions = get_brats_regions(gt)
            
            dice_metric(y_pred=seg_regions, y=gt_regions)
            dice_scores = dice_metric.aggregate().cpu().numpy()[0] # [WT, TC, ET]
            dice_metric.reset()
            
            mean_dice = np.nanmean(dice_scores)
            
            results.append({
                "id": patient_id,
                "wt": dice_scores[0],
                "tc": dice_scores[1],
                "et": dice_scores[2],
                "mean": mean_dice
            })
            
            if (i + 1) % 10 == 0:
                print(f"Progres: {i+1}/{len(val_files)}...")

    # Sortăm după Mean Dice (crescător)
    results.sort(key=lambda x: x["mean"])
    
    print("\n--- TOP 5 CEI MAI GREI PACIENȚI (Lowest Dice) ---")
    for i in range(min(5, len(results))):
        p = results[i]
        print(f"{i+1}. ID: {p['id']} | Mean: {p['mean']:.4f} | WT: {p['wt']:.4f}, TC: {p['tc']:.4f}, ET: {p['et']:.4f}")

if __name__ == "__main__":
    find_hardest_patient()
