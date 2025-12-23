import json
import os
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, RandCropByPosNegLabeld, RandFlipd,
    RandAffined, EnsureTyped, Lambdad
)

def build_brats2024_list_from_json(json_path: str, split: str = "train"):
    """
    Construiește lista de fișiere folosind metadatele din JSON.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    items = []
    for entry in data:
        if entry.get("final_split") != split:
            continue
            
        sample = {
            "subject_id": entry["subject_id"],
            "patient_id": entry["patient_id"]
        }
        
        # Extragem căile pentru fiecare tip de fișier
        for file_info in entry["all_files"]:
            f_type = file_info["type"]
            if f_type == "segmentation":
                sample["label"] = file_info["seg_path"]
            elif f_type == "t1_contrast":
                sample["t1c"] = file_info["t1c_path"]
            elif f_type == "t1_native":
                sample["t1n"] = file_info["t1_path"]
            elif f_type == "t2":
                sample["t2f"] = file_info["t2_path"]
            elif f_type == "t2w":
                sample["t2w"] = file_info["t2w_path"]
        
        # Grupăm modalitățile într-o listă pentru LoadImaged
        if all(k in sample for k in ["t1c", "t1n", "t2f", "t2w"]):
            sample["image"] = [sample["t1c"], sample["t1n"], sample["t2f"], sample["t2w"]]
            # Curățăm cheile temporare
            for k in ["t1c", "t1n", "t2f", "t2w"]:
                del sample[k]
            
            if "label" in sample or split == "test":
                items.append(sample)

    if len(items) == 0:
        raise RuntimeError(f"Nu am gasit samples pentru split-ul {split} in {json_path}.")
    
    print(f"Loaded {len(items)} samples for {split}.")
    return items

def remap_seg_0_1_2_4_to_0_1_2_3(x):
    # BraTS seg e de obicei {0,1,2,4}. Pentru training multi-class e mai comod {0,1,2,3}.
    # x poate fi numpy sau torch; MONAI trece de obicei numpy înainte de EnsureTyped.
    import numpy as np
    x = np.asarray(x)
    x = x.copy()
    x[x == 4] = 3
    return x

train_transforms = Compose([
    # 1) Load: dacă "image" e listă de path-uri => MONAI le încarcă și le stack-uiește pe canal (C=4)
    LoadImaged(keys=["image", "label"]),

    # 2) pune channel first: image -> [4, D, H, W], label -> [1, D, H, W]
    EnsureChannelFirstd(keys=["image", "label"]),

    # 3) uniformizează orientarea (foarte util să eviți mismatch)
    Orientationd(keys=["image", "label"], axcodes="RAS"),

    # 4) remap etichete 4->3 (opțional dar recomandat)
    Lambdad(keys="label", func=remap_seg_0_1_2_4_to_0_1_2_3),

    # 5) normalizezi intensitatea pe nonzero, channel-wise (bun pentru MRI)
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

    # 6) patch sampling: focalizăm mai mult pe tumoră (SOTA improvement)
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96), 
        pos=3, neg=1,           # Am crescut raportul la 3:1 pentru a vedea mai multă tumoră
        num_samples=2,          
        image_key="image",
        image_threshold=0
    ),

    # 7) augmentări simple și safe
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

    RandAffined(
        keys=["image", "label"],
        prob=0.2,
        rotate_range=(0.1, 0.1, 0.1),
        translate_range=(10, 10, 10),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest"),
        padding_mode="border",
    ),

    # 8) treci în torch.Tensor + metadata ok
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Lambdad(keys="label", func=remap_seg_0_1_2_4_to_0_1_2_3),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])


# --------- Dataset / Loader ---------

if __name__ == "__main__":
    # 1. Pregătești lista de fișiere din JSON
    base_dir = Path(__file__).parent.parent.parent
    json_path = base_dir / "dataset" / "BRATS" / "brats_metadata_splits.json"
    
    train_files = build_brats2024_list_from_json(str(json_path), split="train")

    # 2. Creezi Dataset-ul MONAI
    train_ds = Dataset(data=train_files, transform=train_transforms)

    # 3. DataLoader pentru training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    # Test:
    for batch in train_loader:
        print(batch["image"].shape)
        print(batch["label"].shape)
        break