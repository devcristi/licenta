import json
import os
from pathlib import Path
import numpy as np
import torch
from monai.data import DataLoader, Dataset, NibabelReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, RandCropByPosNegLabeld, RandFlipd,
    EnsureTyped, Lambdad, SpatialPadd, RandZoomd,
    RandGaussianNoised, RandBiasFieldd, RandAdjustContrastd,
    RandSimulateLowResolutiond, RandKSpaceSpikeNoised, RandShiftIntensityd
)

# --- CONFIGURAȚIE GLOBALĂ ---
PATCH_SIZE = (128, 128, 128) # Optimizat pentru 32GB VRAM

# Fallback pentru RandChannelDropoutd
try:
    from monai.transforms import RandChannelDropoutd
except Exception:
    from monai.transforms import MapTransform, RandomizableTransform
    class RandChannelDropoutd(RandomizableTransform, MapTransform):
        def __init__(self, keys, prob=0.1, channel_idxs=None, allow_missing_keys=False):
            MapTransform.__init__(self, keys, allow_missing_keys)
            RandomizableTransform.__init__(self)
            self.prob = prob
            self.channel_idxs = channel_idxs

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                if key not in d: continue
                img = d[key]
                was_torch = isinstance(img, torch.Tensor)
                arr = img.detach().cpu().numpy() if was_torch else np.asarray(img)
                ccount = arr.shape[0]
                idxs = self.channel_idxs if self.channel_idxs is not None else list(range(ccount))
                for c in idxs:
                    if 0 <= c < ccount and self.R.random() < self.prob:
                        arr[c, ...] = 0
                d[key] = torch.as_tensor(arr) if was_torch else arr
            return d

def build_brats2024_list_from_json(json_path: str, split: str = "train"):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    items = []
    for entry in data:
        if entry.get("final_split") != split:
            continue
            
        sample = {"subject_id": entry["subject_id"], "patient_id": entry["patient_id"]}
        
        for file_info in entry["all_files"]:
            f_type = file_info["type"]
            if f_type == "segmentation": sample["label"] = file_info["seg_path"]
            elif f_type == "t1_contrast": sample["t1c"] = file_info["t1c_path"]
            elif f_type == "t1_native": sample["t1n"] = file_info["t1_path"]
            elif f_type == "t2": sample["t2f"] = file_info["t2_path"]
            elif f_type == "t2w": sample["t2w"] = file_info["t2w_path"]
        
        if all(k in sample for k in ["t1c", "t1n", "t2f", "t2w"]):
            sample["image"] = [sample["t1c"], sample["t1n"], sample["t2f"], sample["t2w"]]
            for k in ["t1c", "t1n", "t2f", "t2w"]: del sample[k]
            if "label" in sample or split == "test":
                items.append(sample)

    print(f"✅ Loaded {len(items)} samples for {split}.")
    return items

def remap_seg_0_1_2_4_to_0_1_2_3(x):
    x = np.asarray(x)
    x = x.copy()
    x[x == 4] = 3
    return x

# --- TRANSFORMĂRI ANTRENARE ---
train_transforms = Compose([
    # 1. Încărcare și Orientare
    LoadImaged(keys=["image", "label"], reader=NibabelReader()),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    
    # 2. Resampling Spațial (Standardizează 155 vs 130 felii la 1mm)
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    
    # 3. Padding (Rezolvă volumele mai mici decât patch-ul de 128)
    SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE, mode="constant"),
    
    Lambdad(keys="label", func=remap_seg_0_1_2_4_to_0_1_2_3),
    EnsureTyped(keys=["image", "label"]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

    # 4. Crop și Augmentări
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=PATCH_SIZE, pos=4, neg=1, num_samples=1,
        image_key="image", image_threshold=0
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest"), prob=0.3),
    RandBiasFieldd(keys=["image"], coeff_range=(0.2, 0.3), prob=0.3),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandChannelDropoutd(keys=["image"], channel_idxs=[0, 1, 2, 3], prob=0.2),
    EnsureTyped(keys=["image", "label"]),
])

# --- TRANSFORMĂRI VALIDARE ---
val_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=NibabelReader()),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    # Oglindește train-ul pentru consistență
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE, mode="constant"),
    Lambdad(keys="label", func=remap_seg_0_1_2_4_to_0_1_2_3),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])