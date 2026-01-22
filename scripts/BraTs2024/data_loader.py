import json
import os
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset, NibabelReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, RandCropByPosNegLabeld, RandFlipd,
    EnsureTyped, Lambdad,
    RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, RandBiasFieldd, RandAdjustContrastd,
    RandSimulateLowResolutiond, RandGaussianSmoothd, RandZoomd,
    RandKSpaceSpikeNoised,
)

# Fallback pentru RandChannelDropoutd dacă versiunea MONAI instalată nu îl exportă
try:
    from monai.transforms import RandChannelDropoutd
except Exception:
    import numpy as np
    import torch
    from monai.transforms import MapTransform, RandomizableTransform

    class RandChannelDropoutd(RandomizableTransform, MapTransform):
        """
        Fallback simplu pentru a elimina (zero) canale/modalități ale imaginii.
        Comportament: pentru fiecare canal din `channel_idxs`, cu probabilitate `prob`
        îl setează la zero. Folosește RNG intern pentru reproducibilitate cu seed MONAI.
        """
        def __init__(self, keys, prob=0.1, channel_idxs=None, allow_missing_keys=False):
            MapTransform.__init__(self, keys, allow_missing_keys)
            RandomizableTransform.__init__(self)
            self.prob = prob
            self.channel_idxs = channel_idxs

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                if key not in d:
                    continue
                img = d[key]
                # Convertim la numpy pentru manipulare uniformă
                was_torch = isinstance(img, torch.Tensor)
                arr = img.detach().cpu().numpy() if was_torch else np.asarray(img)
                # asigurăm axa canalului ca prima dimensiune
                if arr.ndim < 4:
                    # așteptăm shape (C, D, H, W)
                    pass

                ccount = arr.shape[0]
                idxs = self.channel_idxs if self.channel_idxs is not None else list(range(ccount))
                for c in idxs:
                    if c < 0 or c >= ccount:
                        continue
                    if self.R.random() < self.prob:
                        arr[c, ...] = 0

                d[key] = torch.as_tensor(arr) if was_torch else arr
            return d

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
    # --- 1. PREPARARE STRUCTURALĂ (PE CPU) ---
    # Încărcăm imaginea cu precizie float32 și eticheta cu uint8 pentru a economisi memorie
    LoadImaged(keys="image", reader=NibabelReader(dtype=np.float32)),
    LoadImaged(keys="label", reader=NibabelReader(dtype=np.uint8)),

    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    
    # Remap label ÎNAINTE de crop (0,1,2,4 -> 0,1,2,3)
    Lambdad(keys="label", func=remap_seg_0_1_2_4_to_0_1_2_3),
    
    # Asigurăm că datele sunt tensori PyTorch cu tipul corect DUPĂ manipulări numpy
    EnsureTyped(keys=["image", "label"]),

    # Normalizăm intensitatea înainte de crop (stabilizează distribuția)
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

    # --- 2. CROP DEVREME (CRITIC - reducem volumul ACUM) ---
    # De aici încolo lucrăm doar cu cuburi mici (96x96x96) = ~27x mai puțin calcul
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=4, neg=1, 
        num_samples=1,
        image_key="image",
        image_threshold=0
    ),

    # --- 3. AUGMENTĂRI PE PATCH-URI MICI (CPU, dar rapid acum!) ---
    
    # Augmentări Geometrice
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest"), prob=0.3),

    # Augmentări de Intensitate / Artefacte (acum pe patch-uri mici = rapid)
    RandBiasFieldd(keys=["image"], coeff_range=(0.2, 0.3), prob=0.3),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.2),
    RandKSpaceSpikeNoised(keys=["image"], prob=0.1, intensity_range=(5, 11)),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 4.5)),
    
    # Modality Dropout
    RandChannelDropoutd(keys=["image"], channel_idxs=[0, 1, 2, 3], prob=0.2),
    
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    # Încărcăm imaginea cu precizie float32 și eticheta cu uint8 pentru a economisi memorie
    LoadImaged(keys="image", reader=NibabelReader(dtype=np.float32)),
    LoadImaged(keys="label", reader=NibabelReader(dtype=np.uint8)),

    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Lambdad(keys="label", func=remap_seg_0_1_2_4_to_0_1_2_3),
    
    # Asigurăm că datele sunt tensori PyTorch cu tipul corect DUPĂ manipulări numpy
    EnsureTyped(keys=["image", "label"]),

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