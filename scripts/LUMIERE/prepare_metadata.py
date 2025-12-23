import os
import json
from pathlib import Path

def scan_lumiere(base_path):
    base_path = Path(base_path)
    imaging_path = base_path / "Imaging"
    
    dataset = []
    
    # Iterăm prin pacienți
    patients = [d for d in imaging_path.iterdir() if d.is_dir() and d.name.startswith("Patient-")]
    
    for patient_dir in patients:
        patient_id = patient_dir.name
        # Iterăm prin vizite (weeks)
        weeks = [d for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith("week-")]
        
        for week_dir in weeks:
            week_id = week_dir.name
            
            # Modalități în spațiul ATLAS (înregistrate și skull-stripped)
            atlas_path = week_dir / "DeepBraTumIA-segmentation" / "atlas"
            
            t1c = atlas_path / "skull_strip" / "ct1_skull_strip.nii.gz"
            t1n = atlas_path / "skull_strip" / "t1_skull_strip.nii.gz"
            t2 = atlas_path / "skull_strip" / "t2_skull_strip.nii.gz"
            flair = atlas_path / "skull_strip" / "flair_skull_strip.nii.gz"
            
            # Label în spațiul ATLAS
            label = atlas_path / "segmentation" / "seg_mask.nii.gz"
            
            # Verificăm dacă toate fișierele există
            if all(p.exists() for p in [t1c, t1n, t2, flair, label]):
                dataset.append({
                    "subject_id": f"{patient_id}_{week_id}",
                    "patient_id": patient_id,
                    "week": week_id,
                    "image": [str(t1c), str(t1n), str(t2), str(flair)],
                    "label": str(label)
                })
    
    return dataset

if __name__ == "__main__":
    LUMIERE_ROOT = r"D:\study\licenta\creier\dataset\LUMIERE"
    data = scan_lumiere(LUMIERE_ROOT)
    
    output_path = Path(LUMIERE_ROOT) / "lumiere_metadata.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Am găsit {len(data)} vizite valide.")
    print(f"Metadatele au fost salvate în: {output_path}")
