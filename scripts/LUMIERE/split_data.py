import json
import random
from pathlib import Path

def split_lumiere(json_path, train_ratio=0.8):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Grupăm vizitele pe pacienți
    patient_groups = {}
    for entry in data:
        pid = entry["patient_id"]
        if pid not in patient_groups:
            patient_groups[pid] = []
        patient_groups[pid].append(entry)
    
    pids = list(patient_groups.keys())
    random.seed(42)
    random.shuffle(pids)
    
    split_idx = int(len(pids) * train_ratio)
    train_pids = set(pids[:split_idx])
    val_pids = set(pids[split_idx:])
    
    for entry in data:
        if entry["patient_id"] in train_pids:
            entry["split"] = "train"
        else:
            entry["split"] = "val"
            
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Split finalizat:")
    print(f" - Pacienți Train: {len(train_pids)} ({sum(1 for e in data if e['split'] == 'train')} vizite)")
    print(f" - Pacienți Val: {len(val_pids)} ({sum(1 for e in data if e['split'] == 'val')} vizite)")

if __name__ == "__main__":
    JSON_PATH = r"D:\study\licenta\creier\dataset\LUMIERE\lumiere_metadata.json"
    split_lumiere(JSON_PATH)
