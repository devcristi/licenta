import json
from collections import Counter

def verify_split(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    train_pids = {d["patient_id"] for d in data if d["split"] == "train"}
    val_pids = {d["patient_id"] for d in data if d["split"] == "val"}
    
    # 1. Zero Leakage Check
    intersection = train_pids.intersection(val_pids)
    print(f"--- Verificare Leakage ---")
    if not intersection:
        print("✅ SUCCESS: Zero leakage! Niciun pacient nu apare in ambele seturi.")
    else:
        print(f"❌ ERROR: Leakage detectat pentru pacientii: {intersection}")
    
    # 2. Distributie Vizite
    train_visits = [d["patient_id"] for d in data if d["split"] == "train"]
    val_visits = [d["patient_id"] for d in data if d["split"] == "val"]
    
    train_counts = Counter(train_visits)
    val_counts = Counter(val_visits)
    
    print(f"\n--- Statistici Vizite ---")
    print(f"Train: {len(train_visits)} vizite de la {len(train_pids)} pacienti.")
    print(f"Val:   {len(val_visits)} vizite de la {len(val_pids)} pacienti.")
    
    print(f"\n--- Top Pacienti in Val (Dominanta) ---")
    for pid, count in val_counts.most_common(5):
        percentage = (count / len(val_visits)) * 100
        print(f"Patient {pid}: {count} vizite ({percentage:.1f}% din Val)")

if __name__ == "__main__":
    JSON_PATH = r"D:\study\licenta\creier\dataset\LUMIERE\lumiere_metadata.json"
    verify_split(JSON_PATH)
