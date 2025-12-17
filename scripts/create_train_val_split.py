import json
import random
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Load the JSON file
json_file = r"D:\study\licenta\creier\dataset\BRATS\brats_metadata.json"

print("Loading JSON file...")
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get only training patients
training_data = [entry for entry in data if entry['is_trainable'] == 1]
validation_data = [entry for entry in data if entry['is_trainable'] == 0]

# Extract unique training patients
unique_training_patients = sorted(list(set([entry['patient_id'] for entry in training_data])))

print(f"Total training patients: {len(unique_training_patients)}")

# Split 80/20 on patient level
split_idx = int(len(unique_training_patients) * 0.8)
train_patients = set(unique_training_patients[:split_idx])
val_split_patients = set(unique_training_patients[split_idx:])

print(f"Train set patients: {len(train_patients)} ({len(train_patients)/len(unique_training_patients)*100:.1f}%)")
print(f"Val split set patients: {len(val_split_patients)} ({len(val_split_patients)/len(unique_training_patients)*100:.1f}%)")

# Create new JSON with split information
new_data = []

# Process training data
for entry in training_data:
    new_entry = entry.copy()
    if entry['patient_id'] in train_patients:
        new_entry['final_split'] = 'train'
    else:
        new_entry['final_split'] = 'val_from_train'
    new_data.append(new_entry)

# Keep validation data as is
for entry in validation_data:
    new_entry = entry.copy()
    new_entry['final_split'] = 'test'
    new_data.append(new_entry)

# Count records per split
split_counts = defaultdict(int)
split_visits = defaultdict(list)
split_patients = defaultdict(set)

for entry in new_data:
    split = entry['final_split']
    split_counts[split] += 1
    split_visits[split].append(entry)
    split_patients[split].add(entry['patient_id'])

# Print summary
print("\n" + "="*80)
print("FINAL DATASET SPLIT (80/20 SPLIT ON TRAINING PATIENTS)")
print("="*80)

print(f"\n🔵 TRAIN (80% of original training patients):")
print(f"   Patients: {len(split_patients['train'])}")
print(f"   Visits: {split_counts['train']}")

print(f"\n🟡 VALIDATION (20% of original training patients):")
print(f"   Patients: {len(split_patients['val_from_train'])}")
print(f"   Visits: {split_counts['val_from_train']}")

print(f"\n🔴 TEST (original validation set):")
print(f"   Patients: {len(split_patients['test'])}")
print(f"   Visits: {split_counts['test']}")

print(f"\n📊 TOTAL:")
print(f"   Patients: {len(split_patients['train']) + len(split_patients['val_from_train']) + len(split_patients['test'])}")
print(f"   Visits: {split_counts['train'] + split_counts['val_from_train'] + split_counts['test']}")

# Save new JSON
output_json_file = r"D:\study\licenta\creier\dataset\BRATS\brats_metadata_splits.json"
print(f"\n\nSaving to {output_json_file}...")
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

# Create summary statistics
stats = {
    "split_methodology": "80/20 split on patient level from training set",
    "train": {
        "patients": len(split_patients['train']),
        "visits": split_counts['train'],
        "avg_visits_per_patient": split_counts['train'] / len(split_patients['train']) if split_patients['train'] else 0,
        "patient_ids": sorted(list(split_patients['train']))
    },
    "validation": {
        "patients": len(split_patients['val_from_train']),
        "visits": split_counts['val_from_train'],
        "avg_visits_per_patient": split_counts['val_from_train'] / len(split_patients['val_from_train']) if split_patients['val_from_train'] else 0,
        "patient_ids": sorted(list(split_patients['val_from_train']))
    },
    "test": {
        "patients": len(split_patients['test']),
        "visits": split_counts['test'],
        "avg_visits_per_patient": split_counts['test'] / len(split_patients['test']) if split_patients['test'] else 0,
        "patient_ids": sorted(list(split_patients['test']))
    }
}

stats_file = r"D:\study\licenta\creier\dataset\BRATS\brats_splits_statistics.json"
print(f"Saving statistics to {stats_file}...")
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print("✅ Done!")
