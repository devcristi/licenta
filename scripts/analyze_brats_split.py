import json
import pandas as pd
from collections import defaultdict

# Load the JSON file
json_file = r"D:\study\licenta\creier\dataset\BRATS\brats_metadata.json"

print("Loading JSON file...")
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Analyze the data
training_patients = set()
validation_patients = set()
training_visits = []
validation_visits = []

for entry in data:
    patient_id = entry['patient_id']
    is_trainable = entry['is_trainable']
    
    if is_trainable == 1:
        training_patients.add(patient_id)
        training_visits.append(entry)
    else:
        validation_patients.add(patient_id)
        validation_visits.append(entry)

# Calculate statistics
stats = {
    "summary": {
        "total_records": len(data),
        "training_records": len(training_visits),
        "validation_records": len(validation_visits),
        "unique_training_patients": len(training_patients),
        "unique_validation_patients": len(validation_patients),
        "unique_total_patients": len(training_patients | validation_patients),
        "patients_in_both_sets": len(training_patients & validation_patients)
    },
    "training_data": {
        "unique_patient_ids": sorted(list(training_patients)),
        "total_visits": len(training_visits),
        "visits_per_patient": {}
    },
    "validation_data": {
        "unique_patient_ids": sorted(list(validation_patients)),
        "total_visits": len(validation_visits),
        "visits_per_patient": {}
    }
}

# Count visits per patient
for entry in training_visits:
    patient_id = entry['patient_id']
    if patient_id not in stats['training_data']['visits_per_patient']:
        stats['training_data']['visits_per_patient'][patient_id] = 0
    stats['training_data']['visits_per_patient'][patient_id] += 1

for entry in validation_visits:
    patient_id = entry['patient_id']
    if patient_id not in stats['validation_data']['visits_per_patient']:
        stats['validation_data']['visits_per_patient'][patient_id] = 0
    stats['validation_data']['visits_per_patient'][patient_id] += 1

# Create a detailed breakdown
print("\n" + "="*80)
print("BRATS DATASET SPLIT ANALYSIS")
print("="*80)

print(f"\n📊 SUMMARY STATISTICS:")
print(f"   Total records (visits): {stats['summary']['total_records']}")
print(f"   Training records: {stats['summary']['training_records']}")
print(f"   Validation records: {stats['summary']['validation_records']}")
print(f"\n   Unique training patients: {stats['summary']['unique_training_patients']}")
print(f"   Unique validation patients: {stats['summary']['unique_validation_patients']}")
print(f"   Total unique patients: {stats['summary']['unique_total_patients']}")
print(f"   Patients in both sets: {stats['summary']['patients_in_both_sets']}")

print(f"\n🏥 TRAINING DATA:")
print(f"   Patients: {stats['summary']['unique_training_patients']}")
print(f"   Total visits: {stats['summary']['training_records']}")
print(f"   Avg visits per patient: {stats['summary']['training_records'] / stats['summary']['unique_training_patients']:.2f}")

print(f"\n✅ VALIDATION DATA:")
print(f"   Patients: {stats['summary']['unique_validation_patients']}")
print(f"   Total visits: {stats['summary']['validation_records']}")
print(f"   Avg visits per patient: {stats['summary']['validation_records'] / stats['summary']['unique_validation_patients']:.2f}")

# Show visits per patient for training
print(f"\n📋 TRAINING PATIENTS - VISITS BREAKDOWN:")
visit_counts = stats['training_data']['visits_per_patient']
visit_distribution = defaultdict(int)
for patient_id, count in visit_counts.items():
    visit_distribution[count] += 1

for visits, patients_count in sorted(visit_distribution.items()):
    print(f"   {patients_count} patients with {visits} visit(s)")

# Show visits per patient for validation
print(f"\n📋 VALIDATION PATIENTS - VISITS BREAKDOWN:")
visit_counts_val = stats['validation_data']['visits_per_patient']
visit_distribution_val = defaultdict(int)
for patient_id, count in visit_counts_val.items():
    visit_distribution_val[count] += 1

for visits, patients_count in sorted(visit_distribution_val.items()):
    print(f"   {patients_count} patients with {visits} visit(s)")

# Save stats to JSON
stats_file = r"D:\study\licenta\creier\dataset\BRATS\brats_split_statistics.json"
print(f"\n\nSaving statistics to {stats_file}...")
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

# Save as CSV for easy viewing
csv_file = r"D:\study\licenta\creier\dataset\BRATS\brats_split_summary.csv"
print(f"Saving summary to {csv_file}...")

summary_data = {
    'Metric': [
        'Total Records (Visits)',
        'Training Records',
        'Validation Records',
        'Unique Training Patients',
        'Unique Validation Patients',
        'Total Unique Patients',
        'Patients in Both Sets',
        'Avg Training Visits per Patient',
        'Avg Validation Visits per Patient'
    ],
    'Value': [
        stats['summary']['total_records'],
        stats['summary']['training_records'],
        stats['summary']['validation_records'],
        stats['summary']['unique_training_patients'],
        stats['summary']['unique_validation_patients'],
        stats['summary']['unique_total_patients'],
        stats['summary']['patients_in_both_sets'],
        f"{stats['summary']['training_records'] / stats['summary']['unique_training_patients']:.2f}",
        f"{stats['summary']['validation_records'] / stats['summary']['unique_validation_patients']:.2f}"
    ]
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(csv_file, index=False)

print("\n✅ Done! Statistics saved.")
