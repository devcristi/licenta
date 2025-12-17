import os
import re
import json
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Define base paths
base_path = r"D:\study\licenta\creier\dataset\BRATS"
training_path = os.path.join(base_path, "BraTS2024-BraTS-GLI-TrainingData", "training_data1_v2")
additional_training_path = os.path.join(base_path, "BraTS2024-BraTS-GLI-AdditionalTrainingData", "training_data_additional")
validation_path = os.path.join(base_path, "BraTS2024-BraTS-GLI-ValidationData", "validation_data")

excel_file = os.path.join(base_path, "BraTS-PTG supplementary demographic information and metadata.xlsx")

def parse_subject_id(folder_name):
    """Extract patient_id and visit_id from folder name like BraTS-GLI-00005-100"""
    match = re.match(r'BraTS-GLI-(\d+)-(\d+)', folder_name)
    if match:
        return folder_name, match.group(1), match.group(2)
    return None, None, None

def scan_folder(folder_path, is_trainable):
    """Scan a folder and return list of patient-visit records"""
    records = []
    if os.path.exists(folder_path):
        for folder_name in os.listdir(folder_path):
            folder_full_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(folder_full_path):
                subject_id, patient_id, visit_id = parse_subject_id(folder_name)
                if subject_id:
                    records.append({
                        'subject_id': subject_id,
                        'patient_id': patient_id,
                        'visit_id': visit_id,
                        'is_trainable': is_trainable,
                        'folder_type': folder_path.split('\\')[-2:],
                        'file_path': folder_full_path
                    })
    return records

# Scan all folders
print("Scanning training data...")
records = scan_folder(training_path, 1)
print(f"Found {len(records)} training records")

print("Scanning additional training data...")
additional_records = scan_folder(additional_training_path, 1)
print(f"Found {len(additional_records)} additional training records")
records.extend(additional_records)

print("Scanning validation data...")
validation_records = scan_folder(validation_path, 0)
print(f"Found {len(validation_records)} validation records")
records.extend(validation_records)

# Sort by patient_id and visit_id
records.sort(key=lambda x: (int(x['patient_id']), int(x['visit_id'])))

# Create DataFrame
df = pd.DataFrame(records)
print(f"\nTotal records: {len(df)}")
print(df.head(10))

# Load existing Excel file
print(f"\nLoading Excel file: {excel_file}")
try:
    existing_df = pd.read_excel(excel_file)
    print(f"Existing Excel has {len(existing_df)} rows")
except:
    print("No existing Excel file found, creating new one")
    existing_df = pd.DataFrame()

# Add new columns to the records dataframe
output_df = df[['subject_id', 'patient_id', 'visit_id', 'is_trainable']].copy()

# Save to Excel
print(f"\nSaving to Excel: {excel_file}")
with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
    output_df.to_excel(writer, sheet_name='BraTS Data', index=False)

print("Excel file saved!")

# Create JSON with file paths
def get_files_in_folder(folder_path):
    """Recursively get all files in a folder and categorize them"""
    files = []
    
    if os.path.exists(folder_path):
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                file_full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_full_path, folder_path)
                
                # Determine file type based on filename
                file_type = 'unknown'
                if 't1n' in filename.lower() or '-t1.nii' in filename.lower():
                    file_type = 't1_native'
                elif 't1c' in filename.lower() or '-t1c.nii' in filename.lower():
                    file_type = 't1_contrast'
                elif 't2w' in filename.lower():
                    file_type = 't2w'
                elif 't2' in filename.lower() and 'flair' not in filename.lower():
                    file_type = 't2'
                elif 'flair' in filename.lower():
                    file_type = 'flair'
                elif 'seg' in filename.lower():
                    file_type = 'segmentation'
                
                # Create a short key for this file type
                if file_type == 't1_native':
                    key = 't1_path'
                elif file_type == 't1_contrast':
                    key = 't1c_path'
                elif file_type == 't2w':
                    key = 't2w_path'
                elif file_type == 't2':
                    key = 't2_path'
                elif file_type == 'flair':
                    key = 'flair_path'
                elif file_type == 'segmentation':
                    key = 'seg_path'
                else:
                    key = f'{file_type}_path'
                
                file_obj = {
                    'filename': filename,
                    'relative_path': relative_path,
                    'type': file_type
                }
                
                # Add the typed path key
                file_obj[key] = file_full_path
                
                files.append(file_obj)
    
    return files

json_data = []
for rec in records:
    files = get_files_in_folder(rec['file_path'])
    
    entry = {
        'subject_id': rec['subject_id'],
        'patient_id': rec['patient_id'],
        'visit_id': rec['visit_id'],
        'is_trainable': rec['is_trainable'],
        'folder_path': rec['file_path'],
        'data_type': 'training' if rec['is_trainable'] == 1 else 'validation',
        'all_files': files
    }
    
    json_data.append(entry)

json_file = os.path.join(base_path, "brats_metadata.json")
print(f"\nSaving JSON file: {json_file}")
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"JSON file saved!")
print(f"\nSummary:")
print(f"Total records: {len(json_data)}")
print(f"Training records: {len([r for r in json_data if r['is_trainable'] == 1])}")
print(f"Validation records: {len([r for r in json_data if r['is_trainable'] == 0])}")
