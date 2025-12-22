#!/usr/bin/env python3
"""
Script pentru analiza metadatelor BraTS dataset.
Citeste Excel-ul si extrage informatii despre pacienti si scan-urile lor.
"""

import pandas as pd
import os
from collections import defaultdict

# Path-ul catre fișierul Excel
EXCEL_FILE = r'D:\study\licenta\creier\dataset\BRATS\BraTS-PTG supplementary demographic information and metadata.xlsx'

def analyze_brats_metadata():
    """Analizează metadatele BraTS din Excel."""
    
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ Fișierul nu există: {EXCEL_FILE}")
        return
    
    print(f"📂 Citesc fișierul: {EXCEL_FILE}\n")
    
    try:
        # Încearcă să citească Excel-ul
        df = pd.read_excel(EXCEL_FILE)
        
        print(f"✅ Excel citit cu succes!")
        print(f"📊 Dimensiuni: {df.shape[0]} rânduri, {df.shape[1]} coloane")
        print(f"\n📋 Coloane disponibile:")
        for i, col in enumerate(df.columns):
            print(f"   {i}: {col}")
        
        print(f"\n📄 Primele 5 rânduri:")
        print(df.head())
        
        # Incearcă să găsești coloane relevante (id, patient, scan, etc)
        print(f"\n" + "="*80)
        print("ANALIZA PACIENȚI ȘI SCAN-URI")
        print("="*80)
        
        # Cautam coloane cu cuvinte-cheie
        id_col = None
        patient_col = None
        scan_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['id', 'patient', 'brats']):
                if id_col is None:
                    id_col = col
            if any(keyword in col_lower for keyword in ['scan', 'timepoint', 'visit']):
                if scan_col is None:
                    scan_col = col
        
        if id_col is None:
            print("\n⚠️ Nu am găsit coloană ID/Patient/BraTS")
            print("Încerc să folosesc prima coloană...")
            id_col = df.columns[0]
        
        print(f"\n🔍 Coloană ID folosită: '{id_col}'")
        if scan_col:
            print(f"🔍 Coloană Scan folosită: '{scan_col}'")
        
        # Parsează ID-uri și numără pacienți/scan-uri
        patients_dict = defaultdict(list)
        
        for idx, row in df.iterrows():
            patient_id_str = str(row[id_col]).strip()
            
            # Extrage ID pacient (BraTS-GLI-XXXXX)
            if 'BraTS' in patient_id_str or 'GLI' in patient_id_str:
                # Format: BraTS-GLI-XXXXX-YYY
                parts = patient_id_str.split('-')
                if len(parts) >= 4:
                    # BraTS, GLI, XXXXX, YYY
                    patient_id = f"{parts[0]}-{parts[1]}-{parts[2]}"  # BraTS-GLI-XXXXX
                    scan_id = parts[3] if len(parts) > 3 else "unknown"  # YYY
                    
                    patients_dict[patient_id].append(scan_id)
        
        if patients_dict:
            print(f"\n✅ Pacienți unici găsiți: {len(patients_dict)}")
            
            # Statistici despre scan-uri
            scan_counts = defaultdict(int)
            for patient_id, scans in patients_dict.items():
                num_scans = len(set(scans))  # Scan-uri unice
                scan_counts[num_scans] += 1
            
            print(f"\n📊 Distribuția scan-urilor per pacient:")
            for num_scans in sorted(scan_counts.keys()):
                count = scan_counts[num_scans]
                print(f"   • {num_scans} scan-uri: {count} pacienți")
            
            # Exemple
            print(f"\n📋 Exemple de pacienți și scan-urile lor:")
            for i, (patient_id, scans) in enumerate(sorted(patients_dict.items())[:10]):
                unique_scans = sorted(set(scans))
                print(f"   {patient_id}:")
                for scan in unique_scans:
                    count = scans.count(scan)
                    print(f"      └─ {scan}")
                print()
            
            if len(patients_dict) > 10:
                print(f"   ... și {len(patients_dict) - 10} alți pacienți\n")
            
            # CSV export
            print(f"\n💾 Export detaliat:")
            export_data = []
            for patient_id, scans in sorted(patients_dict.items()):
                unique_scans = sorted(set(scans))
                export_data.append({
                    'Patient': patient_id,
                    'Number_of_Scans': len(unique_scans),
                    'Scan_IDs': ', '.join(unique_scans)
                })
            
            export_df = pd.DataFrame(export_data)
            csv_output = r'D:\study\licenta\creier\scripts\brats_patients_summary.csv'
            export_df.to_csv(csv_output, index=False)
            print(f"   ✅ Salvat la: {csv_output}")
            
            print(f"\n" + "="*80)
            print(f"REZUMAT FINAL:")
            print(f"  • Total pacienți unici: {len(patients_dict)}")
            print(f"  • Total scan-uri: {sum(len(set(scans)) for scans in patients_dict.values())}")
            print(f"  • Media scan-uri/pacient: {sum(len(set(scans)) for scans in patients_dict.values()) / len(patients_dict):.2f}")
            print("="*80)
        
        else:
            print("\n⚠️ Nu am putut extrage date în format BraTS.")
            print("\n📊 Primele 20 de rânduri din dataset:")
            print(df.head(20).to_string())
    
    except Exception as e:
        print(f"❌ Eroare la citirea fișierului: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    analyze_brats_metadata()
