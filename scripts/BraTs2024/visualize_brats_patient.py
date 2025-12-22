#!/usr/bin/env python3
"""
BraTS Patient Visualization Script
Displays BraTS-GLI-00006 (scans 100 and 101) with all modalities and segmentation overlays
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from pathlib import Path

# Setup paths
brats_root = Path(r'D:\study\licenta\creier\dataset\BRATS\BraTS2024-BraTS-GLI-TrainingData\training_data1_v2')
patient_ids = ['BraTS-GLI-00006-100']

print(f"BRATS root: {brats_root}")
print(f"Root exists: {brats_root.exists()}")


def load_brats_patient(patient_id, brats_root):
    """Load all 4 MRI sequences and segmentation for a BraTS patient"""
    patient_dir = brats_root / patient_id
    
    modalities = {
        't1n': f'{patient_id}-t1n.nii.gz',  # T1 native
        't1c': f'{patient_id}-t1c.nii.gz',  # T1 with contrast
        't2w': f'{patient_id}-t2w.nii.gz',  # T2 weighted
        't2f': f'{patient_id}-t2f.nii.gz',  # T2-FLAIR
        'seg': f'{patient_id}-seg.nii.gz'   # Segmentation
    }
    
    data = {}
    for mod_name, filename in modalities.items():
        filepath = patient_dir / filename
        if filepath.exists():
            img = nib.load(str(filepath))
            data[mod_name] = img.get_fdata()
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} not found")
    
    return data


def get_middle_slice(data_shape):
    """Get the middle slice in Z axis"""
    return data_shape[2] // 2


def normalize_slice(slice_2d):
    """Normalize a 2D slice for display"""
    p2, p98 = np.percentile(slice_2d[slice_2d > 0], [2, 98]) if np.any(slice_2d > 0) else (0, 1)
    normalized = (slice_2d - p2) / (p98 - p2 + 1e-6)
    return np.clip(normalized, 0, 1)


# Load data for both scans
all_scans_data = {}
for patient_id in patient_ids:
    print(f"\nLoading {patient_id}...")
    all_scans_data[patient_id] = load_brats_patient(patient_id, brats_root)

# Get middle slices
middle_slices = {}
for patient_id, data in all_scans_data.items():
    if 't1c' in data:
        middle_idx = get_middle_slice(data['t1c'].shape)
        middle_slices[patient_id] = middle_idx
        print(f"{patient_id}: middle slice = {middle_idx} / {data['t1c'].shape[2]}")

# Create comprehensive visualization with better layout
fig = plt.figure(figsize=(32, 20))
gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)

for scan_idx, patient_id in enumerate(patient_ids):
    data = all_scans_data[patient_id]
    middle_idx = middle_slices[patient_id]
    
    scan_label = patient_id.split('-')[-1]  # Get "100" or "101"
    
    # ===== ROW 1, COL 1: T1c (with contrast) =====
    ax1 = fig.add_subplot(gs[0, 0])
    if 't1c' in data:
        img_slice = data['t1c'][:, :, middle_idx]
        img_rotated = np.rot90(img_slice, k=1)  # Rotate 90 degrees left
        ax1.imshow(img_rotated, cmap='gray')
        ax1.set_title('T1 Contrast Enhanced', fontsize=18, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # ===== ROW 1, COL 2: T2-FLAIR =====
    ax2 = fig.add_subplot(gs[0, 1])
    if 't2f' in data:
        img_slice = data['t2f'][:, :, middle_idx]
        img_rotated = np.rot90(img_slice, k=1)  # Rotate 90 degrees left
        ax2.imshow(img_rotated, cmap='gray')
        ax2.set_title('T2-FLAIR', fontsize=18, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # ===== ROW 2, COL 1: Segmentation Mask Only =====
    ax3 = fig.add_subplot(gs[1, 0])
    if 'seg' in data:
        seg_slice = data['seg'][:, :, middle_idx]
        
        # Create colored segmentation visualization
        seg_colored = np.zeros((*seg_slice.shape, 3))
        
        # Necrotic core (label 1) - RED
        seg_colored[seg_slice == 1] = [1, 0, 0]
        # Edema (label 2) - BLUE
        seg_colored[seg_slice == 2] = [0, 0, 1]
        # Enhancing (label 4) - GREEN
        seg_colored[seg_slice == 4] = [0, 1, 0]
        
        seg_rotated = np.rot90(seg_colored, k=1)  # Rotate 90 degrees left
        ax3.imshow(seg_rotated)
        ax3.set_title('Segmentation Mask', fontsize=18, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # ===== ROW 2, COL 2: Overlay on T1c =====
    ax4 = fig.add_subplot(gs[1, 1])
    if 'seg' in data and 't1c' in data:
        t1c_slice = data['t1c'][:, :, middle_idx]
        seg_slice = data['seg'][:, :, middle_idx]
        
        # Background: T1c grayscale
        ax4.imshow(np.rot90(t1c_slice, k=1), cmap='gray', alpha=0.85)
        
        # Overlay: colored segmentation with transparency
        seg_colored_alpha = np.zeros((*seg_slice.shape, 4))
        seg_colored_alpha[seg_slice == 1] = [1, 0, 0, 0.75]  # Red - Necrotic
        seg_colored_alpha[seg_slice == 2] = [0, 0, 1, 0.65]  # Blue - Edema
        seg_colored_alpha[seg_slice == 4] = [0, 1, 0, 0.75]  # Green - Enhancing
        
        seg_rotated = np.rot90(seg_colored_alpha, k=1)  # Rotate 90 degrees left
        ax4.imshow(seg_rotated)
        ax4.set_title('Tumor Overlay', fontsize=18, fontweight='bold', pad=10)
    ax4.axis('off')

plt.savefig(r'd:\study\licenta\creier\scripts\brats_00006_100_visualization.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved to: brats_00006_100_visualization.png")

plt.show()
print("✅ Done!")
