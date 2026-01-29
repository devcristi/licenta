# 🧠 Bachelor thesis — Post-Treatment Adult Glioma (BraTS 2024) Segmentation + Longitudinal Analysis

Bachelor thesis codebase for **3D deep learning on post-treatment adult glioma MRI** using **BraTS 2024 (BraTS-GLI)**, with optional transfer/longitudinal experiments on **LUMIERE**.

The core goal is to build a **clean, reproducible PyTorch pipeline** for **multi-modal 3D segmentation**, with **patient-level splitting** to avoid leakage across multiple visits/timepoints.

---

## ✨ Project Highlights

- **3D multi-modal tumor/subregion segmentation** (baseline: 3D U-Net style models)
- **Patient-level split** (train/val), while training on **visits** (timepoints)
- **Inference-only test set** (official BraTS Validation: no `seg`)
- **Advanced Inference Pipeline**:
  - **Test-Time Augmentation (TTA)**: Spatial flipping on Coronal and Sagittal axes to stabilize predictions.
  - **Threshold Tuning**: Optimized decision threshold at **0.75** for improved precision on external clinical data.
  - **Post-processing**: Largest Connected Component (LCC) filtering to eliminate isolated noise.

---

## 📊 Dataset Statistics & Benchmarks

### 1. BraTS 2024 — BraTS-GLI (Post-Treatment)
- 🔵 **TRAIN (internal)**: 584 patients | 1,324 visits  
- 🟡 **VAL (internal)**: 147 patients | 297 visits  
- 🔴 **TEST (official BraTS validation, unlabeled)**: 87 patients | 188 visits  
- **Total**: 818 unique patients | 1,809 MRI sessions (visits)

### 2. LUMIERE External Validation (3-Fold CV Results)
Supervised metrics computed using the optimized 3D U-Net pipeline:

| Region | Dice Score (Mean ± Std) | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- |
| **WT (Whole Tumor)** | **0.7592 ± 0.0199** | **0.8503** | > 0.99 |
| **TC (Tumor Core)** | **0.7316 ± 0.0236** | **0.8391** | > 0.99 |
| **ET (Enhancing Tumor)** | **0.7030 ± 0.0242** | **0.7853** | > 0.99 |
| **GLOBAL MEAN** | **0.7313 ± 0.0222** | **0.8249** | **0.9978** |

- **HD95 WT (Mean)**: **20.00 mm** (stabilized via LCC and TTA).
### 3. Model Evolution: Baseline vs. Optimized Pipeline
This comparison highlights the performance jump from the initial fine-tuning attempt to the current 3-fold CV pipeline with TTA and LCC.

| Model Version | WT Dice | TC Dice | ET Dice | Mean Dice | HD95 WT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial 3D U-Net (BraTS weights)** | 0.7675 | 0.7217 | 0.6699 | 0.7197 | nan |
| **Optimized 3D U-Net (TTA + LCC)** | **0.7592** | **0.7316** | **0.7030** | **0.7313** | **20.00 mm** |
| **Improvement (%)** | **-1.08%*** | **+1.37%** | **+4.94%** | **+1.61%** | **Stabilized** |

*\*Note: The slight decrease in WT Dice is a trade-off for significantly higher precision in TC/ET and the stabilization of the HD95 metric from 'nan' to 20mm.*
### 1. LUMIERE External Validation (3-Fold CV Results)
Supervised metrics computed using the optimized 3D U-Net pipeline (TTA + LCC + Post-proc):

| Region | Dice Score (Mean ± Std) | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- |
| **WT (Whole Tumor)** | **0.7592 ± 0.0199** | **0.8503** | > 0.99 |
| **TC (Tumor Core)** | **0.7316 ± 0.0236** | **0.8391** | > 0.99 |
| **ET (Enhancing Tumor)** | **0.7030 ± 0.0242** | **0.7853** | > 0.99 |
| **GLOBAL MEAN** | **0.7313 ± 0.0222** | **0.8249** | **0.9978** |

- **HD95 WT (Mean)**: **20.00 mm** (stabilized via LCC and TTA).

### 2. Model Evolution: Baseline vs. Optimized Pipeline
Detailed comparison between the initial fine-tuning (without augmentations) and the current optimized pipeline:

| Model Version | WT Dice | TC Dice | ET Dice | Mean Dice | HD95 WT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial 3D U-Net (BraTS weights)** | **0.7675** | 0.7217 | 0.6699 | 0.7197 | nan |
| **Optimized 3D U-Net (TTA + LCC)** | 0.7592 | **0.7316** | **0.7030** | **0.7313** | **20.00 mm** |
| **Improvement (%)** | **-1.08%** | **+1.37%** | **+4.94%** | **+1.61%** | **Stabilized** |

> **Analysis**: The optimized model significantly outperforms the baseline in the **ET (Enhancing Tumor)** region (+4.94%), which is the most clinically critical area. While WT Dice saw a minor decrease due to the stricter 0.75 threshold, the overall Mean Dice improved and the HD95 metric was successfully stabilized from 'nan' to 20mm.

---
---

## 🔬 Architectural Roadmap (Battle of Models)

The project compares several SOTA architectures to improve boundary precision and sensitivity:
1. **3D U-Net (Optimized Baseline)**: Enhanced with Focal Tversky Loss.
2. **Attention 3D U-Net**: Integration of Attention Gates to focus on tumor boundaries.
3. **SegResNet**: Asymmetric encoder-decoder with residual blocks for high efficiency.
4. **Swin UNETR 2.5D v2**: Hierarchical Vision Transformers for global context capture.
5. **Siamese Networks**: A dedicated stage for **Longitudinal Analysis**, utilizing twin networks to learn similarity metrics and track tumor progression/regression over multiple timepoints for the same patient.

---

## 🔑 Important: How the Split Works (No Leakage)

BraTS includes multiple visits per patient (e.g., `...-100`, `...-101`, etc.).

- **Indexing is visit-level** (one record per `subject_id`)
- **Splitting is patient-level** (one patient belongs to exactly one split)
- All visits of a patient stay together (prevents inflated metrics)

---

## 🗂️ Repository Structure

```text
.
├─ dataset/
│   └─ BRATS/ ...
├─ scripts/
│   ├─ process_brats_metadata.py
│   ├─ analyze_brats_split.py
│   ├─ create_train_val_split.py
│   ├─ LUMIERE/
│   │   └─ cross_val_lumiere.py (TTA, LCC, HD95 metrics)
│   └─ visualization/ ...
├─ .gitignore
├─ README.md
└─ requirements.txt