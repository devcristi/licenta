# 🧠 licenta — Post-Treatment Adult Glioma (BraTS 2024) Segmentation + Longitudinal Analysis

Bachelor thesis codebase for **3D deep learning on post-treatment adult glioma MRI** using **BraTS 2024 (BraTS-GLI)**, with optional transfer/longitudinal experiments on **LUMIERE**.

The core goal is to build a **clean, reproducible PyTorch pipeline** for **multi-modal 3D segmentation**, with **patient-level splitting** to avoid leakage across multiple visits/timepoints.

---

## ✨ Project Highlights

- **3D multi-modal tumor/subregion segmentation** (baseline: 3D U-Net style models)
- **Patient-level split** (train/val), while training on **visits** (timepoints)
- **Inference-only test set** (official BraTS Validation: no `seg`)
- Dataset tooling:
  - metadata parsing
  - split analysis
  - train/val split generation
  - basic visualization notebooks

> Note: The “official validation” in BraTS is **unlabeled** here, so supervised metrics are computed on an **internal validation split** sampled from trainable patients.

---

## 📊 Dataset Statistics (Current Split)

### BraTS 2024 — BraTS-GLI (Post-Treatment)
- 🔵 **TRAIN (internal)**: 584 patients | 1,324 visits  
- 🟡 **VAL (internal)**: 147 patients | 297 visits  
- 🔴 **TEST (official BraTS validation, unlabeled)**: 87 patients | 188 visits  
- **Total**: 818 unique patients | 1,809 MRI sessions (visits)

### MRI Sequences (per visit)
- `t1n` — T1 native (non-contrast)
- `t1c` — T1 contrast-enhanced
- `t2w` — T2 weighted
- `t2f` — T2 FLAIR
- `seg` — segmentation label (**trainable splits only**)

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
│  └─ BRATS/
│     ├─ brats_metadata.json
│     ├─ brats_metadata_splits.json
│     └─ brats_splits_statistics.json
├─ scripts/
│  ├─ process_brats_metadata.py
│  ├─ analyze_brats_split.py
│  ├─ create_train_val_split.py
│  ├─ data_loader.ipynb
│  └─ brats_visualization.ipynb
├─ .gitignore
├─ README.md
└─ requirements.txt
