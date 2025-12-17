# 🧠 Brain Tumor Segmentation & Classification with Deep Learning

## Project Overview

Comprehensive deep learning project for **brain tumor segmentation and classification from MRI scans** with focus on:

- **Automated segmentation** using 3D U-Net and CNN architectures
- **Transfer learning** from BRATS → LUMIERE datasets
- **Explainability (XAI)** with Grad-CAM and SHAP
- **Robustness evaluation** against adversarial attacks
- **Clinical-grade metrics** for medical imaging

## 📊 Dataset Statistics

### BRATS 2024
- **Training**: 584 patients | 1,324 visits (80%)
- **Validation**: 147 patients | 297 visits (20%)
- **Test**: 87 patients | 188 visits
- **Total**: 818 unique patients | 1,809 MRI sessions

### LUMIERE
- **Patients**: 90 with expert ratings
- **Modalities**: T1, T1c, T2w, T2-FLAIR

### MRI Sequences
- T1 native (t1_path)
- T1 contrast (t1c_path)
- T2-weighted (t2w_path)
- T2 FLAIR (t2_path)
- Segmentation labels (seg_path)

## 🏗️ Project Structure

```
dataset/
  ├── BRATS/
  │   ├── BraTS2024-BraTS-GLI-TrainingData/
  │   ├── BraTS2024-BraTS-GLI-AdditionalTrainingData/
  │   ├── BraTS2024-BraTS-GLI-ValidationData/
  │   ├── brats_metadata.json
  │   ├── brats_metadata_splits.json
  │   └── brats_splits_statistics.json
  └── LUMIERE/
      ├── Imaging/ (Patient-001 to Patient-090)
      └── LUMIERE-ExpertRating.csv

scripts/
  ├── process_brats_metadata.py
  ├── analyze_brats_split.py
  ├── create_train_val_split.py
  ├── data_loader.ipynb
  └── brats_visualization.ipynb
```

## 🚀 Quick Start

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Generate Metadata
```bash
python scripts/process_brats_metadata.py
python scripts/analyze_brats_split.py
python scripts/create_train_val_split.py
```

## 📈 Training

### Pre-training on BRATS
```bash
python models/training.py --dataset brats --epochs 100
```

### Fine-tuning on LUMIERE
```bash
python models/training.py --dataset lumiere --pretrained brats_model.pth
```

## 🔍 Explainability & Robustness

### Grad-CAM Visualization
```python
from models.xai_explainability import GradCAM
grad_cam = GradCAM(model)
heatmap = grad_cam.generate(mri_scan)
```

### Robustness Testing
```bash
python models/robustness_evaluation.py --attack fgsm
python models/robustness_evaluation.py --corruption gaussian_noise
```

## 📊 Evaluation Metrics

- Dice Coefficient
- Hausdorff Distance
- Sensitivity/Specificity
- AUC-ROC
- Uncertainty Quantification

## 📚 References

- BRATS Dataset: [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- U-Net: [Arxiv 1505.04597](https://arxiv.org/abs/1505.04597)
- Grad-CAM: [Arxiv 1610.02055](https://arxiv.org/abs/1610.02055)
- SHAP: [Arxiv 1705.07874](https://arxiv.org/abs/1705.07874)

## 📝 License

MIT License

---

**Status**: 🚀 In Development  
**Last Updated**: December 2025
