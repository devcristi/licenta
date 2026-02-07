# Robust, Reproducible Longitudinal 3D Glioma Segmentation & RANO-Aligned Volumetry (BraTS 2024 + LUMIERE)

**Bachelor’s Thesis (in progress)** — Native 3D segmentation on multi-modal MRI + longitudinal volumetric response assessment, engineered around:
- **Reproducible experiments** (fixed seeds, pinned deps, YAML configs, TensorBoard tracking)
- **Utility + convergence evaluation** (Dice, Sens/Spec, best-epoch selection, learning dynamics)
- **Robustness / boundary behavior** using **HD95** (WT/TC/ET) and ablation-driven improvements
- **Leakage prevention** via strict **patient-level** splitting for longitudinal data
- **Model auditing** via **XAI** (Grad-CAM / attention maps) for interpretability & failure-case debugging
- **System performance** improvements (persistent SSD caching + bfloat16) enabling high-fidelity 3D runs

---

## What this repo does

### Tasks
1) **Native 3D tumor segmentation** on multi-modal MRI (WT/TC/ET).  
2) **Longitudinal volumetry (mL)** computed from predicted 3D masks per visit.  
3) **RANO-aligned response support** by mapping volumetric changes to expert-rated clinical labels on LUMIERE.

### What is measured (experiment discipline)
- **Utility:** Dice (WT/TC/ET + mean), Sensitivity, Specificity  
- **Robustness:** **HD95** (WT/TC/ET) to capture boundary outliers  
- **Convergence:** best epoch tracking + curve inspection (TensorBoard)  
- **System performance:** pipeline throughput + VRAM efficiency improvements (caching / mixed precision)

---

## Datasets

### BraTS 2024 (BraTS-GLI)
Used for controlled benchmarking of architectures and training strategies.

### LUMIERE (clinical, longitudinal)
Clinical “messy” MRI where robustness matters:
- heterogeneous acquisition protocols/scanners
- artifacts (motion/ghosting), intensity shifts, bias field effects
- distribution shift vs curated benchmarks
  
**Design principle:** improvements must hold under clinical variability and be validated through **ablations + robustness metrics (HD95)**, not only mean Dice.
>**Note:** BraTS 2024 was used for pre-training, and LUMIERE for final validation. I've used transfer learning technique.

---

## Models benchmarked (CNN vs Transformer)

- **3D U-Net (MONAI)** — baseline strong 3D segmentation backbone  
- **SegResNet (MONAI)** — efficient & strong boundaries, competitive mean Dice  
- **Swin UNETR (Transformer)** — stronger global context, attention-based representations

All runs are controlled via YAML configs and tracked in TensorBoard for comparable A/B evaluation.

---

## Results (BraTS 2024) — best epochs observed

> Metrics are reported as **WT/TC/ET**, **Mean Dice**.  
> HD95 is the **95th percentile Hausdorff distance** (lower is better).

### 📊 Metrics Tables (BraTS 2024)

### ✅ 3D U-Net — *Baseline*
> **Note:** This was the **first baseline model** and at that time **HD95 was not yet included** in the evaluation pipeline.

**Best Mean Dice (Epoch 71)**
| Metric | WT | TC | ET | Mean / Notes |
|---|---:|---:|---:|---:|
| **Dice** | 0.8105 | 0.7018 | 0.6783 | **0.7302 (Mean Dice)** |
| **Sensitivity** | 0.8429 | 0.7305 | 0.7142 | **0.7625 (Mean Sens)** |
| **Specificity** | 0.9983 | 0.9992 | 0.9992 | **0.9989 (Mean Spec)** |
| **HD95** | — | — | — | *Not reported for this baseline* |

To improve the overall performances, I've added specific augmentations: RandomSpacing / RandomZoom, Noise, Blur, Gamma/Contrast, RandomBiasField, Modality Dropout.

After applied augmentations, the same architecture obtained:

**Best Mean Dice (Epoch 158)**

| Metric | WT | TC | ET | Mean / Notes |
|---|---:|---:|---:|---:|
| **Dice** | 0.8570 | 0.8028 | 0.7830 | **0.8143 (Mean Dice)** |
| **Sensitivity** | 0.8570 | 0.8312 | 0.8188 | **0.8357 (Mean Sens)** |
| **Specificity** | 0.9988 | 0.9992 | 0.9992 | **0.9991 (Mean Spec)** |
| **HD95** | — | — | — | **Not logged at that stage of the evaluation pipeline** |

> **Note:** Both models were tested locally on an RTX 3060 laptop GPU (6GB VRAM), 16GB RAM.

**Lumiere Validation**

| Fold | WT (Whole Tumor) | TC (Tumor Core) | ET (Enhancing) | Mean Dice |
|---|---:|---:|---:|---:|
| Fold 1 | 0.7526 | 0.7014 | 0.6212 | 0.6917 |
| Fold 2 | 0.7601 | 0.7110 | 0.6763 | 0.7158 |
| Fold 3 | 0.7895 | 0.7526 | 0.7121 | 0.7514 |
| **Mean (CV)** | **0.7675 ± 0.01** | **0.7217 ± 0.02** | **0.6699 ± 0.03** | **0.7197** |


---

### ✅ SegResNet — *Best epoch observed Epoch 220*

| Metric | WT | TC | ET | Mean / Notes |
|---|---:|---:|---:|---:|
| **Dice** | 0.8671 | 0.8262 | 0.8071 | **0.8335 (Mean Dice)** |
| **HD95** | 7.9393 | 10.3352 | 14.1483 | *(lower is better)* |
| **Sensitivity** | 0.8720 | 0.8055 | 0.7954 | **0.8243 (Mean Sens)** |
| **Specificity** | 0.9988 | 0.9994 | 0.9994 | **0.9992 (Mean Spec)** |

---

### ✅ Swin UNETR — *Best epoch Epoch 250*

| Metric | WT | TC | ET | Mean / Notes |
|---|---:|---:|---:|---:|
| **Dice** | 0.8627 | 0.8150 | 0.7884 | **0.8221 (Mean Dice)** |
| **HD95** | 6.1865 | 9.5515 | 14.9278 | *(lower is better)* |
| **Sensitivity** | 0.8563 | 0.7834 | 0.7628 | **0.8008 (Mean Sens)** |
| **Specificity** | 0.9989 | 0.9994 | 0.9994 | **0.9992 (Mean Spec)** |

---

## 🔎 Model Comparison (Best Epochs)

> Quick comparison across architectures using the best observed epoch per model.  
> **Lower HD95 is better** (boundary robustness).  
> **3D U-Net baseline** does not include HD95 at that stage.

| Model | Best Epoch | Mean Dice | Mean Sens | Mean Spec | WT Dice | TC Dice | ET Dice | WT HD95 | TC HD95 | ET HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **3D U-Net (Baseline)** | 158 | **0.8143** | **0.8357** | **0.9991** | 0.8570 | 0.8028 | 0.7830 | — | — | — |
| **SegResNet** | 220 | **0.8335** | **0.8243** | **0.9992** | 0.8671 | 0.8262 | 0.8071 | 7.9393 | 10.3352 | 14.1483 |
| **Swin UNETR** | 250 | **0.8221** | **0.8008** | **0.9992** | 0.8627 | 0.8150 | 0.7884 | 6.1865 | 9.5515 | 14.9278 |

>**Conclusion:** SegResNet delivers the best overall segmentation utility in this benchmark (highest Mean Dice 0.8335 and strongest ET Dice 0.8071), making it the most reliable “default” backbone. Swin UNETR is competitive and shows better boundary robustness on WT/TC (lower HD95 6.19/9.55 vs 7.94/10.34), but at the cost of lower Mean Dice (0.8221) and lower sensitivity (0.8008 mean) on this setup. The 3D U-Net baseline is a strong first model (Mean Dice 0.8143) but lacks HD95 reporting at that stage, so boundary robustness comparisons are limited for it. The loss function was DiceCELoss. One of my future research plans is to test Focal Tversky and Dice + Focal loss.

>**Note:** Swin UNETR offers stronger global context and attention-based representations. While requiring higher data volume to reach peak volumetric accuracy, Transformers provide superior anatomic consistency and boundary smoothness compared to CNNs, even within a limited 250-epoch budget.
>
>**Experimental protocol:** Models are first benchmarked on BraTS 2024 for controlled architecture selection. The selected backbone is then validated on LUMIERE using the same patient-level split policy and longitudinal volumetry pipeline. At this stage, only the 3D U-Net baseline has completed the full LUMIERE validation, while SegResNet and Swin UNETR are currently benchmarked only on BraTS.
>
## Reproducibility

This repo is engineered for comparable experiments:
- **Fixed seeds** (deterministic where supported)
- **Pinned dependencies** (version control for core libs)
- **YAML configs** controlling model/augs/loss/optimizer/inference
- **TensorBoard tracking** for metrics, learning curves, and stability checks

Outcome: controlled A/B comparisons and defensible ablation conclusions.

---

## Leakage prevention (integrity requirement)

Splitting is strictly **patient-level**:
- all longitudinal timepoints of a patient remain in the same fold
- prevents leakage across visits
- evaluation reflects truly unseen anatomies and visit histories

---

## System performance engineering

To enable higher-fidelity 3D experimentation (larger patches, heavier models) and faster iteration:
- **Persistent SSD caching** reduces IO/decode overhead
- **bfloat16 mixed precision** improves throughput and memory efficiency

These optimizations directly increase experiment velocity and feasibility for 3D transformer models.

---

## Interpretability (XAI) for auditing

Auditing via XAI is used to verify the integrity of the learning process rather than for mere visualization. By inspecting Grad-CAM and Attention Maps, the pipeline ensures that:
-Feature Verification: The model is attending to actual tumor pathology and not to imaging artifacts or non-relevant structures (e.g., ensuring SegResNet is not erroneously "hooked" on the skull instead of the lesion).
-Failure-Case Debugging: Identifying the root cause of FP/FN results at the ET boundaries.
-Trustworthiness: Supporting explainable longitudinal response assessment outputs for medical validation.

---

## Author - **Cristian-Daniel BOABEȘ**
