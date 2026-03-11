# AppleGrade_Classificator

# 🍎 Apple Quality Detection — Deep Learning Classifier

> Automated apple quality classification using Transfer Learning, Explainable AI (XAI), and hybrid CNN/ViT architectures.  
> University project · Hochschule Ludwigshafen am Rhein · MSc Business Informatics · Module MW341

---

## 📌 Overview

This project develops a **multi-class image classifier** that automatically sorts apples into three quality grades based on visual features — replacing manual, error-prone sorting in agricultural supply chains.

The system compares six model configurations across three state-of-the-art backbone architectures, each paired with both a neural and a classical classifier head. Model decisions are made interpretable using **LIME** and **SHAP** (XAI methods).

---

## 🏆 Results at a Glance

| Architecture | Head | Accuracy | F1 (weighted) |
|---|---|---|---|
| **ConvNeXt (small)** | **Deep Learning** | **0.875** | **0.875** |
| Swin Transformer (tiny) | Deep Learning | 0.830 | 0.825 |
| EfficientNet (B3) | Deep Learning | 0.742 | 0.745 |
| EfficientNet (B3) | Random Forest | 0.726 | 0.733 |
| ConvNeXt (small) | Random Forest | 0.469 | 0.431 |
| Swin Transformer (tiny) | Random Forest | 0.435 | 0.378 |

**→ ConvNeXt + Deep Learning head achieved the best overall performance.**

---

## 🍏 Problem Statement

Manual apple sorting is:
- Time-consuming and expensive
- Error-prone and inconsistent
- Difficult to scale with growing harvest volumes

**Target classes:**
1. **1st Category** — Flawless apples (retail-ready)
2. **2nd Category** — Minor defects (secondary market)
3. **3rd Category** — Heavily defective (animal feed / disposal)

More precise automated classification of *Category 2* maximizes farmer profits and reduces food waste.

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Total images | 1,263 |
| Format | JPEG |
| Image size (after preprocessing) | 224 × 224 px |
| Class distribution | 462 / 451 / 350 |
| Split | 80% train / 10% val / 10% test |
| Perspectives | Top-down and side view |

Class imbalance was addressed using **inverse-frequency class weights**:
```
Class weights: {0: 0.908, 1: 0.931, 2: 1.201}
```

---

## 🏗️ Architectures

### EfficientNet (B3)
CNN-based architecture using **Compound Scaling** to jointly optimize depth, width, and input resolution. Backbone: MBConv blocks with Depthwise Separable Convolution and Squeeze-and-Excitation.

### Swin Transformer (Tiny)
Vision Transformer with **Shifted Window Self-Attention** — reduces complexity from O(N²) to O(N) while maintaining global context through cross-window connections. Combines hierarchical feature extraction with Transformer-style global reasoning.

### ConvNeXt (Small)
A pure-CNN architecture modernized with **Transformer-era design principles** — large 7×7 depthwise convolutions, GELU activations, and a hierarchical stage structure inspired by Swin. Demonstrates that well-designed CNNs can match Transformer performance.

### Hybrid Approach (CNN/ViT + Random Forest)
All three backbones were also evaluated with their neural classification head replaced by a **Random Forest** trained on extracted feature embeddings. This tests whether generic features generalize well to classical ensemble methods.

---

## ⚙️ Training Setup

All experiments followed a **two-phase training protocol**:

1. **Warmup** — Only the classification head is trained (`backbone.trainable = False`)
2. **Fine-Tuning** — Full model is trained with a lower learning rate

| Hyperparameter | Detail |
|---|---|
| Optimizer | AdamW |
| LR Schedule | CosineDecay |
| Loss | Categorical Crossentropy + Label Smoothing (0.1) |
| Callbacks | EarlyStopping, ModelCheckpoint, TensorBoard |
| Mixed Precision | float16 |
| Seed | 42 (fully deterministic) |

---

## 🔍 Explainability (XAI)

### LIME (Local Interpretable Model-Agnostic Explanations)
Segments images into superpixels and fits a local linear model to approximate the classifier's decision boundary. Highlights which image regions *support* or *contradict* a predicted class.

**Finding:** ConvNeXt attends broadly to surface texture and shape. Swin Transformer focuses more sharply on specific discriminative regions.

### SHAP (SHapley Additive exPlanations)
Distributes prediction "credit" fairly across all input features using cooperative game theory (Shapley values). Used for both **local** (single image) and **global** (aggregated) analysis.

**Finding:** Swin Transformer consistently focuses on object contours across all images. ConvNeXt shows higher sensitivity to background artefacts in the global view.

> ⚠️ High classification accuracy ≠ high robustness. ConvNeXt achieves better metrics but is more susceptible to irrelevant image artefacts than Swin Transformer.

---

## 🛠️ Tech Stack

```
Python 3.x
TensorFlow / Keras
tfswin (Swin Transformer for TF)
scikit-learn (Random Forest, metrics)
SHAP
LIME (lime)
OpenCV (cv2)
Matplotlib / Seaborn
Kaggle Notebooks (GPU/TPU training environment)
```

---

## 📁 Project Structure

```
├── Projektbericht.ipynb        # Main Jupyter Notebook (full pipeline)
└── README.md
```

**The notebook covers:**
- Data loading & pipeline
- Data augmentation
- Class weight computation
- Model training (all 6 experiments)
- Evaluation (Accuracy, F1, Confusion Matrix, ROC/PR curves)
- LIME explanations
- SHAP explanations (local + global)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow tfswin scikit-learn shap lime opencv-python matplotlib seaborn
```

### Dataset

The dataset is **not included** in this repository (proprietary, manually labelled). To reproduce the experiments, provide a directory with the following structure:

```
Apple_transform/
├── 1st_cat/
│   └── *.jpg
├── 2nd_cat/
│   └── *.jpg
└── 3rd_cat/
    └── *.jpg
```

Update `DATA_DIR` in the notebook to point to your dataset location.

### Run on Kaggle

This project was developed on **Kaggle Notebooks** with free GPU access.  
To run it:
1. Upload the notebook to Kaggle
2. Attach the dataset as a Kaggle Dataset input
3. Enable GPU accelerator
4. Run all cells sequentially

> **Note:** Kaggle requires phone number verification to use non-native libraries and internet access inside notebooks.

---

## 📊 Key Findings

- **ConvNeXt + DL Head** is the strongest performer overall (F1 = 0.875), confirming modern CNN designs remain competitive with Transformers on small datasets.
- **Hybrid RF heads** significantly underperform for Transformer architectures, suggesting their feature spaces require non-linear neural classification.
- **EfficientNet + RF** is the only hybrid configuration that performs comparably to its neural counterpart, likely due to the more compatible feature geometry of CNN embeddings.
- **XAI reveals a trade-off** between accuracy and interpretability/robustness — Swin Transformer shows more semantically coherent attention patterns despite slightly lower accuracy.

---

## ⚠️ Limitations

- Small dataset (1,263 images) limits generalizability
- Controlled recording conditions may not transfer to real production environments
- Models differ in parameter count and training budget, limiting direct comparability
- Results are indicative signals, not statistically validated performance 

---

## 📚 References

- Tan & Le (2019) — EfficientNet: Rethinking Model Scaling for CNNs. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Liu et al. (2021) — Swin Transformer: Hierarchical ViT using Shifted Windows. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- Liu et al. (2022) — A ConvNet for the 2020s. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Ribeiro et al. (2016) — "Why Should I Trust You?": Explaining the Predictions of Any Classifier. [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- Breiman (2001) — Random Forests. Machine Learning.
- Pan & Yang (2010) — A Survey on Transfer Learning. IEEE TKDE.

---

