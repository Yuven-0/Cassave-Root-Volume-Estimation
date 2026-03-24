# 🌿 Cassava Root Volume Estimation

> A deep learning solution for **non-invasive cassava root volume prediction** using Ground Penetrating Radar (GPR) scan images, built for the [CGIAR Root Volume Estimation Challenge]([CGIAR Root Volume Estimation Challenge](https://zindi.africa/competitions/cgiar-root-volume-estimation-challenge)).

**By Yuven Blowria, Puneet Madan, Pehar Jhamb**

---

## 📋 Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Our Approach](#our-approach)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [References](#references)

---

## Background

Cassava is Africa's most important tuber crop, supporting the livelihoods of over **300 million people** and contributing over 50% of local food intake across many regions. Despite Africa producing more than 50% of the world's cassava, average yield per hectare (8.9 t/ha) remains far below the crop's potential (up to 80 t/ha).

A key bottleneck is the inability to efficiently measure root volume at scale. Traditional methods require physically uprooting and measuring plants, a process that is **destructive, labor-intensive, error-prone, and not scalable** across large breeding trials or field surveys.

---

## Problem Statement

This project addresses the need for a **non-invasive, automated method** to estimate cassava storage root volume directly from GPR scan images, without uprooting the plant. This enables:

- Early yield prediction before harvest
- Non-destructive selection of high-performing genotypes
- Real-time, scalable root growth tracking
- Support for precision agriculture and breeding programs

---

## Our Approach

We built a **CNN + LSTM hybrid model in PyTorch** that processes sequences of GPR scan image slices to predict root volume as a continuous scalar value.

The key insight: a GPR scan of a cassava root produces a series of 2D grayscale cross-section images (slices) at different depths. Volume estimation isn't just about any individual slice, it depends on the **spatial pattern across the full sequence** of slices. A CNN alone misses this depth progression; our model uses a CNN to extract per-slice features, then an LSTM to model the sequential structure.

### Why CNN + LSTM?

| Component | Role |
|-----------|------|
| **CNN** | Extracts spatial features from each individual 2D GPR slice |
| **LSTM** | Models the sequential depth-wise structure across 21 slices |
| **Feature Fusion** | Combines Left and Right GPR view representations |
| **FC Head** | Predicts the final root volume scalar |

### Literature Inspiration

- **Atanbori et al. (2019)**, CNNs are effective for processing plant root imagery (real + synthetic), inspiring our CNN backbone.
- **Adebayo (2023)**, Identified that no existing model used GPR for cassava root volume, directly motivating our approach.
- **Fariñas et al. (2019)**, Non-destructive ultrasonic/spectral sensing reinforced the value of indirect signal interpretation via ML.
- **Wang et al. (2025)**, Validated the use of sequential spectral patterns, inspiring our use of LSTM for depth-wise modeling.

---

## Model Architecture

```
Input: (batch, 2, 21, 128, 128)
  └── 2 sides (Left & Right GPR views)
  └── 21 grayscale slices per side
  └── 128×128 pixels per slice

For each side:
  ├── Reshape: (batch×21, 1, 128, 128)
  ├── CNN Feature Extraction:
  │     Conv2d(1→16, 3×3, stride=2) + ReLU
  │     Conv2d(16→32, 3×3, stride=2) + ReLU
  │     AdaptiveAvgPool2d(1)
  │     → (batch, 21, 32)
  └── LSTM(input=32, hidden=64)
        → final hidden state: (batch, 64)

Feature Fusion:
  Concatenate L + R outputs → (batch, 128)

Fully Connected Head:
  Linear(128→64) + ReLU
  Linear(64→1)

Output: Single predicted root volume (scalar)
```

---

## Dataset

The dataset comes from the **CGIAR Root Volume Estimation Challenge** and consists of GPR scan images of cassava plants paired with measured root volumes.

### CSV Files

| File | Rows | Description |
|------|------|-------------|
| `Train.csv` | 386 | Training samples with ground-truth root volume |
| `Test.csv` | 130 | Test samples for inference |
| `Sample_Submission.csv` | 130 | Expected submission format |

### CSV Columns

| Column | Description |
|--------|-------------|
| `ID` | Unique sample identifier |
| `FolderName` | GPR image folder for this sample (links to `data/train/` or `data/test/`) |
| `PlantNumber` | Root number within the scan (up to 7 roots per scan) |
| `Side` | GPR scan direction: `L` (left) or `R` (right) |
| `Start` | Starting slice index for this root |
| `End` | Ending slice index for this root |
| `RootVolume` | Target variable, measured root volume *(train only)* |
| `Genotype` | Cassava variety (7 unique: IITA-TMS-IBA000070, IBA154810, IBA980581, TMEB419, TMEB693, DIXON, IKN130010) |
| `Stage` | Growth stage: `Early` or `Late` |

### Root Volume Distribution (Train)

| Stat | Value |
|------|-------|
| Mean | 2.05 |
| Std | 1.53 |
| Min | 0.00 |
| Median | 1.90 |
| Max | 11.00 |

### Image Data

```
data/
├── train/        # 98 folders, each named by FolderName
│   └── <FolderName>/
│       ├── <FolderName>_L_001.png   # Left view, slice 1
│       ├── <FolderName>_L_002.png
│       ├── ...
│       ├── <FolderName>_L_021.png   # Left view, slice 21
│       ├── <FolderName>_R_001.png   # Right view, slice 1
│       └── ...
└── test/
    └── <FolderName>/
        └── ...

train_labels/     # YOLO-format bounding box labels for root regions
```

Each folder contains up to **42 grayscale PNG images** (21 slices × 2 sides). Each image is a 2D cross-sectional GPR scan at a specific depth.

---

## Data Preprocessing

The `RootVolumeImageDataset` class handles all preprocessing:

1. **Load image slices**, For each sample, slices 001–021 are loaded for both the Left (L) and Right (R) GPR views.
2. **Handle missing/corrupted files**, If any slice is missing or unreadable, it is replaced with a zero-filled black image rather than raising an error.
3. **Transform**, Each slice is converted to PIL, resized to `128×128`, and converted to a PyTorch tensor.
4. **Stack slices**, 21 slices per side are stacked to shape `(21, H, W)`. Both sides are combined into a final tensor of shape `(2, 21, H, W)`.

---

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss function | MSE |
| Batch size | 8 |
| Max epochs | 100 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | patience=15 |
| Train/val split | 80/20 (random, seed=42) |
| Image size | 128×128 |

The best model checkpoint is saved to `Root_model.pt` based on lowest validation loss.

---

## Results

| Split | RMSE |
|-------|------|
| Validation | 1.374 |
| Public Leaderboard | 1.075 |
| **Private Leaderboard** | **1.385** |

The private leaderboard RMSE closely matches the validation RMSE, indicating **good generalization** with no significant overfitting. The model predicts root volume within approximately **±1.4 units** of the true value.

Training history is saved to `training_history.png`:

![Training History](training_history.png)

---

## Repository Structure

```
Cassave-Root-Volume-Estimation/
│
├── CGIAR Root Volume Estimation Challenge.ipynb   # Main model: data loading, training, inference
├── Visualization.ipynb                            # EDA and GPR image visualization
│
├── Train.csv                                      # Training metadata + labels
├── Test.csv                                       # Test metadata
├── Sample_Submission.csv                          # Submission format
├── submission.csv                                 # Final model predictions
│
├── training_history.png                           # Train/val loss and RMSE curves
│
├── data/
│   ├── train/          # GPR scan images for training (98 plant folders)
│   └── test/           # GPR scan images for inference
│
└── train_labels/       # Bounding box label files for root regions (YOLO format)
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Model

1. Clone the repository:
```bash
git clone https://github.com/Yuven-0/Cassave-Root-Volume-Estimation.git
cd Cassave-Root-Volume-Estimation
```

2. Ensure the data directory is populated with GPR image folders under `data/train/` and `data/test/`.

3. Open and run `CGIAR Root Volume Estimation Challenge.ipynb` in Jupyter:
```bash
jupyter notebook "CGIAR Root Volume Estimation Challenge.ipynb"
```

4. Run the final cell (`main()`) to train the model and generate `submission.csv`.

### Exploratory Data Analysis

Open `Visualization.ipynb` to explore the GPR scan images, visualize root regions with bounding overlays, and understand the data structure before training.

---

## Challenges Faced

**Data complexity**, Each GPR scan can contain up to 7 individual roots. The images are split into Left and Right views, making it non-trivial to understand which regions correspond to which root.

**Sequence awareness**, A plain CNN processes each slice independently and cannot capture how root volume accumulates across depth. Adding an LSTM on top of CNN features was the key architectural improvement that significantly reduced RMSE.

**Initial direction**, Most prior implementations for similar tasks used pre-trained YOLO models for detection, which were unsuitable here since we needed end-to-end volume regression from scratch.

---

## Future Work

- **Use tabular metadata**, The `Genotype`, `Stage`, `Start`, and `End` columns in the CSV are currently unused by the model. Incorporating these as additional inputs to the FC head could improve accuracy, as root morphology varies significantly between genotypes and growth stages.
- **Group-aware train/val split**, The current split is row-level; using a group-based split on `FolderName` would prevent the same plant from appearing in both train and validation sets.
- **Data augmentation**, With ~310 training samples, augmentations like horizontal flipping, brightness jitter, or random crops on GPR slices could improve generalization.
- **Deeper CNN backbone**, Adding a third conv layer with BatchNorm could extract richer spatial features from the 128×128 slices.
- **Transfer to other crops**, The GPR + CNN-LSTM pipeline could potentially be adapted for other root crops (yam, sweet potato, sugarcane) given labeled GPR scan data.

---

## References

1. Adebayo, W. G. (2023). *Cassava Production in Africa: A Panel Analysis of the Drivers and Trends.*
2. Atanbori, J., et al. (2019). *Convolutional Neural Net-Based Cassava Storage Root Counting Using Real and Synthetic Images.* Frontiers in Plant Science, 10, 1516.
3. Liu, X., et al. (2017). *Ground Penetrating Radar (GPR) Detects Fine Roots of Agricultural Crops in the Field.*
4. Lantini, L., et al. (2020). *Application of Ground Penetrating Radar for Mapping Tree Root System Architecture and Mass Density of Street Trees.*
5. Fariñas, M. D., et al. (2019). *Instantaneous and Non-Destructive Relative Water Content Estimation from Deep Learning Applied to Resonant Ultrasonic Spectra of Plant Leaves.*
6. Wang, et al. (2025). *Estimating Maize Leaf Water Content Using Machine Learning with Diverse Multispectral Image Features.*
7. Antúnez, P., et al. (2024). *Predictive Modeling of Volume and Biomass in Pinus pseudostrobus Using Machine Learning and Allometric Approaches.*

---

*Built as part of the CGIAR Root Volume Estimation Challenge. For questions, open an issue or reach out via GitHub.*
