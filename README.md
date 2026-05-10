# CECS 551 – Deep Learning-Based Automated Detection of Cardiac Arrhythmia from ECG Signals

**Course:** CECS 551 – Deep Learning | California State University Long Beach | Spring 2026  
**Team:** Jashwanth Adapureddi · Leela Prasad Reddy Surasani · Mohan Sai Teja Chebrolu  

---

## Project Overview

This project implements and compares three models for automated cardiac arrhythmia detection from ECG signals under identical experimental conditions:

| # | Model | Type | Description |
|---|---|---|---|
| 1 | **Random Forest** | Traditional ML baseline | 10 handcrafted statistical features per heartbeat |
| 2 | **1D CNN** | Deep learning | Automatic morphological feature extraction from raw waveform |
| 3 | **Hybrid CNN-LSTM** | Deep learning | CNN + LSTM for intra-beat temporal pattern modeling |

**Dataset:** [ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (Kaggle: shayanfazeli/heartbeat) — derived from the MIT-BIH Arrhythmia Database (PhysioNet), 109,446 samples across 5 AAMI classes.

---

## Results Summary

| Model | Accuracy | Macro F1 | Macro AUROC |
|---|---|---|---|
| Random Forest | 0.8627 | 0.6368 | 0.9372 |
| **1D CNN** | **0.9555** | **0.8223** | **0.9876** |
| CNN-LSTM | 0.9420 | 0.7866 | 0.9887 |

**Best model:** 1D CNN — Macro F1 = 0.8223, Macro AUROC = 0.9876

### Per-Class F1-Score

| Class | RF F1 | CNN F1 | LSTM F1 | Test Support |
|---|---|---|---|---|
| Normal (N) | 0.92 | 0.98 | 0.97 | 18,118 |
| Supraventricular Ectopic (S) | 0.43 | 0.67 | 0.57 | 556 |
| Ventricular Ectopic (V) | 0.63 | 0.88 | 0.90 | 1,448 |
| Fusion Beat (F) | 0.45 | 0.61 | 0.52 | 162 |
| Unknown/Paced (Q) | 0.75 | 0.97 | 0.97 | 1,608 |
| **Macro Average** | **0.64** | **0.82** | **0.79** | 21,892 |

### ROC AUC (CNN-LSTM, One-vs-Rest)

| Class | AUC |
|---|---|
| Normal (N) | 0.992 |
| Supraventricular Ectopic (S) | 0.965 |
| Ventricular Ectopic (V) | 0.995 |
| Fusion Beat (F) | 0.992 |
| Unknown/Paced (Q) | 0.999 |

---

## Repository Structure

```
CECS551-ECG-Arrhythmia/
│
├── README.md                             ← This file
├── requirements.txt                      ← Python dependencies (pinned versions)
├── CECS551-ECG-Arrhythmia.ipynb          ← Complete notebook: Phase 3 EDA + Phase 4 training
└── CECS551_ECG_Arrhythmia.pdf            ← IEEE-style final report
```

> **Note:** The notebook contains the full pipeline in a single file — Phase 3 EDA (sections 1–9) followed by Phase 4 model training and evaluation (sections 10–15). All plots are generated and saved automatically when cells are run.

---

## Dataset Setup

The dataset is **not included** in this repository due to file size (~500 MB). To reproduce results:

1. Go to Kaggle: [shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
2. Click **Download** — extract the zip to get:
   - `mitbih_train.csv` — 87,554 samples
   - `mitbih_test.csv` — 21,892 samples
3. Upload both files to **Google Drive** in a folder named `CECS551`:
   ```
   MyDrive/
   └── CECS551/
       ├── mitbih_train.csv
       └── mitbih_test.csv
   ```

---

## How to Reproduce Results

### Recommended — Google Colab (GPU)

1. Upload `CECS551-ECG-Arrhythmia.ipynb` to [Google Colab](https://colab.research.google.com)
2. Make sure both CSV files are in `MyDrive/CECS551/` (see Dataset Setup above)
3. Click **Runtime → Run all**
4. When prompted, sign in to Google to mount Drive
5. All results, plots, and metrics generate automatically

> Training time: ~5 min (Random Forest) + ~25 min (CNN) + ~35 min (CNN-LSTM) on Colab GPU

### Alternative — Local Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/Jashu-18/CECS551-ECG-Arrhythmia.git
cd CECS551-ECG-Arrhythmia

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place CSV files in the project root directory

# 4. Update Drive paths in notebook cell to local paths:
#    TRAIN_PATH = 'mitbih_train.csv'
#    TEST_PATH  = 'mitbih_test.csv'

# 5. Launch Jupyter
jupyter notebook CECS551-ECG-Arrhythmia.ipynb
```

---

## Dependencies

All dependencies are listed in `requirements.txt`:

```
tensorflow>=2.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Reproducibility Notes

| Setting | Value |
|---|---|
| Random seed (all) | `SEED = 42` |
| Train/test split | Pre-defined Kaggle split (87,554 / 21,892) |
| Validation split | 15% stratified holdout from resampled training set |
| SMOTE | Applied **only** to training set — never to validation or test |
| Class imbalance | Two-stage: undersample Class 0 to 10,000 + SMOTE Classes 1 & 3 to 6,500 |
| Early stopping | `monitor=val_loss`, `patience=10`, `restore_best_weights=True` |
| Primary metric | Macro-averaged F1-score (not accuracy) |

---

## Preprocessing Pipeline

1. **Z-score normalization** per sample (mean=0, std=1) applied to all splits
2. **Undersampling** — Class 0 (Normal) reduced from 72,471 → 10,000
3. **SMOTE** (k=5) — Class 1: 2,223 → 6,500 | Class 3: 641 → 6,500
4. **Class-weighted loss** during deep learning training

---

## Model Architectures

### 1D CNN (157,189 parameters)
```
Input (187, 1)
→ Conv1D(64, kernel=5, ReLU, same) → MaxPool1D(2)
→ Conv1D(128, kernel=3, ReLU, same) → MaxPool1D(2)
→ Conv1D(256, kernel=3, ReLU, same)
→ GlobalAveragePooling1D
→ Dense(128, ReLU) → Dropout(0.4)
→ Dense(5, Softmax)
```

### Hybrid CNN-LSTM
```
Input (187, 1)
→ Conv1D(64, kernel=5, ReLU, same) → MaxPool1D(2)
→ Conv1D(128, kernel=3, ReLU, same) → MaxPool1D(2)
→ LSTM(64)
→ Dense(64, ReLU) → Dropout(0.3)
→ Dense(5, Softmax)
```

---

## Class Label Mapping

| Class | AAMI Code | Label | Train Count | % |
|---|---|---|---|---|
| 0 | N | Normal Beat | 72,471 | 82.8% |
| 1 | S | Supraventricular Ectopic Beat | 2,223 | 2.5% |
| 2 | V | Ventricular Ectopic Beat | 5,788 | 6.6% |
| 3 | F | Fusion Beat | 641 | 0.7% |
| 4 | Q | Unknown / Paced Beat | 6,431 | 7.3% |

---

## Citation

**Dataset:**
> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).

**Key references:**
> Kiranyaz S, Ince T, Gabbouj M. Real-time patient-specific ECG classification by 1-D convolutional neural networks. IEEE Trans Biomed Eng. 2016;63(3):664-675.

> Rajpurkar P, et al. Cardiologist-level arrhythmia detection with convolutional neural networks. arXiv:1707.01836. 2017.
