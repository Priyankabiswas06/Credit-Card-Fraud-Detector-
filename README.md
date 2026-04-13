# 💳 Credit Card Fraud Detection

A complete, end-to-end machine learning project for detecting fraudulent credit card transactions using Python and scikit-learn.

---

## 📁 Project Structure

```
credit_card_fraud_detection/
│
├── data/                        # Generated dataset (created at runtime)
│   └── transactions.csv
│
├── models/                      # Saved model artifacts
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── best_model_name.txt
│
├── outputs/                     # Plots, reports, comparison tables
│   ├── eda/                     # EDA charts
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── feature_importance.png
│   ├── model_comparison.csv
│   └── *_report.txt
│
├── generate_data.py             # Step 1 – Synthetic dataset generator
├── preprocessing.py             # Feature engineering + train/test split
├── eda.py                       # Step 2 – Exploratory data analysis
├── train_models.py              # Step 3 – Train & evaluate models
├── predict.py                   # Step 4 – Load model and predict
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / unzip the project
cd credit_card_fraud_detection

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start (Run in Order)

### Step 1 — Generate the Dataset
```bash
python generate_data.py
```
Creates `data/transactions.csv` with 50,000 transactions (~2% fraud).

### Step 2 — Exploratory Data Analysis
```bash
python eda.py
```
Saves 5 EDA charts to `outputs/eda/`.

### Step 3 — Train Models
```bash
python train_models.py
```
Trains **Logistic Regression**, **Random Forest**, and **Gradient Boosting**, prints a comparison table, and saves the best model + scaler to `models/`.

### Step 4 — Predict
```bash

```

---
### Models Compared
- **Logistic Regression** — fast linear baseline
- **Random Forest** — robust ensemble, handles non-linearity well
- **Gradient Boosting** — typically best precision-recall trade-off

---

## 📈 Key Metrics

For imbalanced fraud detection, **PR-AUC (Average Precision)** is more meaningful than accuracy.

| Metric | Description |
|--------|-------------|
| ROC-AUC | Overall discriminative ability |
| PR-AUC | Precision-Recall trade-off at all thresholds |
| Recall (Fraud) | % of actual fraud caught |
| Precision (Fraud) | % of flagged transactions that are actually fraud |

---

## 🔧 Customisation

- **Dataset size**: Edit `NUM_TRANSACTIONS` in `generate_data.py`
- **Fraud ratio**: Edit `FRAUD_RATIO` in `generate_data.py`
- **Add a model**: Add an entry to the `MODELS` dict in `train_models.py`
- **Real data**: Replace `generate_data.py` with your own CSV loader (keep the same column names or update `FEATURE_COLS` in `preprocessing.py`)

---

## 📦 Dependencies
```
scikit-learn >= 1.3
imbalanced-learn >= 0.11
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
```
