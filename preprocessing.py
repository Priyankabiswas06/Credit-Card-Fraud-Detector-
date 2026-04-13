"""
preprocessing.py
----------------
Feature engineering, preprocessing pipeline, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataframe."""
    df = df.copy()

    # Log-transform skewed columns
    df["log_amount"]        = np.log1p(df["amount"])
    df["log_dist_from_home"]= np.log1p(df["dist_from_home"])

    # Is the transaction during off-hours (midnight–6 AM)?
    df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 else 0)

    # High-frequency flag
    df["high_freq"] = (df["n_txn_1h"] >= 4).astype(int)

    # Interaction: large amount + night
    df["night_large_amt"] = df["is_night"] * df["log_amount"]

    return df


FEATURE_COLS = [
    "log_amount", "log_dist_from_home",
    "hour", "is_night", "high_freq", "night_large_amt",
    "merchant_cat", "n_txn_1h",
    "v1", "v2", "v3", "v4", "v5",
]
TARGET_COL = "is_fraud"


# ── Preprocessing Pipeline ────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, test_size: float = 0.2, use_smote: bool = True):
    """
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df = engineer_features(df)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Handle class imbalance with SMOTE on training set only
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {y_train.sum():,} fraud / {(y_train==0).sum():,} legit in train set")

    return X_train, X_test, y_train, y_test, scaler


def load_and_preprocess(csv_path: str = "data/transactions.csv", use_smote: bool = True):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows  |  Fraud rate: {df[TARGET_COL].mean()*100:.2f}%")
    return preprocess(df, use_smote=use_smote)
