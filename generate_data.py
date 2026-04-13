"""
generate_data.py
----------------
Generates a synthetic credit card transaction dataset for fraud detection.
Saves it to data/transactions.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

NUM_TRANSACTIONS = 50000
FRAUD_RATIO = 0.02  # 2% fraud

n_fraud = int(NUM_TRANSACTIONS * FRAUD_RATIO)
n_legit = NUM_TRANSACTIONS - n_fraud


def generate_legit(n):
    return pd.DataFrame({
        "amount":        np.random.exponential(scale=80,  size=n).clip(0.5, 5000),
        "hour":          np.random.choice(range(24), size=n, p=_hour_probs()),
        "merchant_cat":  np.random.randint(1, 16, size=n),
        "dist_from_home": np.random.exponential(scale=20, size=n).clip(0, 500),
        "n_txn_1h":      np.random.poisson(lam=1.2, size=n).clip(0, 10),
        "v1": np.random.normal(0,  1, n),
        "v2": np.random.normal(0,  1, n),
        "v3": np.random.normal(0,  1, n),
        "v4": np.random.normal(0,  1, n),
        "v5": np.random.normal(0,  1, n),
        "is_fraud": 0,
    })


def generate_fraud(n):
    return pd.DataFrame({
        "amount":        np.random.exponential(scale=300, size=n).clip(10, 10000),
        "hour":          np.random.choice(range(24), size=n, p=_fraud_hour_probs()),
        "merchant_cat":  np.random.randint(1, 16, size=n),
        "dist_from_home": np.random.exponential(scale=200, size=n).clip(0, 5000),
        "n_txn_1h":      np.random.poisson(lam=4.5, size=n).clip(0, 20),
        "v1": np.random.normal(-2, 2, n),
        "v2": np.random.normal( 3, 2, n),
        "v3": np.random.normal(-1, 2, n),
        "v4": np.random.normal( 2, 2, n),
        "v5": np.random.normal(-3, 2, n),
        "is_fraud": 1,
    })


def _hour_probs():
    # Business hours heavier
    w = np.array([0.5,0.3,0.2,0.2,0.3,0.5,1,2,3,4,5,5,5,5,5,5,5,5,4,3,2,1.5,1,0.7])
    return w / w.sum()


def _fraud_hour_probs():
    # Night-heavy
    w = np.array([3,3,3,3,2,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,2,2,3,3,3,3])
    return w / w.sum()


if __name__ == "__main__":
    legit = generate_legit(n_legit)
    fraud = generate_fraud(n_fraud)
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Dataset saved → data/transactions.csv")
    print(f"  Total rows : {len(df):,}")
    print(f"  Fraud rows : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f}%)")
