"""
predict.py
----------
Load the saved model and predict on new transactions.

Usage:
    python predict.py                          # demo with random samples
    python predict.py --csv path/to/file.csv   # predict on a CSV file
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from preprocessing import engineer_features, FEATURE_COLS


def load_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/best_model_name.txt") as f:
        name = f.read().strip()
    print(f"Loaded model: {name}")
    return model, scaler


def predict_df(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    df_feat = engineer_features(df)
    X = scaler.transform(df_feat[FEATURE_COLS].values)
    proba  = model.predict_proba(X)[:, 1]
    labels = model.predict(X)
    df = df.copy()
    df["fraud_probability"] = np.round(proba, 4)
    df["prediction"]        = labels
    df["verdict"]           = df["prediction"].map({0: "✅ Legit", 1: "🚨 FRAUD"})
    return df


def demo():
    """Run predictions on a handful of hand-crafted examples."""
    samples = pd.DataFrame({
        "amount":        [25.0,  4500.0, 85.0,  12000.0, 60.0],
        "hour":          [14,     2,      11,     3,       9],
        "merchant_cat":  [3,      7,      3,      7,       5],
        "dist_from_home":[5.0,   450.0,  12.0,  900.0,   8.0],
        "n_txn_1h":      [1,      7,      2,      9,       1],
        "v1": [ 0.2, -3.1,  0.1, -3.5,  0.3],
        "v2": [-0.1,  4.2, -0.2,  4.8, -0.1],
        "v3": [ 0.0, -1.8,  0.1, -2.1,  0.0],
        "v4": [ 0.1,  2.9, -0.1,  3.2,  0.2],
        "v5": [-0.2, -4.1,  0.0, -4.6, -0.1],
    })

    model, scaler = load_artifacts()
    result = predict_df(samples, model, scaler)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n── Predictions ─────────────────────────────────────────────")
    print(result[["amount","hour","dist_from_home","n_txn_1h",
                   "fraud_probability","verdict"]].to_string(index=False))


def predict_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    model, scaler = load_artifacts()
    result = predict_df(df, model, scaler)
    out_path = csv_path.replace(".csv", "_predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"\nPredictions saved → {out_path}")
    print(f"Flagged as fraud: {result['prediction'].sum():,} / {len(result):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Predictor")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file with transactions to predict")
    args = parser.parse_args()

    if args.csv:
        predict_csv(args.csv)
    else:
        demo()
