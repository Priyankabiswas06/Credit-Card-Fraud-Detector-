"""
eda.py
------
Exploratory Data Analysis — generates charts saved to outputs/eda/.
Run before train_models.py to understand the dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/eda", exist_ok=True)

df = pd.read_csv("data/transactions.csv")

print(f"Shape : {df.shape}")
print(f"Fraud : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(df.describe().round(2))


# ── 1. Class Distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["is_fraud"].value_counts()
ax.bar(["Legit", "Fraud"], counts.values, color=["steelblue", "tomato"])
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 100, f"{v:,}", ha="center")
plt.tight_layout()
plt.savefig("outputs/eda/class_distribution.png", dpi=150)
plt.close()
print("→ outputs/eda/class_distribution.png")


# ── 2. Transaction Amount: Fraud vs Legit ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, log in zip(axes, [False, True]):
    for label, color in [(0, "steelblue"), (1, "tomato")]:
        vals = df.loc[df["is_fraud"]==label, "amount"]
        if log: vals = np.log1p(vals)
        ax.hist(vals, bins=60, alpha=0.6, color=color,
                label=["Legit","Fraud"][label], density=True)
    ax.set_title(f"Amount Distribution {'(log scale)' if log else ''}")
    ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda/amount_distribution.png", dpi=150)
plt.close()
print("→ outputs/eda/amount_distribution.png")


# ── 3. Fraud by Hour ──────────────────────────────────────────────────────────
hourly = df.groupby("hour")["is_fraud"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(hourly["hour"], hourly["is_fraud"]*100, color="tomato", alpha=0.8)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Fraud Rate (%)")
ax.set_title("Fraud Rate by Hour of Day")
ax.set_xticks(range(24))
plt.tight_layout()
plt.savefig("outputs/eda/fraud_by_hour.png", dpi=150)
plt.close()
print("→ outputs/eda/fraud_by_hour.png")


# ── 4. Correlation Heatmap ────────────────────────────────────────────────────
num_cols = ["amount","dist_from_home","n_txn_1h","v1","v2","v3","v4","v5","is_fraud"]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.3, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/eda/correlation_heatmap.png", dpi=150)
plt.close()
print("→ outputs/eda/correlation_heatmap.png")


# ── 5. Distance from Home ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color in [(0, "steelblue"), (1, "tomato")]:
    vals = np.log1p(df.loc[df["is_fraud"]==label, "dist_from_home"])
    ax.hist(vals, bins=50, alpha=0.6, color=color,
            label=["Legit","Fraud"][label], density=True)
ax.set_xlabel("log(1 + Distance from Home)")
ax.set_title("Distance from Home: Fraud vs Legit")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda/distance_distribution.png", dpi=150)
plt.close()
print("→ outputs/eda/distance_distribution.png")

print("\nEDA complete. All charts saved to outputs/eda/")
