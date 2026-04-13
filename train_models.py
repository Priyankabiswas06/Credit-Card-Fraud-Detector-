"""
train_models.py
---------------
Trains and compares multiple classifiers:
    - Logistic Regression (baseline)
    - Random Forest
    - Gradient Boosting (XGBoost-style via sklearn)
    - Isolation Forest (unsupervised anomaly detection)

Saves the best model and scaler to models/.
Prints a comparison table and saves classification reports to outputs/.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics        import (classification_report, confusion_matrix,
                                    roc_auc_score, average_precision_score,
                                    precision_recall_curve, roc_curve)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_preprocess

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Model Definitions ─────────────────────────────────────────────────────────

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=5,
        subsample=0.8, random_state=42
    ),
}


# ── Evaluation Helpers ────────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc  = roc_auc_score(y_test, y_proba)
    pr_auc   = average_precision_score(y_test, y_proba)
    report   = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(report)
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # Save report
    safe_name = name.replace(" ", "_").lower()
    with open(f"outputs/{safe_name}_report.txt", "w") as f:
        f.write(f"{name}\n{'='*55}\n")
        f.write(report)
        f.write(f"\nROC-AUC : {roc_auc:.4f}\nPR-AUC  : {pr_auc:.4f}\n")

    return {
        "name": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cm": cm,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"], ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    plt.savefig(f"outputs/{safe}_confusion_matrix.png", dpi=150)
    plt.close()


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['name']}  (AUC={r['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],"k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150)
    plt.close()
    print("  → outputs/roc_curves.png")


def plot_pr_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    for r in results:
        prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        plt.plot(rec, prec, label=f"{r['name']}  (AP={r['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/pr_curves.png", dpi=150)
    plt.close()
    print("  → outputs/pr_curves.png")


def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title("Feature Importances — Best Model")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("  → outputs/feature_importance.png")


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    print("Loading and preprocessing data …")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    results = []
    trained = {}

    for name, model in MODELS.items():
        print(f"\nTraining: {name} …")
        model.fit(X_train, y_train)
        trained[name] = model
        res = evaluate(name, model, X_test, y_test)
        plot_confusion_matrix(res["cm"], name)
        results.append(res)

    # ── Comparison Table ──────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  MODEL COMPARISON")
    print("="*55)
    comp = pd.DataFrame([{"Model": r["name"], "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"]}
                         for r in results]).sort_values("PR-AUC", ascending=False)
    print(comp.to_string(index=False))
    comp.to_csv("outputs/model_comparison.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)

    # ── Best Model ────────────────────────────────────────────────────────────
    best_name   = comp.iloc[0]["Model"]
    best_model  = trained[best_name]
    print(f"\nBest model by PR-AUC: {best_name}")

    from preprocessing import FEATURE_COLS
    plot_feature_importance(best_model, FEATURE_COLS)

    # Save
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    print(f"\nSaved → models/best_model.pkl")
    print(f"Saved → models/scaler.pkl")
    print("\nAll done! Check the outputs/ folder for plots and reports.")


if __name__ == "__main__":
    main()
