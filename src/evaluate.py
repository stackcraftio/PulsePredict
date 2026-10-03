"""Evaluation helpers: metrics, confusion matrix, ROC curves, permutation importance."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def confusion_dict(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_confusion_matrix(cm: dict, model_name: str, path) -> None:
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(arr, cmap="Blues")
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            colour = "white" if arr[i, j] > arr.max() / 2 else "black"
            ax.text(j, i, f"{arr[i, j]:,}\n{labels[i][j]}", ha="center", va="center",
                    color=colour, fontsize=10)
    ax.set_xticks([0, 1], ["Predicted: No CVD", "Predicted: CVD"])
    ax.set_yticks([0, 1], ["Actual: No CVD", "Actual: CVD"])
    ax.set_title(f"Confusion matrix – {model_name} (test set)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_roc_curves(curves: dict, selected: str, path) -> None:
    """curves: {model_name: (fpr, tpr, auc)}; the selected model is drawn emphasised."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for name, (fpr, tpr, auc) in curves.items():
        sel = name == selected
        ax.plot(fpr, tpr, lw=2.4 if sel else 1.1, alpha=1 if sel else 0.6,
                label=f"{name} (AUC {auc:.3f})" + (" – selected" if sel else ""))
    ax.plot([0, 1], [0, 1], ls="--", color="#9CA3AF", lw=0.9, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (test set)", fontsize=10, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def importance_table(pipeline, X_test, y_test, random_state=42, max_rows=5000) -> pd.DataFrame:
    """Model-agnostic permutation importance (drop in ROC-AUC when a raw input is shuffled)."""
    if len(X_test) > max_rows:
        idx = X_test.sample(max_rows, random_state=random_state).index
        X_test, y_test = X_test.loc[idx], y_test.loc[idx]
    res = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=random_state, n_jobs=1)
    return (pd.DataFrame({"feature": X_test.columns, "importance": res.importances_mean,
                          "std": res.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def plot_importance(table: pd.DataFrame, model_name: str, path) -> None:
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.barh(t["feature"], t["importance"], xerr=t["std"], color="#4C78A8")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title(f"Permutation importance – {model_name}", fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
