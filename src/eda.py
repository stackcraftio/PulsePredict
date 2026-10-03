"""Eight deliberately chosen EDA figures. Used by both the training script (saves PNGs)
and the Streamlit Data Explorer (renders them live)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import TARGET  # noqa: E402

C0, C1, CA = "#4C78A8", "#E4572E", "#6B7280"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def _rate_bar(ax, labels, rates, counts, color=C1):
    bars = ax.bar(labels, rates * 100, color=color)
    for b, r, n in zip(bars, rates, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{r*100:.0f}%\n(n={n:,})",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(100 * 0.5, color=CA, lw=0.8, ls="--")


def _grouped_rate(df, key):
    g = df.groupby(key, observed=True)[TARGET].agg(["mean", "count"])
    return g.index.astype(str), g["mean"].to_numpy(), g["count"].to_numpy()


def fig_class_balance(df):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["No CVD (0)", "CVD (1)"], counts.to_numpy(), color=[C0, C1])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    _style(ax, "Target / class balance", ylabel="Records")
    fig.tight_layout()
    return fig


def fig_age_distribution(df):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.linspace(30, 65, 29)
    ax.hist(df.loc[df[TARGET] == 0, "age_years"], bins=bins, alpha=0.65, color=C0, label="No CVD")
    ax.hist(df.loc[df[TARGET] == 1, "age_years"], bins=bins, alpha=0.65, color=C1, label="CVD")
    ax.legend(frameon=False)
    _style(ax, "Age distribution", "Age (years)", "Records")
    fig.tight_layout()
    return fig


def fig_bp_distribution(df):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(df["systolic_bp"], bins=40, alpha=0.7, color=C1, label="Systolic")
    ax.hist(df["diastolic_bp"], bins=40, alpha=0.7, color=C0, label="Diastolic")
    ax.axvline(140, color=CA, ls="--", lw=0.9)
    ax.text(141, ax.get_ylim()[1] * 0.92, "140 mmHg", fontsize=8, color=CA)
    ax.legend(frameon=False)
    _style(ax, "Blood-pressure distribution (cleaned)", "mmHg", "Records")
    fig.tight_layout()
    return fig


def fig_bmi_distribution(df):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(df["bmi"], bins=45, color=C0, alpha=0.85)
    for x, lab in [(18.5, "18.5"), (25, "25"), (30, "30")]:
        ax.axvline(x, color=CA, ls="--", lw=0.9)
        ax.text(x + 0.2, ax.get_ylim()[1] * 0.92, lab, fontsize=8, color=CA)
    _style(ax, "BMI distribution (engineered feature)", "BMI (kg/m²)", "Records")
    fig.tight_layout()
    return fig


def fig_cholesterol_vs_outcome(df):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels, rates, counts = _grouped_rate(df, "cholesterol")
    names = {"1": "Normal", "2": "Above normal", "3": "Well above"}
    _rate_bar(ax, [names[l] for l in labels], rates, counts)
    _style(ax, "CVD rate by cholesterol category", "Cholesterol", "CVD rate (%)")
    fig.tight_layout()
    return fig


def fig_age_vs_outcome(df):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    band = pd.cut(df["age_years"], bins=[29, 35, 40, 45, 50, 55, 60, 65],
                  labels=["30-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65"])
    labels, rates, counts = _grouped_rate(df.assign(age_band=band), "age_band")
    _rate_bar(ax, labels, rates, counts)
    _style(ax, "CVD rate by age band", "Age band (years)", "CVD rate (%)")
    fig.tight_layout()
    return fig


def fig_bp_vs_outcome(df):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    cat = pd.cut(df["systolic_bp"], bins=[0, 119, 129, 139, 159, 400],
                 labels=["<120", "120-129", "130-139", "140-159", "160+"])
    labels, rates, counts = _grouped_rate(df.assign(bp_cat=cat), "bp_cat")
    _rate_bar(ax, labels, rates, counts)
    _style(ax, "CVD rate by systolic blood pressure", "Systolic BP (mmHg)", "CVD rate (%)")
    fig.tight_layout()
    return fig


def fig_lifestyle(df):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    cols = {"smoking": "Smokes", "alcohol": "Drinks alcohol", "active": "Physically active"}
    prev = [df[c].mean() * 100 for c in cols]
    axes[0].bar(list(cols.values()), prev, color=C0)
    for i, p in enumerate(prev):
        axes[0].text(i, p + 1, f"{p:.0f}%", ha="center", fontsize=9)
    axes[0].set_ylim(0, 100)
    _style(axes[0], "Lifestyle prevalence", ylabel="% of records")
    x = np.arange(len(cols))
    r_yes = [df.loc[df[c] == 1, TARGET].mean() * 100 for c in cols]
    r_no = [df.loc[df[c] == 0, TARGET].mean() * 100 for c in cols]
    axes[1].bar(x - 0.2, r_no, 0.4, color=C0, label="No")
    axes[1].bar(x + 0.2, r_yes, 0.4, color=C1, label="Yes")
    axes[1].set_xticks(x, list(cols.values()))
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)
    _style(axes[1], "CVD rate by lifestyle factor", ylabel="CVD rate (%)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


EDA_FIGURES = {
    "eda_01_class_balance": fig_class_balance,
    "eda_02_age_distribution": fig_age_distribution,
    "eda_03_bp_distribution": fig_bp_distribution,
    "eda_04_bmi_distribution": fig_bmi_distribution,
    "eda_05_cholesterol_vs_outcome": fig_cholesterol_vs_outcome,
    "eda_06_age_vs_outcome": fig_age_vs_outcome,
    "eda_07_bp_vs_outcome": fig_bp_vs_outcome,
    "eda_08_lifestyle": fig_lifestyle,
}


def make_all(df: pd.DataFrame) -> dict:
    return {name: fn(df) for name, fn in EDA_FIGURES.items()}


def save_all(df: pd.DataFrame, out_dir) -> None:
    for name, fig in make_all(df).items():
        fig.savefig(out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
