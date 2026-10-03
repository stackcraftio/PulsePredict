"""Loading, validating and cleaning the cardiovascular dataset.

Expected raw schema (Kaggle "Cardiovascular Disease dataset", semicolon-separated):
id, age (days), gender (1=female, 2=male), height (cm), weight (kg), ap_hi (systolic),
ap_lo (diastolic), cholesterol (1-3), gluc (1-3), smoke, alco, active, cardio (target)
"""
from __future__ import annotations

import pandas as pd

from src.config import TARGET

RAW_COLUMNS = [
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio",
]

RENAME_MAP = {
    "gender": "sex",
    "ap_hi": "systolic_bp",
    "ap_lo": "diastolic_bp",
    "gluc": "glucose",
    "smoke": "smoking",
    "alco": "alcohol",
    "cardio": TARGET,
}

# Plausibility limits (documented in README / notebook). Values outside these are treated as
# data-entry errors, not as extreme-but-real patients.
LIMITS = {
    "height": (120, 220),        # cm
    "weight": (30, 250),         # kg
    "systolic_bp": (70, 250),    # mmHg
    "diastolic_bp": (40, 150),   # mmHg
    "bmi": (12, 70),             # kg/m^2
}

CLEAN_COLUMNS = [
    "age_years", "sex", "height", "weight", "bmi", "systolic_bp", "diastolic_bp",
    "cholesterol", "glucose", "smoking", "alcohol", "active", TARGET,
]


def load_raw(path) -> pd.DataFrame:
    """Read the raw CSV (semicolon-separated like the Kaggle file; falls back to commas)."""
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def validate_raw(df: pd.DataFrame) -> None:
    """Structural validation. Raises ValueError if the file cannot be used at all."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Raw data contains no rows")
    non_numeric = [c for c in RAW_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Columns must be numeric: {non_numeric}")
    if not set(df["cardio"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column 'cardio' must be binary (0/1)")


def _drop(df: pd.DataFrame, keep: pd.Series, steps: list, rule: str, why: str) -> pd.DataFrame:
    steps.append({"rule": rule, "rationale": why, "rows_removed": int((~keep).sum())})
    return df[keep]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw frame. Returns (clean_df, report). Every removal rule is counted."""
    validate_raw(raw)
    steps: list[dict] = []
    df = raw[RAW_COLUMNS].copy()

    df = _drop(df, ~df.duplicated(subset=[c for c in df.columns if c != "id"]), steps,
               "Duplicate observations (ignoring id)",
               "Identical clinical records with different ids are almost certainly repeats.")
    df = _drop(df, df.notna().all(axis=1), steps, "Missing values",
               "Rows with any missing field cannot be scored reliably.")
    valid_codes = (
        df["gender"].isin([1, 2]) & df["cholesterol"].isin([1, 2, 3]) & df["gluc"].isin([1, 2, 3])
        & df["smoke"].isin([0, 1]) & df["alco"].isin([0, 1]) & df["active"].isin([0, 1])
    )
    df = _drop(df, valid_codes, steps, "Invalid category codes",
               "Categorical fields must hold their documented codes.")

    df = df.rename(columns=RENAME_MAP)
    df["age_years"] = df["age"] / 365.25
    df = df.drop(columns=["id", "age"])

    for col in ["height", "weight"]:
        lo, hi = LIMITS[col]
        df = _drop(df, df[col].between(lo, hi), steps, f"Implausible {col} (outside {lo}-{hi})",
                   "Values like a 55 cm adult height or 10 kg weight are entry errors.")
    for col in ["systolic_bp", "diastolic_bp"]:
        lo, hi = LIMITS[col]
        df = _drop(df, df[col].between(lo, hi), steps,
                   f"Implausible {col} (outside {lo}-{hi} mmHg)",
                   "Negative values and values like 11500 or 16020 are typing errors.")
    df = _drop(df, df["systolic_bp"] > df["diastolic_bp"], steps,
               "Systolic not greater than diastolic",
               "Systolic pressure is physiologically always above diastolic (swapped entries).")

    df["bmi"] = df["weight"] / (df["height"] / 100.0) ** 2
    lo, hi = LIMITS["bmi"]
    df = _drop(df, df["bmi"].between(lo, hi), steps, f"Implausible BMI (outside {lo}-{hi})",
               "Combination of height and weight is not physiologically credible.")

    int_cols = ["sex", "cholesterol", "glucose", "smoking", "alcohol", "active", TARGET,
                "height", "systolic_bp", "diastolic_bp"]
    df[int_cols] = df[int_cols].astype(int)
    df = df[CLEAN_COLUMNS].reset_index(drop=True)

    report = {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "rows_removed_total": int(len(raw) - len(df)),
        "percent_removed": round(100 * (len(raw) - len(df)) / len(raw), 2),
        "steps": steps,
    }
    return df, report


def dataset_summary(raw: pd.DataFrame, clean_df: pd.DataFrame, report: dict) -> dict:
    """Dataset-level facts shown on the dashboard Overview / Data Explorer pages."""
    return {
        "observations_raw": int(len(raw)),
        "observations_clean": int(len(clean_df)),
        "input_features": 11,
        "engineered_features": ["bmi"],
        "missing_values_raw": int(raw.isna().sum().sum()),
        "duplicates_removed": int(report["steps"][0]["rows_removed"]),
        "positive_class_rate": round(float(clean_df[TARGET].mean()), 4),
        "age_range_years": [round(float(clean_df["age_years"].min()), 1),
                            round(float(clean_df["age_years"].max()), 1)],
    }
