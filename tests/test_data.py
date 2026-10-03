import pandas as pd
import pytest

from src.data import RAW_COLUMNS, clean, validate_raw
from src.synthetic import generate_synthetic_cardio


def _valid_row(**overrides):
    row = {"id": 1, "age": 19000, "gender": 1, "height": 165, "weight": 70.0, "ap_hi": 120,
           "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1, "cardio": 0}
    row.update(overrides)
    return row


def test_validate_raw_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate_raw(pd.DataFrame([{"id": 1, "age": 100}]))


def test_validate_raw_rejects_non_binary_target():
    with pytest.raises(ValueError):
        validate_raw(pd.DataFrame([_valid_row(cardio=3)]))


def test_clean_removes_duplicates_ignoring_id():
    raw = pd.DataFrame([_valid_row(id=1), _valid_row(id=2)])
    df, report = clean(raw)
    assert len(df) == 1
    assert report["steps"][0]["rows_removed"] == 1


def test_clean_removes_impossible_blood_pressure():
    raw = pd.DataFrame([
        _valid_row(id=1),
        _valid_row(id=2, ap_hi=16020, height=170),
        _valid_row(id=3, ap_hi=-120, height=171),
        _valid_row(id=4, ap_hi=70, ap_lo=110, height=172),  # diastolic above systolic
    ])
    df, _ = clean(raw)
    assert len(df) == 1


def test_clean_output_schema_and_derived_features():
    raw = pd.DataFrame([_valid_row(age=18262, height=170, weight=80.0)])
    df, _ = clean(raw)
    assert abs(df.loc[0, "age_years"] - 50.0) < 0.05
    assert abs(df.loc[0, "bmi"] - 80 / 1.7 ** 2) < 1e-9
    assert {"systolic_bp", "diastolic_bp", "target"}.issubset(df.columns)


def test_synthetic_data_matches_raw_schema_and_cleans_without_nans():
    raw = generate_synthetic_cardio(n=3000, seed=1)
    assert list(raw.columns) == RAW_COLUMNS
    df, report = clean(raw)
    assert df.isna().sum().sum() == 0
    assert report["rows_clean"] < report["rows_raw"]  # dirt was injected and removed
