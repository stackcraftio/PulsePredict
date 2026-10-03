import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_INPUT_COLUMNS, MODEL_PATH, TARGET
from src.predict import load_pipeline, predict_one, risk_category
from src.preprocessing import add_bmi, build_pipeline


def test_add_bmi_formula():
    out = add_bmi(pd.DataFrame({"weight": [81.0], "height": [180.0]}))
    assert abs(out.loc[0, "bmi"] - 25.0) < 1e-9


def test_pipeline_fits_and_returns_valid_probabilities(small_clean_df):
    X, y = small_clean_df[MODEL_INPUT_COLUMNS], small_clean_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))
    assert set(pipe.predict(X)).issubset({0, 1})


def test_saved_pipeline_loads_and_is_a_full_pipeline():
    pipe = joblib.load(MODEL_PATH)
    assert list(pipe.named_steps) == ["features", "preprocess", "model"]


def test_predict_one_returns_expected_fields():
    load_pipeline(force=True)
    result = predict_one({
        "age_years": 55, "sex": 1, "height": 160, "weight": 90, "systolic_bp": 150,
        "diastolic_bp": 95, "cholesterol": 3, "glucose": 2, "smoking": 0, "alcohol": 0, "active": 0,
    })
    assert set(result) == {"prediction", "probability", "risk_category"}
    assert 0 <= result["probability"] <= 1


def test_risk_category_threshold():
    assert risk_category(0.5) == "Higher-risk profile"
    assert risk_category(0.49) == "Lower-risk profile"
