"""Shared fixtures. Guarantees a model file exists (trains a tiny one if the repo copy is missing)."""
import pytest

from src.config import MODEL_INPUT_COLUMNS, MODEL_PATH, TARGET
from src.data import clean
from src.synthetic import generate_synthetic_cardio


@pytest.fixture(scope="session")
def small_clean_df():
    raw = generate_synthetic_cardio(n=4000, seed=7)
    df, _ = clean(raw)
    return df


@pytest.fixture(scope="session", autouse=True)
def ensure_model(small_clean_df):
    if not MODEL_PATH.exists():
        import joblib
        from sklearn.linear_model import LogisticRegression
        from src.preprocessing import build_pipeline

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        pipe = build_pipeline(LogisticRegression(max_iter=500))
        pipe.fit(small_clean_df[MODEL_INPUT_COLUMNS], small_clean_df[TARGET])
        joblib.dump(pipe, MODEL_PATH)


@pytest.fixture
def valid_payload():
    return {
        "age": 45, "sex": 2, "height": 170, "weight": 78,
        "systolic_bp": 135, "diastolic_bp": 85,
        "cholesterol": 1, "glucose": 1, "smoking": 0, "alcohol": 0, "active": 1,
    }


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the SQLite logger at a throwaway file so tests never touch real data."""
    path = tmp_path / "test_predictions.db"
    monkeypatch.setenv("PULSEPREDICT_DB", str(path))
    return path
