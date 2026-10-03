"""Model loading and single-record inference used by the API."""
from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODEL_INPUT_COLUMNS, MODEL_PATH, RISK_THRESHOLD

_pipeline = None


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained pipeline file cannot be found or loaded."""


def load_pipeline(path=MODEL_PATH, force: bool = False):
    """Load the fitted sklearn Pipeline once and cache it."""
    global _pipeline
    if _pipeline is None or force:
        try:
            _pipeline = joblib.load(path)
        except FileNotFoundError as exc:
            raise ModelNotAvailableError(f"Model file not found at {path}. Run `python -m src.train`.") from exc
        except Exception as exc:  # corrupt file, sklearn version mismatch, etc.
            raise ModelNotAvailableError(f"Could not load model: {exc}") from exc
    return _pipeline


def model_is_loaded() -> bool:
    try:
        load_pipeline()
        return True
    except ModelNotAvailableError:
        return False


def risk_category(probability: float) -> str:
    return "Higher-risk profile" if probability >= RISK_THRESHOLD else "Lower-risk profile"


def predict_one(features: dict) -> dict:
    """features must contain every column in MODEL_INPUT_COLUMNS (raw, un-scaled values)."""
    pipeline = load_pipeline()
    frame = pd.DataFrame([{c: features[c] for c in MODEL_INPUT_COLUMNS}])
    probability = float(pipeline.predict_proba(frame)[0, 1])
    prediction = int(pipeline.predict(frame)[0])
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": risk_category(probability),
    }
