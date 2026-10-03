"""PulsePredict FastAPI backend.   Run:  uvicorn api.main:app --reload"""
from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from src.config import MODEL_VERSION
from src.database import get_recent, get_stats, init_db, log_prediction
from src.predict import ModelNotAvailableError, model_is_loaded, predict_one

app = FastAPI(
    title="PulsePredict API",
    version=MODEL_VERSION,
    description="Cardiovascular risk-profile model served via FastAPI. "
                "Educational portfolio project – not a medical diagnostic system.",
)


@app.get("/")
def root() -> dict:
    return {
        "name": "PulsePredict API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/stats"],
        "disclaimer": "Educational portfolio project. Not a medical device.",
    }


@app.get("/health")
def health() -> dict:
    loaded = model_is_loaded()
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = predict_one(payload.to_model_input())
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    init_db()
    log_prediction(result["prediction"], result["probability"], MODEL_VERSION)
    return PredictionResponse(**result, model_version=MODEL_VERSION)


@app.get("/stats")
def stats() -> dict:
    """Small internal analytics view over the SQLite prediction log."""
    init_db()
    return {**get_stats(), "recent": get_recent(10)}
