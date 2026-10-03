import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_db(temp_db):
    yield


def test_root_returns_api_info():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "PulsePredict API"


def test_health_endpoint_reports_model_loaded():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_loaded": True}


def test_valid_prediction_returns_200_and_required_fields(valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert {"prediction", "risk_category", "probability", "model_version"}.issubset(body)
    assert body["prediction"] in (0, 1)


def test_probability_is_between_zero_and_one(valid_payload):
    body = client.post("/predict", json=valid_payload).json()
    assert 0.0 <= body["probability"] <= 1.0


def test_negative_age_is_rejected_with_422(valid_payload):
    valid_payload["age"] = -400
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_missing_field_is_rejected_with_422(valid_payload):
    del valid_payload["systolic_bp"]
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_invalid_category_is_rejected_with_422(valid_payload):
    valid_payload["cholesterol"] = 7
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_systolic_must_exceed_diastolic(valid_payload):
    valid_payload["systolic_bp"], valid_payload["diastolic_bp"] = 80, 90
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_successful_prediction_is_logged_to_sql(valid_payload):
    assert client.get("/stats").json()["predictions_made"] == 0
    client.post("/predict", json=valid_payload)
    stats = client.get("/stats").json()
    assert stats["predictions_made"] == 1
    assert len(stats["recent"]) == 1


def test_rejected_request_is_not_logged(valid_payload):
    valid_payload["age"] = -1
    client.post("/predict", json=valid_payload)
    assert client.get("/stats").json()["predictions_made"] == 0
