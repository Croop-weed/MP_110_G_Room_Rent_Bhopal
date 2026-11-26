"""FastAPI endpoint tests using dependency injection instead of a live MLflow server."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeModel:
    def predict(self, frame):
        return [5500.0]


@pytest.fixture
def client():
    app.state.model_loader = Mock()
    app.state.model_loader.load.return_value = FakeModel()
    with TestClient(app) as test_client:
        yield test_client


def valid_payload():
    return {
        "distance_to_campus_km": 5.0,
        "room_area_sqft": 300,
        "sharing": 2,
        "meals_per_day": 2,
        "building_age_yr": 10,
        "wifi_mbps": 100,
        "furnishing": "semi",
        "locality": "Sector East",
    }


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_successful_prediction(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    assert response.json() == {"prediction": 5500.0}


def test_missing_field(client):
    payload = valid_payload()
    del payload["wifi_mbps"]
    assert client.post("/predict", json=payload).status_code == 422


def test_extra_field(client):
    payload = valid_payload()
    payload["record_id"] = "R1"
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "field,value", [("room_area_sqft", 651), ("building_age_yr", -1)]
)
def test_mandatory_numeric_validation(client, field, value):
    payload = valid_payload()
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


def test_invalid_data_type(client):
    payload = valid_payload()
    payload["sharing"] = "two"
    assert client.post("/predict", json=payload).status_code == 422


def test_model_loading_failure():
    from app.model_loader import ModelLoadingError

    loader = Mock()
    loader.load.side_effect = ModelLoadingError("missing")
    app.state.model_loader = loader
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=valid_payload())
    assert response.status_code == 503


def test_prediction_failure():
    model = Mock()
    model.predict.side_effect = RuntimeError("boom")
    loader = Mock()
    loader.load.return_value = model
    app.state.model_loader = loader
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=valid_payload())
    assert response.status_code == 500
