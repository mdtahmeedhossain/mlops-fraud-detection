import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api import app
from src.predict import FraudPredictor


@pytest.fixture
def mock_predictor():
    pred = MagicMock(spec=FraudPredictor)
    pred.is_loaded = True
    pred.predict.return_value = {
        "fraud_probability": 0.75,
        "is_fraud": True,
        "confidence": 0.75,
    }
    pred.get_model_info.return_value = {
        "model_type": "XGBClassifier",
        "n_features": 100,
        "metrics": {"auc_roc": 0.95, "f1": 0.6},
    }
    return pred


@pytest.fixture
def client(mock_predictor):
    with patch("src.api.predictor", mock_predictor):
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "XGBClassifier"
        assert data["n_features"] == 100
        assert data["metrics"]["auc_roc"] == 0.95

    def test_model_info_not_loaded(self, mock_predictor):
        mock_predictor.is_loaded = False
        with patch("src.api.predictor", mock_predictor):
            client = TestClient(app)
            response = client.get("/model-info")
            assert response.status_code == 503


class TestPredictEndpoint:
    def test_predict_fraud(self, client):
        payload = {
            "TransactionAmt": 500.0,
            "card1": 9999,
            "ProductCD": "W",
            "P_emaildomain": "gmail.com",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "confidence" in data
        assert data["is_fraud"] is True

    def test_predict_with_minimal_data(self, client):
        payload = {"TransactionAmt": 10.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_with_default_values(self, client):
        payload = {}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_model_not_loaded(self, mock_predictor):
        mock_predictor.is_loaded = False
        with patch("src.api.predictor", mock_predictor):
            client = TestClient(app)
            response = client.post(
                "/predict", json={"TransactionAmt": 100.0}
            )
            assert response.status_code == 503

    def test_predict_returns_valid_probability(self, client):
        payload = {"TransactionAmt": 250.0, "card1": 5000}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert 0 <= data["fraud_probability"] <= 1
        assert 0.5 <= data["confidence"] <= 1.0
