import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.predict import FraudPredictor


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


@pytest.fixture
def mock_feature_columns():
    return ["TransactionAmt", "card1", "card2", "C1", "C2", "D1"]


@pytest.fixture
def predictor_with_model(mock_model, mock_feature_columns):
    pred = FraudPredictor()
    pred.model = mock_model
    pred.feature_columns = mock_feature_columns
    pred.metrics = {"auc_roc": 0.95, "f1": 0.6}
    pred._loaded = True
    return pred


class TestFraudPredictor:
    def test_predict_returns_correct_structure(self, predictor_with_model):
        data = {"TransactionAmt": 100.0, "card1": 1234, "C1": 1}
        result = predictor_with_model.predict(data)

        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "confidence" in result

    def test_predict_high_fraud_probability(self, predictor_with_model):
        data = {"TransactionAmt": 500.0, "card1": 9999}
        result = predictor_with_model.predict(data)

        assert result["fraud_probability"] == 0.8
        assert result["is_fraud"] is True
        assert result["confidence"] == 0.8

    def test_predict_low_fraud_probability(self, predictor_with_model, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        data = {"TransactionAmt": 20.0}
        result = predictor_with_model.predict(data)

        assert result["fraud_probability"] == 0.1
        assert result["is_fraud"] is False
        assert result["confidence"] == 0.9

    def test_predict_handles_missing_features(self, predictor_with_model):
        data = {"TransactionAmt": 50.0}
        result = predictor_with_model.predict(data)
        assert "fraud_probability" in result

    def test_predict_handles_extra_features(self, predictor_with_model):
        data = {"TransactionAmt": 50.0, "extra_feature": 999, "another": "value"}
        result = predictor_with_model.predict(data)
        assert "fraud_probability" in result

    def test_predict_with_dataframe(self, predictor_with_model):
        df = pd.DataFrame([{"TransactionAmt": 100.0, "card1": 5000}])
        result = predictor_with_model.predict(df)
        assert "fraud_probability" in result

    def test_get_model_info(self, predictor_with_model):
        info = predictor_with_model.get_model_info()
        assert info["model_type"] == "XGBClassifier"
        assert info["n_features"] == 6
        assert info["metrics"]["auc_roc"] == 0.95

    def test_is_loaded_property(self):
        pred = FraudPredictor()
        assert pred.is_loaded is False

    def test_predict_raises_when_no_model(self, tmp_path):
        pred = FraudPredictor()
        with patch("src.predict.MODEL_PATH", tmp_path / "nonexistent.joblib"):
            with pytest.raises(FileNotFoundError):
                pred.predict({"TransactionAmt": 100})
