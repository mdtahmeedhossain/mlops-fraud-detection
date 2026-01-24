import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    encode_categoricals,
    engineer_features,
    handle_missing_values,
    select_features,
)


@pytest.fixture
def sample_transaction_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionDT": np.random.randint(0, 86400 * 30, n),
            "TransactionAmt": np.random.exponential(50, n),
            "isFraud": np.random.binomial(1, 0.03, n),
            "ProductCD": np.random.choice(["W", "C", "R", "H", "S"], n),
            "card1": np.random.randint(1000, 20000, n).astype(float),
            "card2": np.random.choice([111, 222, 333, np.nan], n),
            "card3": np.random.choice([150, 185, np.nan], n),
            "card4": np.random.choice(["visa", "mastercard", "discover", np.nan], n),
            "card5": np.random.choice([100, 200, 300, np.nan], n),
            "card6": np.random.choice(["debit", "credit", np.nan], n),
            "addr1": np.random.choice([300, 200, 100, np.nan], n),
            "addr2": np.random.choice([87, 60, np.nan], n),
            "dist1": np.random.exponential(10, n),
            "dist2": np.random.exponential(50, n),
            "P_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", np.nan], n
            ),
            "R_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", np.nan], n
            ),
            "C1": np.random.randint(0, 10, n).astype(float),
            "C2": np.random.randint(0, 10, n).astype(float),
            "C3": np.random.randint(0, 5, n).astype(float),
            "C4": np.random.randint(0, 5, n).astype(float),
            "C5": np.random.randint(0, 5, n).astype(float),
            "C6": np.random.randint(0, 10, n).astype(float),
            "C7": np.random.randint(0, 5, n).astype(float),
            "C8": np.random.randint(0, 5, n).astype(float),
            "C9": np.random.randint(0, 5, n).astype(float),
            "C10": np.random.randint(0, 5, n).astype(float),
            "C11": np.random.randint(0, 10, n).astype(float),
            "C12": np.random.randint(0, 5, n).astype(float),
            "C13": np.random.randint(0, 30, n).astype(float),
            "C14": np.random.randint(0, 10, n).astype(float),
            "D1": np.random.choice([0, 1, 14, 30, np.nan], n),
            "D2": np.random.choice([0, 1, np.nan], n),
            "D3": np.random.choice([0, 1, np.nan], n),
            "D4": np.random.choice([0, 1, np.nan], n),
            "D5": np.random.choice([0, 1, np.nan], n),
            "D10": np.random.choice([0, 1, np.nan], n),
            "D11": np.random.choice([0, 100, 200, np.nan], n),
            "D15": np.random.choice([0, 100, 200, np.nan], n),
        }
    )


class TestEngineerFeatures:
    def test_creates_log_amount(self, sample_transaction_data):
        result = engineer_features(sample_transaction_data)
        assert "TransactionAmt_log" in result.columns
        assert (result["TransactionAmt_log"] >= 0).all()

    def test_creates_decimal_feature(self, sample_transaction_data):
        result = engineer_features(sample_transaction_data)
        assert "TransactionAmt_decimal" in result.columns

    def test_creates_time_features(self, sample_transaction_data):
        result = engineer_features(sample_transaction_data)
        assert "Transaction_hour" in result.columns
        assert "Transaction_day" in result.columns
        assert result["Transaction_hour"].between(0, 24).all()

    def test_creates_card_addr_count(self, sample_transaction_data):
        result = engineer_features(sample_transaction_data)
        assert "card1_addr1_count" in result.columns
        assert (result["card1_addr1_count"] >= 1).all()

    def test_does_not_modify_original(self, sample_transaction_data):
        original_cols = set(sample_transaction_data.columns)
        engineer_features(sample_transaction_data)
        assert set(sample_transaction_data.columns) == original_cols


class TestEncodeCategoricals:
    def test_encodes_categorical_columns(self, sample_transaction_data):
        result, encoders = encode_categoricals(sample_transaction_data, fit=True)
        assert result["ProductCD"].dtype in [np.int32, np.int64]
        assert "ProductCD" in encoders

    def test_handles_nan_values(self, sample_transaction_data):
        result, _ = encode_categoricals(sample_transaction_data, fit=True)
        # Should not have NaN after encoding
        for col in ["ProductCD", "card4", "card6"]:
            if col in result.columns:
                assert not result[col].isna().any()

    def test_transform_mode_uses_existing_encoders(self, sample_transaction_data):
        _, encoders = encode_categoricals(sample_transaction_data, fit=True)
        result, _ = encode_categoricals(
            sample_transaction_data, encoders=encoders, fit=False
        )
        assert result["ProductCD"].dtype in [np.int32, np.int64]


class TestHandleMissingValues:
    def test_fills_numeric_missing(self, sample_transaction_data):
        result = handle_missing_values(sample_transaction_data)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        non_target = [c for c in numeric_cols if c != "isFraud"]
        for col in non_target:
            assert not result[col].isna().any(), f"Column {col} still has NaN"

    def test_preserves_target(self, sample_transaction_data):
        result = handle_missing_values(sample_transaction_data)
        pd.testing.assert_series_equal(
            result["isFraud"], sample_transaction_data["isFraud"]
        )


class TestSelectFeatures:
    def test_returns_available_features(self, sample_transaction_data):
        result = select_features(sample_transaction_data)
        assert "isFraud" in result.columns
        assert "TransactionAmt" in result.columns

    def test_excludes_non_feature_columns(self, sample_transaction_data):
        result = select_features(sample_transaction_data)
        assert "TransactionID" not in result.columns
