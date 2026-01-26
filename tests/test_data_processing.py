import numpy as np
import pandas as pd
import pytest

from src.preprocessing import FeaturePreprocessor


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


class TestFeaturePreprocessor:
    def test_fit_transform_creates_engineered_features(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        assert "TransactionAmt_log" in result.columns
        assert "TransactionAmt_decimal" in result.columns
        assert "Transaction_hour" in result.columns
        assert "Transaction_day" in result.columns
        assert "card1_addr1_count" in result.columns

    def test_fit_transform_log_amount_non_negative(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        assert (result["TransactionAmt_log"] >= 0).all()

    def test_fit_transform_time_features_valid_range(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        assert result["Transaction_hour"].between(0, 24).all()

    def test_fit_transform_card_addr_count_positive(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        assert (result["card1_addr1_count"] >= 1).all()

    def test_fit_transform_does_not_modify_original(self, sample_transaction_data):
        original_cols = set(sample_transaction_data.columns)
        preprocessor = FeaturePreprocessor()
        preprocessor.fit_transform(sample_transaction_data)
        assert set(sample_transaction_data.columns) == original_cols

    def test_fit_transform_encodes_categoricals(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        assert result["ProductCD"].dtype in [np.int32, np.int64]
        assert "ProductCD" in preprocessor.artifacts.label_encoders

    def test_fit_transform_handles_nan_in_categoricals(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        for col in ["ProductCD", "card4", "card6"]:
            if col in result.columns:
                assert not result[col].isna().any()

    def test_transform_uses_fitted_encoders(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit_transform(sample_transaction_data)
        result = preprocessor.transform(sample_transaction_data)
        assert result["ProductCD"].dtype in [np.int32, np.int64]

    def test_fit_transform_fills_numeric_missing(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        non_target = [c for c in numeric_cols if c != "isFraud"]
        for col in non_target:
            assert not result[col].isna().any(), f"Column {col} still has NaN"

    def test_fit_transform_preserves_target(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        assert "isFraud" in result.columns

    def test_fit_transform_excludes_non_feature_columns(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_transaction_data)
        assert "TransactionID" not in result.columns

    def test_transform_handles_unseen_categories(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit_transform(sample_transaction_data)
        # Create new data with unseen category
        new_data = sample_transaction_data.copy()
        new_data["ProductCD"] = "UNSEEN_CATEGORY"
        # Should not raise, should map to "unknown"
        result = preprocessor.transform(new_data)
        assert not result["ProductCD"].isna().any()

    def test_transform_uses_learned_medians(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit_transform(sample_transaction_data)
        # Create data with all NaN for a numeric column
        new_data = sample_transaction_data.copy()
        new_data["C1"] = np.nan
        result = preprocessor.transform(new_data)
        # Should be filled with the learned median, not 0
        expected_median = preprocessor.artifacts.medians.get("C1", 0.0)
        assert (result["C1"] == expected_median).all()

    def test_transform_raises_when_not_fitted(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        with pytest.raises(RuntimeError):
            preprocessor.transform(sample_transaction_data)

    def test_get_feature_columns_returns_list(self, sample_transaction_data):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit_transform(sample_transaction_data)
        cols = preprocessor.get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert "isFraud" not in cols
