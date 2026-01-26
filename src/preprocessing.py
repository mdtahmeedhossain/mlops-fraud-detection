"""
Preprocessing module for consistent feature engineering between training and inference.

This module ensures that the same transformations applied during training
are replicated exactly during inference, avoiding training/serving skew.
"""

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    LABEL_ENCODERS_PATH,
    NUMERIC_FEATURES,
    PREPROCESSING_ARTIFACTS_PATH,
    TARGET,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingArtifacts:
    """Stores all artifacts needed to replicate preprocessing at inference time."""

    medians: dict[str, float] = field(default_factory=dict)
    card1_addr1_counts: dict[str, int] = field(default_factory=dict)
    label_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)


class FeaturePreprocessor:
    """
    Unified preprocessor for training and inference.

    Ensures consistent feature engineering and imputation between training
    and inference pipelines.
    """

    def __init__(self):
        self.artifacts = PreprocessingArtifacts()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor on training data and transform it.

        This should only be called during training. It learns:
        - Median values for numeric imputation
        - card1_addr1 count mappings
        - Label encodings for categorical features
        """
        df = df.copy()

        # Step 1: Engineer features (before learning medians)
        df = self._engineer_features(df, fit=True)

        # Step 2: Encode categoricals (fit mode)
        df = self._encode_categoricals(df, fit=True)

        # Step 3: Select features
        df = self._select_features(df)

        # Step 4: Learn medians and impute (after feature selection)
        df = self._handle_missing_values(df, fit=True)

        # Store feature columns (excluding target)
        self.artifacts.feature_columns = [
            col for col in df.columns if col != TARGET
        ]

        self._fitted = True
        logger.info(
            f"Preprocessor fitted. Features: {len(self.artifacts.feature_columns)}, "
            f"Medians learned: {len(self.artifacts.medians)}"
        )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        This replicates the exact same transformations used during training,
        using the learned artifacts (medians, count mappings, encoders).
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor not fitted. Call fit_transform first or load artifacts."
            )

        df = df.copy()

        # Apply the same pipeline as training
        df = self._engineer_features(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._select_features(df)
        df = self._handle_missing_values(df, fit=False)

        # Ensure columns match training (handles missing columns)
        df = self._align_columns(df)

        return df

    def _engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create engineered features."""
        # TransactionAmt features
        if "TransactionAmt" in df.columns:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
            df["TransactionAmt_decimal"] = (
                (df["TransactionAmt"] - df["TransactionAmt"].astype(int)) * 1000
            ).astype(int)

        # Time features
        if "TransactionDT" in df.columns:
            df["Transaction_hour"] = (df["TransactionDT"] / 3600) % 24
            df["Transaction_day"] = (df["TransactionDT"] / (3600 * 24)) % 7

        # card1_addr1_count feature
        if "card1" in df.columns and "addr1" in df.columns:
            df["card1_addr1"] = (
                df["card1"].astype(str) + "_" + df["addr1"].astype(str)
            )

            if fit:
                # Learn the count mappings from training data
                count_map = df["card1_addr1"].value_counts().to_dict()
                self.artifacts.card1_addr1_counts = count_map
                df["card1_addr1_count"] = df["card1_addr1"].map(count_map)
            else:
                # Use learned mappings, default to 1 for unseen combinations
                df["card1_addr1_count"] = df["card1_addr1"].map(
                    self.artifacts.card1_addr1_counts
                ).fillna(1).astype(int)

            df.drop("card1_addr1", axis=1, inplace=True)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue

            df[col] = df[col].fillna("unknown").astype(str)

            if fit:
                le = LabelEncoder()
                # Ensure "unknown" is in the classes
                unique_values = list(df[col].unique())
                if "unknown" not in unique_values:
                    unique_values.append("unknown")
                le.fit(unique_values)
                df[col] = le.transform(df[col])
                self.artifacts.label_encoders[col] = le
            else:
                le = self.artifacts.label_encoders.get(col)
                if le is not None:
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else "unknown")
                    df[col] = le.transform(df[col])

        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for modeling."""
        engineered_features = [
            "TransactionAmt_log",
            "TransactionAmt_decimal",
            "Transaction_hour",
            "Transaction_day",
            "card1_addr1_count",
        ]

        all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + engineered_features
        available_features = [col for col in all_features if col in df.columns]

        columns_to_keep = available_features + ([TARGET] if TARGET in df.columns else [])
        return df[columns_to_keep]

    def _handle_missing_values(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """Impute missing values using medians learned from training."""
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == TARGET:
                continue

            if fit:
                # Learn median from training data
                median_val = df[col].median()
                # Handle case where entire column is NaN
                if pd.isna(median_val):
                    median_val = 0.0
                self.artifacts.medians[col] = float(median_val)
                df[col] = df[col].fillna(median_val)
            else:
                # Use learned median, fall back to 0 if not learned
                median_val = self.artifacts.medians.get(col, 0.0)
                df[col] = df[col].fillna(median_val)

        return df

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure inference data has the same columns as training."""
        # Add missing columns with their median values
        for col in self.artifacts.feature_columns:
            if col not in df.columns:
                default_val = self.artifacts.medians.get(col, 0.0)
                df[col] = default_val
                logger.warning(
                    f"Column '{col}' missing in input, filled with {default_val}"
                )

        # Reorder to match training column order
        return df[self.artifacts.feature_columns]

    def save(self, path=None) -> None:
        """Save preprocessing artifacts to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        save_path = path or PREPROCESSING_ARTIFACTS_PATH
        joblib.dump(self.artifacts, save_path)
        logger.info(f"Preprocessing artifacts saved to {save_path}")

        # Also save label encoders separately for backward compatibility
        joblib.dump(self.artifacts.label_encoders, LABEL_ENCODERS_PATH)
        logger.info(f"Label encoders saved to {LABEL_ENCODERS_PATH}")

    def load(self, path=None) -> None:
        """Load preprocessing artifacts from disk."""
        load_path = path or PREPROCESSING_ARTIFACTS_PATH

        if not load_path.exists():
            raise FileNotFoundError(
                f"Preprocessing artifacts not found at {load_path}. Run training first."
            )

        self.artifacts = joblib.load(load_path)
        self._fitted = True
        logger.info(
            f"Preprocessing artifacts loaded. Features: {len(self.artifacts.feature_columns)}"
        )

    def get_feature_columns(self) -> list[str]:
        """Return the list of feature columns expected by the model."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted")
        return self.artifacts.feature_columns
