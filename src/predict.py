import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_PATH,
    TRAINING_METRICS_PATH,
)
from src.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Fraud prediction service that uses the same preprocessing pipeline as training.

    This ensures consistency between training and inference, avoiding the
    training/serving skew that can degrade model performance in production.
    """

    def __init__(self):
        self.model = None
        self.preprocessor: FeaturePreprocessor | None = None
        self.metrics: dict | None = None
        self._loaded = False

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training first."
            )

        self.model = joblib.load(MODEL_PATH)

        # Load the preprocessor with all artifacts
        self.preprocessor = FeaturePreprocessor()
        self.preprocessor.load()

        if TRAINING_METRICS_PATH.exists():
            with open(TRAINING_METRICS_PATH) as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {}

        self._loaded = True
        logger.info(
            f"Model loaded. Features: {len(self.preprocessor.get_feature_columns())}, "
            f"AUC: {self.metrics.get('auc_roc', 'N/A')}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, data: dict | pd.DataFrame) -> dict | list[dict]:
        """
        Make fraud predictions on input data.

        The input data goes through the same preprocessing pipeline used during
        training, ensuring consistent feature engineering and imputation.

        Args:
            data: Either a dict (single transaction) or DataFrame (batch)

        Returns:
            Prediction result(s) with fraud_probability, is_fraud, and confidence
        """
        if not self._loaded:
            self.load()

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Apply the same preprocessing as training
        df_processed = self.preprocessor.transform(df)

        proba = self.model.predict_proba(df_processed)[:, 1]
        predictions = (proba >= 0.5).astype(int)

        results = []
        for i in range(len(df_processed)):
            results.append(
                {
                    "fraud_probability": float(np.round(proba[i], 4)),
                    "is_fraud": bool(predictions[i]),
                    "confidence": float(
                        np.round(max(proba[i], 1 - proba[i]), 4)
                    ),
                }
            )

        return results[0] if len(results) == 1 else results

    def get_model_info(self) -> dict:
        if not self._loaded:
            self.load()

        return {
            "model_type": "XGBClassifier",
            "n_features": len(self.preprocessor.get_feature_columns()),
            "feature_columns": self.preprocessor.get_feature_columns(),
            "metrics": self.metrics,
            "model_path": str(MODEL_PATH),
        }


predictor = FraudPredictor()
