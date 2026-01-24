import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS_PATH,
    LABEL_ENCODERS_PATH,
    MODEL_PATH,
    TRAINING_METRICS_PATH,
)

logger = logging.getLogger(__name__)


class FraudPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.label_encoders = None
        self.metrics = None
        self._loaded = False

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training first."
            )
        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                f"Feature columns not found at {FEATURE_COLUMNS_PATH}. "
                "Run training first."
            )

        self.model = joblib.load(MODEL_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        if LABEL_ENCODERS_PATH.exists():
            self.label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        else:
            self.label_encoders = {}

        if TRAINING_METRICS_PATH.exists():
            with open(TRAINING_METRICS_PATH) as f:
                self.metrics = json.load(f)

        self._loaded = True
        logger.info(
            f"Model loaded. Features: {len(self.feature_columns)}, "
            f"AUC: {self.metrics.get('auc_roc', 'N/A')}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, data: dict | pd.DataFrame) -> dict:
        if not self._loaded:
            self.load()

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        for col in CATEGORICAL_FEATURES:
            if col in df.columns and self.label_encoders and col in self.label_encoders:
                le = self.label_encoders[col]
                known = set(le.classes_)
                df[col] = df[col].fillna("unknown").astype(str)
                df[col] = df[col].apply(lambda x, k=known: x if x in k else "unknown")
                df[col] = le.transform(df[col])

        df = df.reindex(columns=self.feature_columns, fill_value=0)
        df = df.fillna(0)

        proba = self.model.predict_proba(df)[:, 1]
        predictions = (proba >= 0.5).astype(int)

        results = []
        for i in range(len(df)):
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
            "n_features": len(self.feature_columns),
            "metrics": self.metrics,
            "model_path": str(MODEL_PATH),
        }


predictor = FraudPredictor()
