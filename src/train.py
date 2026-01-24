import json
import logging

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    FEATURE_COLUMNS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PARAMS,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    TARGET,
    TRAINING_METRICS_PATH,
)
from src.data_processing import process_data

logger = logging.getLogger(__name__)


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PROCESSED_TRAIN_FILE.exists() or not PROCESSED_TEST_FILE.exists():
        logger.info("Processed data not found. Running data processing pipeline...")
        return process_data()

    train_df = pd.read_parquet(PROCESSED_TRAIN_FILE)
    test_df = pd.read_parquet(PROCESSED_TEST_FILE)
    return train_df, test_df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> tuple[XGBClassifier, dict]:
    if params is None:
        params = MODEL_PARAMS.copy()

    logger.info(f"Training XGBoost with params: {params}")
    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
    }

    logger.info(f"Metrics: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")
    return model, metrics


def save_artifacts(
    model: XGBClassifier, feature_columns: list[str], metrics: dict
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    with open(TRAINING_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")
    logger.info(f"Metrics saved to {TRAINING_METRICS_PATH}")


def run_training() -> tuple[XGBClassifier, dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df, test_df = load_processed_data()

    feature_columns = [col for col in train_df.columns if col != TARGET]
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    with mlflow.start_run(run_name="xgboost-fraud-detection"):
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("n_features", len(feature_columns))

        model, metrics = train_model(X_train, y_train, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        importance = pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
        mlflow.log_artifact(str(MODELS_DIR / "feature_importance.csv"))

        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    save_artifacts(model, feature_columns, metrics)

    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, metrics = run_training()
    print("\nTraining complete! Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
