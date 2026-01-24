import logging

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PARAMS,
    RANDOM_STATE,
    TARGET,
)
from src.train import load_processed_data

logger = logging.getLogger(__name__)


MODELS = {
    "logistic-regression": {
        "class": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
    },
    "random-forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": MODEL_PARAMS.copy(),
    },
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def run_comparison():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df, test_df = load_processed_data()

    feature_columns = [col for col in train_df.columns if col != TARGET]
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    results = {}

    for name, spec in MODELS.items():
        logger.info(f"Training {name}...")

        with mlflow.start_run(run_name=f"compare-{name}"):
            mlflow.log_params(spec["params"])
            mlflow.log_param("model_type", name)
            mlflow.log_param("n_features", len(feature_columns))

            model = spec["class"](**spec["params"])

            if name == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
            else:
                model.fit(X_train, y_train)

            metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            results[name] = metrics
            logger.info(f"  {name}: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

    summary = pd.DataFrame(results).T.sort_values("auc_roc", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_comparison()
