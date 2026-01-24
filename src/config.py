from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_TRANSACTION_FILE = DATA_RAW_DIR / "train_transaction.csv"
TRAIN_IDENTITY_FILE = DATA_RAW_DIR / "train_identity.csv"
PROCESSED_TRAIN_FILE = DATA_PROCESSED_DIR / "train.parquet"
PROCESSED_TEST_FILE = DATA_PROCESSED_DIR / "test.parquet"
REFERENCE_DATA_FILE = DATA_PROCESSED_DIR / "reference_data.parquet"

MODEL_PATH = MODELS_DIR / "model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"
TRAINING_METRICS_PATH = MODELS_DIR / "metrics.json"

NUMERIC_FEATURES = [
    "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V44", "V45", "V46", "V47", "V48", "V49", "V50", "V51", "V52", "V53",
    "V54", "V55", "V56", "V57", "V58", "V59", "V60", "V61", "V62", "V63",
    "V64", "V65", "V66", "V67", "V68", "V69", "V70", "V71", "V72", "V73",
    "V74", "V75", "V76", "V77", "V78", "V79", "V80", "V81", "V82", "V83",
    "V84", "V85", "V86", "V87", "V88", "V89", "V90", "V91", "V92", "V93",
    "V94", "V95", "V96", "V97", "V98", "V99", "V100",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain",
    "R_emaildomain",
]

TARGET = "isFraud"
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 20,
    "random_state": RANDOM_STATE,
    "eval_metric": "auc",
    "n_jobs": -1,
}

MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "fraud-detection"

API_HOST = "0.0.0.0"
API_PORT = 8000

DRIFT_THRESHOLD = 0.05
REFERENCE_SAMPLE_SIZE = 5000
