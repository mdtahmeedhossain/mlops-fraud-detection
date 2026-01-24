import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    LABEL_ENCODERS_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    RANDOM_STATE,
    REFERENCE_DATA_FILE,
    REFERENCE_SAMPLE_SIZE,
    TARGET,
    TEST_SIZE,
    TRAIN_IDENTITY_FILE,
    TRAIN_TRANSACTION_FILE,
)

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    logger.info("Loading raw transaction data...")
    transactions = pd.read_csv(TRAIN_TRANSACTION_FILE)
    logger.info(f"Transactions shape: {transactions.shape}")

    if TRAIN_IDENTITY_FILE.exists():
        logger.info("Loading identity data...")
        identity = pd.read_csv(TRAIN_IDENTITY_FILE)
        logger.info(f"Identity shape: {identity.shape}")
        df = transactions.merge(identity, on="TransactionID", how="left")
    else:
        df = transactions

    logger.info(f"Merged data shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = (
        (df["TransactionAmt"] - df["TransactionAmt"].astype(int)) * 1000
    ).astype(int)

    if "TransactionDT" in df.columns:
        df["Transaction_hour"] = (df["TransactionDT"] / 3600) % 24
        df["Transaction_day"] = (df["TransactionDT"] / (3600 * 24)) % 7

    if "card1" in df.columns and "addr1" in df.columns:
        df["card1_addr1"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        df["card1_addr1_count"] = df.groupby("card1_addr1")[
            "card1_addr1"
        ].transform("count")
        df.drop("card1_addr1", axis=1, inplace=True)

    if "P_emaildomain" in df.columns:
        df["P_emaildomain_suffix"] = (
            df["P_emaildomain"]
            .fillna("unknown")
            .apply(lambda x: x.split(".")[-1] if isinstance(x, str) else "unknown")
        )

    return df


def encode_categoricals(
    df: pd.DataFrame, encoders: dict | None = None, fit: bool = True
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("unknown").astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "unknown")
                df[col] = le.transform(df[col])

    return df, encoders


def select_features(df: pd.DataFrame) -> pd.DataFrame:
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


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != TARGET:
            df[col] = df[col].fillna(df[col].median())
    return df


def process_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw_data()
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, fit=True)
    df = select_features(df)
    df = handle_missing_values(df)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    logger.info(
        f"Fraud rate - Train: {train_df[TARGET].mean():.4f}, "
        f"Test: {test_df[TARGET].mean():.4f}"
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_TRAIN_FILE, index=False)
    test_df.to_parquet(PROCESSED_TEST_FILE, index=False)

    reference_sample = train_df.drop(columns=[TARGET]).sample(
        n=min(REFERENCE_SAMPLE_SIZE, len(train_df)), random_state=RANDOM_STATE
    )
    reference_sample.to_parquet(REFERENCE_DATA_FILE, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, LABEL_ENCODERS_PATH)
    logger.info(f"Label encoders saved to {LABEL_ENCODERS_PATH}")

    logger.info(f"Processed data saved to {DATA_PROCESSED_DIR}")
    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_data()
