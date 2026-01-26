import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
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
from src.preprocessing import FeaturePreprocessor

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


def process_data() -> tuple[pd.DataFrame, pd.DataFrame, FeaturePreprocessor]:
    """
    Process raw data and return train/test splits with fitted preprocessor.

    The preprocessor is fitted on the training split only (after splitting)
    to avoid data leakage. The test set is then transformed using the
    fitted preprocessor.

    Returns:
        train_df: Processed training data
        test_df: Processed test data
        preprocessor: Fitted FeaturePreprocessor for use in inference
    """
    df = load_raw_data()

    # Split BEFORE preprocessing to avoid data leakage
    train_raw, test_raw = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    logger.info(f"Split data - Train: {len(train_raw)}, Test: {len(test_raw)}")

    # Fit preprocessor on training data only
    preprocessor = FeaturePreprocessor()
    train_df = preprocessor.fit_transform(train_raw)

    # Transform test data using fitted preprocessor
    test_df = preprocessor.transform(test_raw)

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    logger.info(
        f"Fraud rate - Train: {train_df[TARGET].mean():.4f}, "
        f"Test: {test_df[TARGET].mean():.4f}"
    )

    # Save processed data
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_TRAIN_FILE, index=False)
    test_df.to_parquet(PROCESSED_TEST_FILE, index=False)

    # Save reference data for drift monitoring
    reference_sample = train_df.drop(columns=[TARGET]).sample(
        n=min(REFERENCE_SAMPLE_SIZE, len(train_df)), random_state=RANDOM_STATE
    )
    reference_sample.to_parquet(REFERENCE_DATA_FILE, index=False)

    # Save preprocessing artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor.save()

    logger.info(f"Processed data saved to {DATA_PROCESSED_DIR}")
    return train_df, test_df, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_df, test_df, preprocessor = process_data()
