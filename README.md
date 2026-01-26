# Fraud Detection MLOps Pipeline

End-to-end ML pipeline for credit card fraud detection (IEEE-CIS dataset). XGBoost model served via FastAPI, tracked with MLflow, monitored with Evidently, containerized with Docker.

## Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Data   │───▶│  Preprocessing   │───▶│    Training     │
│  (CSV)      │    │  (fit on train)  │    │  (XGBoost)      │
└─────────────┘    └──────────────────┘    └─────────────────┘
                           │                        │
                           ▼                        ▼
                   ┌──────────────────┐    ┌─────────────────┐
                   │    Artifacts     │    │     Model       │
                   │ (encoders,       │    │   (joblib)      │
                   │  medians, etc)   │    └─────────────────┘
                   └──────────────────┘             │
                           │                        │
                           ▼                        ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  API Input  │───▶│  Preprocessing   │───▶│   Inference     │───▶ Prediction
│  (JSON)     │    │  (transform)     │    │                 │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │ Drift Monitor   │
                                           │ (Evidently)     │
                                           └─────────────────┘
```

Training and inference use the **same preprocessing artifacts**, avoiding train/serve skew.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
make install
```

Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset and place `train_transaction.csv` and `train_identity.csv` in `data/raw/`.

## Usage

```bash
make train          # process data + train model (logs to MLflow)
make serve          # start API locally
make docker-up      # run API + MLflow UI with docker compose
make test           # run tests
```

## API

Once running, docs at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 500.0,
    "card1": 9999,
    "ProductCD": "W",
    "P_emaildomain": "gmail.com",
    "C1": 3,
    "C2": 1
  }'
```

```json
{"fraud_probability": 0.33, "is_fraud": false, "confidence": 0.67}
```

Other endpoints: `GET /health`, `GET /model-info`

## Stack

- **Model**: XGBoost (AUC ~0.94 on test set)
- **Serving**: FastAPI + Uvicorn
- **Tracking**: MLflow (`make mlflow-ui` or `docker compose up`)
- **Monitoring**: Evidently (data drift detection)
- **CI/CD**: GitHub Actions (lint, test, docker build)
- **Containerization**: Docker + Docker Compose

## Project Structure

```
src/
  config.py            - paths, hyperparams, feature lists
  preprocessing.py     - feature engineering, encoding, imputation (shared by train/inference)
  data_processing.py   - data loading and train/test splitting
  train.py             - training loop with MLflow logging
  predict.py           - inference using saved preprocessing artifacts
  monitor.py           - drift detection with Evidently
  api.py               - FastAPI endpoints
models/                - saved model, preprocessing artifacts, metrics
tests/                 - pytest unit + integration tests
```

## Data & Methodology

- **Leakage prevention**: Train/test split happens *before* fitting encoders and computing statistics (medians, count features)
- **Stratified split**: 80/20 split preserving fraud rate (~3.5%) in both sets
- **Preprocessing artifacts**: Label encoders, median values, and count mappings are saved and reused at inference to ensure consistency
- **Unseen categories**: New categorical values at inference time are mapped to "unknown" (included in training vocabulary)

## CI/CD

On push to `main`: lint (ruff) -> test (pytest) -> docker build + verify.

## Deploy (GCP Cloud Run)

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/fraud-detection-api
gcloud run deploy fraud-detection-api \
  --image gcr.io/PROJECT_ID/fraud-detection-api \
  --port 8000 --memory 1Gi --allow-unauthenticated
```
