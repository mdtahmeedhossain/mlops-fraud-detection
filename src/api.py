from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.predict import predictor

app = FastAPI(
    title="Fraud Detection API",
    description="MLOps fraud detection service with XGBoost model",
    version="1.0.0",
)


class TransactionInput(BaseModel):
    TransactionAmt: float = 100.0
    card1: float = 0
    card2: float = 0
    card3: float = 0
    card4: str = "visa"
    card5: float = 0
    card6: str = "debit"
    addr1: float = 0
    addr2: float = 0
    dist1: float = 0
    C1: float = 0
    C2: float = 0
    C3: float = 0
    C4: float = 0
    C5: float = 0
    C6: float = 0
    C7: float = 0
    C8: float = 0
    C9: float = 0
    C10: float = 0
    C11: float = 0
    C12: float = 0
    C13: float = 0
    C14: float = 0
    D1: float = 0
    D2: float = 0
    D3: float = 0
    D4: float = 0
    D5: float = 0
    D10: float = 0
    D11: float = 0
    D15: float = 0
    ProductCD: str = "W"
    P_emaildomain: str = "gmail.com"

    model_config = {"extra": "allow"}


class PredictionOutput(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    metrics: dict | None


@app.on_event("startup")
async def load_model():
    try:
        predictor.load()
    except FileNotFoundError:
        pass


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=predictor.is_loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    info = predictor.get_model_info()
    return ModelInfoResponse(
        model_type=info["model_type"],
        n_features=info["n_features"],
        metrics=info["metrics"],
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    data = transaction.model_dump()

    result = predictor.predict(data)
    return PredictionOutput(**result)
