import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
from pydantic import BaseModel
from typing import Dict

ml_models={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model at startup
    print("Loading trained model...")
    ml_models['fraud_model'] = joblib.load("models/fraud_model.pkl")
    yield
    # Cleanup code can go here if needed
    print("Shutting down application...")
    ml_models.clear()


app = FastAPI(
    title="SentinelStream Fraud Engine",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    {"status": "healthy"}
    is_ready = 'fraud_model' in ml_models
    return{"status": "ready" if is_ready else "not ready",
            "model_loaded": is_ready}

class TransactionPayload(BaseModel):
    transaction_id: str
    features: Dict[str, float]  # Assuming features are represented as a dictionary of floats, will hold 30 numerical values

class FraudScoreResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    decision: str
    latency_ms: float

@app.post("/v1/score", response_model=FraudScoreResponse)
def score_transaction(payload: TransactionPayload):
    start_time=time.perf_counter()
    model = ml_models.get('fraud_model')
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    input_df = pd.DataFrame([payload.features])
    input_df=input_df[model.get_booster().feature_names]  # Ensure the input features are in the same order as the model expects
    probabilities=model.predict_proba(input_df)[0]  # Assuming the model has a predict_proba method
    fraud_prob=float(probabilities[1])  # Assuming the second column corresponds to the positive class (fraud)
    is_fraud=fraud_prob>0.5  # Threshold for fraud classification
    decision="DECLINE" if is_fraud else "APPROVE"
    latency_ms=round((time.perf_counter()-start_time)*1000, 2)
    return FraudScoreResponse(
        transaction_id=payload.transaction_id,
        is_fraud=is_fraud,
        fraud_probability=fraud_prob,
        decision=decision,
        latency_ms=latency_ms
    )