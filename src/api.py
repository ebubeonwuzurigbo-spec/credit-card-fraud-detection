from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib

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