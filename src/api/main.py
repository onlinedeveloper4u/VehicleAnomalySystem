from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from src.api.schemas import SensorData, PredictionResponse
from src.models.predictor import AnomalyDetector
import pandas as pd
import os
from contextlib import asynccontextmanager
from typing import List

# Global detector instance
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    print("Loading models...")
    try:
        detector = AnomalyDetector(model_dir="models")
    except Exception as e:
        print(f"Error loading models: {e}")
        # We might want to fail startup if models are critical, 
        # but for now let's allow it to start and fail on predict
    yield
    print("Shutting down...")

app = FastAPI(title="Vehicle Anomaly Detection System", version="1.0", lifespan=lifespan)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    expected_key = os.environ.get("API_KEY")
    if expected_key and api_key_header == expected_key:
        return api_key_header
    elif not expected_key:
        # If no API key set in env, allow open access (dev mode) or warn?
        # Requirement says "API access shall be protected".
        # So we should block.
        raise HTTPException(
            status_code=403, detail="API Key configuration missing on server"
        )
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )

@app.get("/")
async def root():
    return {
        "message": "Vehicle Anomaly Detection System API",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
async def health_check():
    if detector:
        return {"status": "active", "models_loaded": list(detector.models.keys())}
    return {"status": "unhealthy", "message": "Models not loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: List[SensorData], api_key: str = Depends(get_api_key)):
    if not detector:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Convert list of Pydantic objects to DataFrame
    df = pd.DataFrame([item.dict() for item in data])
    
    try:
        result = detector.predict(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
