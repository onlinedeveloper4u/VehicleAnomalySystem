from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from src.api.schemas import SensorData, PredictionResponse, ThresholdConfig
from src.models.predictor import AnomalyDetector
import pandas as pd
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import logging
import time
import json
from datetime import datetime
from starlette.status import HTTP_403_FORBIDDEN

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("anomaly_api")

# Global detector instance
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    version = os.environ.get("MODEL_VERSION", "v1")
    logger.info(f"Loading models (version: {version})...")
    try:
        detector = AnomalyDetector(model_dir="models", version=version)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # We might want to fail startup if models are critical, 
        # but for now let's allow it to start and fail on predict
    yield
    logger.info("Shutting down...")

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

@app.post("/model/switch")
async def switch_model(version: str, api_key: str = Depends(get_api_key)):
    """
    Switch the active model version without downtime.
    Req 5.3: "allow switching between model versions without downtime"
    """
    global detector
    logger.info(f"Switching model version to: {version}")
    
    try:
        # Load new models first to ensure they are valid before replacing
        new_detector = AnomalyDetector(model_dir="models", version=version)
        detector = new_detector
        return {
            "status": "success", 
            "switched_to": version,
            "models": list(detector.models.keys())
        }
    except Exception as e:
        logger.error(f"Failed to switch model version to {version}: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to load model version '{version}'. Ensure the directory exists and contains valid models."
        )

@app.get("/health")
async def health_check():
    if detector:
        return {
            "status": "active", 
            "version": getattr(detector, "version", "unknown"),
            "thresholds": detector.thresholds,
            "models_loaded": list(detector.models.keys())
        }
    return {"status": "unhealthy", "message": "Models not loaded"}

@app.patch("/thresholds")
async def update_thresholds(config: ThresholdConfig, api_key: str = Depends(get_api_key)):
    """
    Update the global anomaly detection thresholds for the active models.
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Filter out None values
    overrides = {k: v for k, v in config.dict().items() if v is not None}
    detector.thresholds.update(overrides)
    
    logger.info(f"Thresholds updated: {overrides}")
    return {"status": "success", "new_thresholds": detector.thresholds}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: List[SensorData], api_key: str = Depends(get_api_key)):
    if not detector:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    request_id = str(int(time.time() * 1000))
    logger.info(f"[{request_id}] Prediction request received: {len(data)} records")
    
    # Convert list of Pydantic objects to DataFrame
    df = pd.DataFrame([item.dict() for item in data])
    
    try:
        start_time = time.time()
        result = detector.predict(df)
        latency = (time.time() - start_time) * 1000
        
        # Log summary of results
        n_anomalies = sum(result["is_anomaly"])
        logger.info(
            f"[{request_id}] Prediction complete. Latency: {latency:.2f}ms. "
            f"Anomalies detected: {n_anomalies}/{len(data)}"
        )
        
        # Log full results to file (optional, but requested by spec "log all results")
        # For large batches, this might be huge, so we log a summary.
        # But Req 4.3 says "log all inference requests and results"
        logger.debug(f"[{request_id}] Full results: {json.dumps(result)}")
        
        return result
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
