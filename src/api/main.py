from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse, HTMLResponse
from src.api.schemas import SensorData, PredictionResponse, ThresholdConfig, HealthResponse
from src.models.predictor import AnomalyDetector
from src.config import get_settings
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
import logging
import time
import json
from datetime import datetime
from starlette.status import HTTP_403_FORBIDDEN
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load settings from .env file
settings = get_settings()

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("anomaly_api")

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Global detector instance
detector: AnomalyDetector | None = None




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for model loading."""
    global detector
    logger.info(f"System startup. Loading models (version: {settings.model_version})...")
    try:
        detector = AnomalyDetector(model_dir=settings.model_dir, version=settings.model_version)
        logger.info(f"Successfully loaded model version: {settings.model_version}")
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to load models. Error: {e}")
        logger.warning("The API will start but /predict will return 503 until models are loaded.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Anomaly detection for vehicle sensor data using Isolation Forest",
    lifespan=lifespan
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Validate API key from request header."""
    if settings.api_key and api_key_header == settings.api_key:
        return api_key_header
    elif not settings.api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="API Key configuration missing on server"
        )
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )






@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with beautiful UI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vehicle Anomaly Detection System</title>
        <style>
            body {
                margin: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            .container {
                text-align: center;
                max-width: 800px;
                padding: 2rem;
                animation: fadeIn 0.8s ease-out;
            }
            h1 {
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                background: linear-gradient(to right, #60a5fa, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.05em;
            }
            p {
                font-size: 1.25rem;
                color: #94a3b8;
                margin-bottom: 3rem;
                line-height: 1.6;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
            }
            .card {
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 1rem;
                padding: 2rem;
                text-decoration: none;
                color: #e2e8f0;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                flex-direction: column;
                align-items: center;
                cursor: pointer;
            }
            .card:hover {
                transform: translateY(-5px);
                border-color: #60a5fa;
                box-shadow: 0 10px 30px -10px rgba(96, 165, 250, 0.3);
                background: rgba(30, 41, 59, 0.9);
            }
            .icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            .card-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #f8fafc;
            }
            .card-desc {
                font-size: 0.9rem;
                color: #94a3b8;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vehicle Guard AI</h1>
            <p>Advanced real-time sensor monitoring and anomaly detection system.<br>Powered by Isolation Forest.</p>
            
            <div class="grid">
                <a href="/docs" class="card">
                    <div class="icon">📚</div>
                    <div class="card-title">API Documentation</div>
                    <div class="card-desc">Interactive Swagger UI for testing predictions.</div>
                </a>
                
                <a href="/health" class="card">
                    <div class="icon">💓</div>
                    <div class="card-title">System Health</div>
                    <div class="card-desc">Check model status, version, and latency.</div>
                </a>
                
                <a href="/redoc" class="card">
                    <div class="icon">📄</div>
                    <div class="card-title">ReDoc Specification</div>
                    <div class="card-desc">Detailed API interface documentation.</div>
                </a>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/model/switch")
async def switch_model(version: str, api_key: str = Depends(get_api_key)):
    """Switch the active model version without downtime."""
    global detector
    logger.info(f"Switching model version to: {version}")
    
    try:
        new_detector = AnomalyDetector(model_dir=settings.model_dir, version=version)
        detector = new_detector
        return {
            "status": "success", 
            "switched_to": version,
            "threshold": detector.threshold
        }
    except Exception as e:
        logger.error(f"Failed to switch model version to {version}: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to load model version '{version}'. Ensure the directory exists."
        )


@app.get("/health", response_class=HTMLResponse)
async def health_check():
    """Check system health and model status with UI."""
    status = "active" if detector else "unhealthy"
    status_color = "#4ade80" if status == "active" else "#ef4444"
    status_text = "System Operational" if status == "active" else "System Degraded"
    
    version = detector.version if detector else "Unknown"
    threshold = str(detector.threshold) if detector else "N/A"
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>System Health | Vehicle Guard AI</title>
        <style>
            body {{
                margin: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .card {{
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 1.5rem;
                padding: 3rem;
                width: 400px;
                text-align: center;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }}
            .status-indicator {{
                width: 20px;
                height: 20px;
                background-color: {status_color};
                border-radius: 50%;
                display: inline-block;
                margin-right: 10px;
                box-shadow: 0 0 20px {status_color};
                animation: pulse 2s infinite;
            }}
            h1 {{
                font-size: 2rem;
                margin: 0 0 2rem 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-bottom: 2rem;
            }}
            .metric {{
                background: rgba(15, 23, 42, 0.5);
                padding: 1rem;
                border-radius: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }}
            .label {{
                font-size: 0.8rem;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }}
            .value {{
                font-size: 1.25rem;
                font-weight: 600;
                color: #f8fafc;
            }}
            .btn {{
                display: inline-block;
                padding: 0.75rem 1.5rem;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                text-decoration: none;
                border-radius: 9999px;
                transition: all 0.2s;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .btn:hover {{
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-2px);
            }}
            @keyframes pulse {{
                0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba({status_color}77, 0.7); }}
                70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba({status_color}77, 0); }}
                100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba({status_color}77, 0); }}
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1><div class="status-indicator"></div> {status_text}</h1>
            
            <div class="metric-grid">
                <div class="metric">
                    <div class="label">Model Version</div>
                    <div class="value">{version}</div>
                </div>
                <div class="metric">
                    <div class="label">Threshold</div>
                    <div class="value">{threshold}</div>
                </div>
            </div>
            
            <a href="/" class="btn">Back to Home</a>
        </div>
    </body>
    </html>
    """








@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(settings.rate_limit)
async def predict(
    request: Request,
    data: List[SensorData],
    api_key: str = Depends(get_api_key)
):
    """
    Predict anomalies for incoming sensor data.
    
    Rate limited and payload size validated.
    """

    
    if not detector:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Payload size validation
    if len(data) > settings.max_records_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Payload too large. Maximum {settings.max_records_per_request} records per request."
        )
    
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty payload. Provide at least one record.")
    
    request_id = str(int(time.time() * 1000))
    logger.info(f"[{request_id}] Prediction request received: {len(data)} records")
    
    # Convert list of Pydantic objects to DataFrame
    df = pd.DataFrame([item.model_dump() for item in data])
    
    try:
        start_time = time.time()
        result = detector.predict(df)
        latency = (time.time() - start_time) * 1000
        
        n_anomalies = sum(result["is_anomaly"])
        
        # Log result

        
        # Structured JSON Log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "n_records": len(data),
            "n_anomalies": n_anomalies,
            "latency_ms": round(latency, 2),
            "version": detector.version
        }
        logger.info(f"INFERENCE_RESULT: {json.dumps(log_entry)}")
        
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
