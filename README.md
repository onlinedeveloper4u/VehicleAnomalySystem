# Vehicle Sensor Anomaly Detection System

A machine learning system for detecting anomalies in vehicle sensor data using Isolation Forest.

## Features
- **Anomaly Detection**: Isolation Forest model for detecting sensor anomalies
- **REST API**: FastAPI-based API with rate limiting and authentication
- **Visualization Dashboard**: Streamlit dashboard for monitoring and predictions
- **Prometheus Metrics**: `/metrics` endpoint for monitoring
- **Structured Logging**: JSON-formatted logs for observability

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Start the API
uvicorn src.api.main:app --reload

```

## Project Structure
```
├── src/
│   ├── api/           # FastAPI application
│   ├── models/        # ML model (Isolation Forest)
│   ├── preprocessing/ # Data transformation
│   ├── utils/         # Logging & alerting
│   └── config.py      # Centralized configuration
├── models/            # Trained model versions
├── tests/             # Pytest test suite
├── dashboard.py       # Streamlit visualization
├── train.py           # Training script
├── evaluate.py        # Evaluation script
└── requirements.txt   # Pinned dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System Info & Links |
| `/health` | GET | System health check |
| `/predict` | POST | Anomaly prediction (requires API key) |

| `/model/switch` | POST | Switch model version |


## Configuration

Set environment variables or use a `.env` file:

```bash
API_KEY=your-secret-key
MODEL_VERSION=v1
RATE_LIMIT=100/minute
MAX_RECORDS_PER_REQUEST=1000
```

## Running Tests

```bash
pytest tests/ -v
```

## Docker Deployment

```bash
docker build -t vehicle-anomaly-system .
docker run -p 10000:10000 -e API_KEY=your-key vehicle-anomaly-system
```


