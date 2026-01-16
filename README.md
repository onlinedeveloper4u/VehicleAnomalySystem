# Vehicle Sensor Anomaly Detection System

A machine learning system for detecting anomalies in vehicle sensor data.

## Features
- **Data Preprocessing**: Normalization and handling of missing values.
- **Anomaly Detection Models**: Isolation Forest, One-Class SVM, Autoencoder.
- **REST API**: FastAPI-based API for real-time predictions.
- **Deployment**: Dockerized for easy deployment (e.g., Render).

## Project Structure
```text
src/
    api/            # FastAPI application
    models/         # ML model logic (Isolation Forest, SVM, Autoencoder)
    preprocessing/  # Scaling & transformation logic
    utils/          # Visualization & helpers
models/             # Stored .pkl and .pth model files
reports/            # Diagnostic performance plots (PNGs)
data/               # Training data and sample JSONs
train.py            # Main training script
evaluate.py         # Performance evaluation & plot generation
benchmark.py        # Latency benchmarking script [NEW]
exploration.ipynb   # Interactive data exploration & training [NEW]
```

## Setup

1. **Clone the repository**
2. **Create and Activate Virtual Environment**
   It's recommended to use a virtual environment to manage dependencies.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `pip` is not found, try `python3 -m pip install -r requirements.txt`*

## Training Models
Models must be trained before running the API.
```bash
python train.py
```
*Note: This might take a while depending on dataset size.*

## Running the API
```bash
uvicorn src.api.main:app --reload
```
The API will be available at `http://localhost:8000`.

### API Key
Set the `API_KEY` environment variable for authentication.
```bash
export API_KEY="your-secret-key"
```

## Docker Deployment
Build and run the container:
```bash
docker build -t vehicle-anomaly-system .
docker run -p 10000:10000 -e API_KEY=test vehicle-anomaly-system
```

## API Endpoints
- `GET /health`: Check system status.
- `POST /predict`: Submit sensor data for anomaly detection.
  - Header: `X-API-Key: <your-key>`
  - Body: JSON array of sensor records.

## Testing
Run unit tests:
```bash
pytest
```
