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
    models/         # ML model logic (Autoencoder) & predictor
    preprocessing/  # Scaling & transformation logic
    utils/          # Helper utilities
models/             # Stored model version (v1)
data/               # Training datasets (normal/abnormal)
notebooks/          # Documentation & exploration notebooks
tests/              # Unit tests
reports/            # placeholder for evaluation plots
requests/           # Sample JSON for API testing
train.py            # Main training script
evaluate.py         # Performance evaluation script
separate_data.py    # Dataset separation script
render.yaml         # Render deployment config
Dockerfile          # Container configuration
requirements.txt    # Python dependencies
README.md           # Documentation
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

## Deployment to Render

I have provided both a `Dockerfile` and a `render.yaml` (Blueprint) to make deployment seamless.

### Option 1: Using the Blueprint (Recommended)
1. **Push your code** to a GitHub repository.
2. Sign in to your [Render Dashboard](https://dashboard.render.com/).
3. Click **New** > **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically detect the `render.yaml` file and create the service.
6. The `API_KEY` will be automatically generated. You can find it in the service's **Environment** tab.

### Option 2: Manual Web Service
1. **Push your code** to GitHub.
2. In Render, click **New** > **Web Service**.
3. Select your repository.
4. Set **Runtime** to `Docker`.
5. Under **Environment Variables**, add:
   - `API_KEY`: A secret string for authentication.
   - `MODEL_VERSION`: `v1` (or your desired version).
6. Click **Create Web Service**.

Render will build the Docker container and expose the API at `https://your-service-name.onrender.com`.

## Testing
Run unit tests:
```bash
pytest
```
