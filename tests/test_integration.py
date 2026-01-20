"""Integration tests for the FastAPI application."""
import pytest
from fastapi.testclient import TestClient
import os
import sys

# Ensure the project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def test_client():
    """Create a test client with API key set."""
    os.environ["API_KEY"] = "test-api-key"
    os.environ["MODEL_VERSION"] = "v1"
    
    from src.api.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Return authentication headers."""
    return {"X-API-Key": "test-api-key"}


@pytest.fixture
def sample_payload():
    """Return a sample prediction payload."""
    return [
        {
            "Battery_Voltage": 350.0,
            "Battery_Current": 50.0,
            "Battery_Temperature": 30.0,
            "Motor_Temperature": 60.0,
            "Motor_Vibration": 1.0,
            "Motor_Torque": 150.0,
            "Motor_RPM": 3000.0,
            "Power_Consumption": 25.0,
            "Brake_Pressure": 300.0,
            "Tire_Pressure": 35.0,
            "Tire_Temperature": 35.0,
            "Suspension_Load": 1000.0,
            "Ambient_Temperature": 25.0,
            "Ambient_Humidity": 50.0,
            "Driving_Speed": 60.0,
            "Vehicle_ID": "test_vehicle"
        }
    ]


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    
    def test_root_returns_html(self, test_client):
        """Test root endpoint returns HTML landing page."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<title>Vehicle Anomaly Detection System</title>" in response.text
        assert "Vehicle Guard AI" in response.text





class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_returns_html(self, test_client):
        """Test health endpoint returns HTML UI."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "System Health" in response.text
        # Check for status indicator pulse
        assert "animation: pulse" in response.text


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""
    
    def test_predict_requires_auth(self, test_client, sample_payload):
        """Test predict endpoint requires API key."""
        response = test_client.post("/predict", json=sample_payload)
        assert response.status_code == 403
    
    def test_predict_with_valid_auth(self, test_client, auth_headers, sample_payload):
        """Test predict endpoint with valid authentication."""
        response = test_client.post("/predict", json=sample_payload, headers=auth_headers)
        # Either 200 (models loaded) or 503 (models not loaded)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "is_anomaly" in data
            assert "scores" in data
            assert len(data["is_anomaly"]) == 1
    
    def test_predict_with_multiple_records(self, test_client, auth_headers, sample_payload):
        """Test predict endpoint with multiple records."""
        payload = sample_payload * 5
        response = test_client.post("/predict", json=payload, headers=auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["is_anomaly"]) == 5
    
    def test_predict_with_invalid_payload(self, test_client, auth_headers):
        """Test predict endpoint with invalid payload."""
        response = test_client.post(
            "/predict", 
            json=[{"invalid_field": 123}], 
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error






