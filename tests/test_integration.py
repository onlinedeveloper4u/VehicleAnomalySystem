"""Integration tests for the FastAPI application."""
import pytest
from fastapi.testclient import TestClient
import os
import sys

# Ensure the project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Fixtures moved to conftest.py


@pytest.fixture
def sample_payload():
    """Return a sample prediction payload."""
    return [
        {
            "engine_id": 1,
            "cycle": 1,
            "setting1": 0.5,
            "setting2": 0.01,
            "setting3": 100.0,
            "s1": 500.0, "s2": 600.0, "s3": 1400.0, "s4": 1200.0, "s5": 14.0,
            "s6": 21.0, "s7": 550.0, "s8": 2300.0, "s9": 8000.0, "s10": 1.3,
            "s11": 47.0, "s12": 520.0, "s13": 2388.0, "s14": 8100.0, "s15": 8.0,
            "s16": 0.03, "s17": 390, "s18": 2388, "s19": 100.0, "s20": 38.0,
            "s21": 23.0
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






