import pytest
import os

# Fixtures from conftest.py are automatically available

def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200

def test_predict_endpoint_unauthorized(test_client):
    """Test access without API key is forbidden."""
    response = test_client.post("/predict", json=[])
    assert response.status_code == 403

def test_predict_endpoint_authorized_but_bad_data(test_client):
    """Test validation error for incomplete data."""
    # Use key defined in conftest.py
    headers = {"X-API-Key": "test-api-key"}
    
    # Missing cycle and engine_id, just s1
    data = [{"s1": 500.0, "setting1": 0.5}] 
    
    response = test_client.post("/predict", json=data, headers=headers)
    assert response.status_code == 422 # Validation Error
