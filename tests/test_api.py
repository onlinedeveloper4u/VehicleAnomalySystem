import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import os
import json

# Set API Key for testing
os.environ["API_KEY"] = "test-key"

client = TestClient(app)

def test_health_check_no_models():
    # Since we haven't trained models in this test environment, it might report unhealthy if models missing
    # But detector loads whatever is in "models" dir. If empty, it loads fine but dicts are empty?
    # Trainer needs to be run first for full test. 
    # But let's check basic connectivity.
    response = client.get("/health")
    assert response.status_code == 200
    # Response detail depends on whether models exist on disk

def test_predict_endpoint_unauthorized():
    response = client.post("/predict", json=[])
    assert response.status_code == 403

def test_predict_endpoint_authorized_but_bad_data():
    headers = {"X-API-Key": "test-key"}
    # Missing fields
    data = [{"Battery_Voltage": 12.0}] 
    response = client.post("/predict", json=data, headers=headers)
    assert response.status_code == 422 # Validation Error

# We need a fixture to train models or mock them for full prediction test
# For now, let's assume we can run the integration test if models exist.
