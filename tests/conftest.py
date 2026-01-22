"""Pytest configuration for test discovery and fixtures."""
import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Environment Variables for Testing (Must be before app import)
os.environ["API_KEY"] = "test-api-key"
os.environ["MODEL_VERSION"] = "v1"


@pytest.fixture(scope="module")
def test_client():
    """Create a test client with API key set."""
    from src.api.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Return authentication headers."""
    return {"X-API-Key": "test-api-key"}
