"""Tests for the DataPreprocessor class."""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.transformer import DataPreprocessor


def test_preprocessor_init():
    """Test preprocessor initialization with default columns."""
    prep = DataPreprocessor()
    assert len(prep.sensor_columns) == 15


def test_preprocessor_fit_transform():
    """Test fit_transform produces correct output shape."""
    # Create dummy data with realistic ranges
    np.random.seed(42)
    data = pd.DataFrame({
        "Battery_Voltage": np.random.uniform(300, 400, 10),
        "Battery_Current": np.random.uniform(-100, 100, 10),
        "Battery_Temperature": np.random.uniform(20, 40, 10),
        "Motor_Temperature": np.random.uniform(30, 80, 10),
        "Motor_Vibration": np.random.uniform(0.1, 2.0, 10),
        "Motor_Torque": np.random.uniform(50, 200, 10),
        "Motor_RPM": np.random.uniform(1000, 5000, 10),
        "Power_Consumption": np.random.uniform(10, 50, 10),
        "Brake_Pressure": np.random.uniform(100, 500, 10),
        "Tire_Pressure": np.random.uniform(30, 40, 10),
        "Tire_Temperature": np.random.uniform(20, 50, 10),
        "Suspension_Load": np.random.uniform(500, 2000, 10),
        "Ambient_Temperature": np.random.uniform(15, 35, 10),
        "Ambient_Humidity": np.random.uniform(30, 70, 10),
        "Driving_Speed": np.random.uniform(0, 120, 10),
    })
    
    prep = DataPreprocessor()
    transformed = prep.fit_transform(data, window=3)
    
    # 15 raw sensors + (15 sensors * 6 derived features) = 105 total columns
    assert transformed.shape == (10, 105)
    # Check no NaN values
    assert not np.isnan(transformed).any()


def test_preprocessor_missing_cols():
    """Test that missing columns raise KeyError."""
    data = pd.DataFrame({"Battery_Voltage": [1, 2, 3]})
    prep = DataPreprocessor()
    prep.fit(pd.DataFrame({col: [1, 2, 3] for col in prep.sensor_columns}))
    
    with pytest.raises(KeyError):
        prep.transform(data)


def test_preprocessor_save_load(tmp_path):
    """Test saving and loading the preprocessor."""
    np.random.seed(42)
    data = pd.DataFrame({col: np.random.rand(10) for col in [
        "Battery_Voltage", "Battery_Current", "Battery_Temperature",
        "Motor_Temperature", "Motor_Vibration", "Motor_Torque",
        "Motor_RPM", "Power_Consumption", "Brake_Pressure",
        "Tire_Pressure", "Tire_Temperature", "Suspension_Load",
        "Ambient_Temperature", "Ambient_Humidity", "Driving_Speed"
    ]})
    
    prep = DataPreprocessor()
    prep.fit_transform(data)
    
    save_path = tmp_path / "scaler.pkl"
    prep.save(str(save_path))
    
    new_prep = DataPreprocessor()
    new_prep.load(str(save_path))
    
    assert new_prep.feature_columns == prep.feature_columns
