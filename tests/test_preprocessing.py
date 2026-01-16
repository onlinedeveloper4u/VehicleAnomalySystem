import pytest
import pandas as pd
import numpy as np
from src.preprocessing.transformer import DataPreprocessor

def test_preprocessor_init():
    prep = DataPreprocessor()
    assert len(prep.sensor_columns) == 15

def test_preprocessor_fit_transform():
    # Create dummy data
    data = pd.DataFrame(np.random.rand(10, 15), columns=[
        "Battery_Voltage", "Battery_Current", "Battery_Temperature",
        "Motor_Temperature", "Motor_Vibration", "Motor_Torque",
        "Motor_RPM", "Power_Consumption", "Brake_Pressure",
        "Tire_Pressure", "Tire_Temperature", "Suspension_Load",
        "Ambient_Temperature", "Ambient_Humidity", "Driving_Speed"
    ])
    
    prep = DataPreprocessor()
    transformed = prep.fit_transform(data)
    
    assert transformed.shape == (10, 15)
    # MinMax scaler should put values between 0 and 1
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.00001 # floating point tolerance

def test_preprocessor_missing_cols():
    data = pd.DataFrame({"Battery_Voltage": [1, 2, 3]})
    prep = DataPreprocessor()
    with pytest.raises(ValueError):
        prep.transform(data)
