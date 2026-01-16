import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

class DataPreprocessor:
    def __init__(self, sensor_columns=None):
        if sensor_columns is None:
            self.sensor_columns = [
                "Battery_Voltage", "Battery_Current", "Battery_Temperature",
                "Motor_Temperature", "Motor_Vibration", "Motor_Torque",
                "Motor_RPM", "Power_Consumption", "Brake_Pressure",
                "Tire_Pressure", "Tire_Temperature", "Suspension_Load",
                "Ambient_Temperature", "Ambient_Humidity", "Driving_Speed"
            ]
        else:
            self.sensor_columns = sensor_columns
        
        self.scaler = MinMaxScaler()

    def fit(self, data: pd.DataFrame):
        """Fits the scaler on the provided data."""
        # Ensure we only work with relevant columns
        data_filtered = data[self.sensor_columns]
        
        # Handle missing values (fit logic might differ, but for MinMax it's just min/max)
        # However, to be safe, we should probably fill na before fitting
        data_filled = data_filtered.ffill().bfill()
        
        self.scaler.fit(data_filled)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms the data using the fitted scaler."""
        # Ensure we only work with relevant columns
        # Check if columns exist
        missing_cols = [col for col in self.sensor_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
            
        data_filtered = data[self.sensor_columns]
        data_filled = data_filtered.ffill().bfill()
        
        return self.scaler.transform(data_filled)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fits and transforms the data."""
        self.fit(data)
        return self.transform(data)

    def save(self, filepath: str):
        """Saves the scaler to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)

    def load(self, filepath: str):
        """Loads the scaler from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        self.scaler = joblib.load(filepath)
        return self
