import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

class DataPreprocessor:
    def __init__(self, sensor_columns=None):
        if sensor_columns is None:
            # Re-integrated Temp and Power for Drift sensitivity
            self.sensor_columns = [
                "Battery_Current", "Motor_Vibration", "Motor_RPM", 
                "Driving_Speed", "Motor_Temperature", "Power_Consumption",
                "Motor_Torque"
            ]
        else:
            self.sensor_columns = sensor_columns
        
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame, window=10):
        """Fits the scaler on CLEANED raw sensor data."""
        cleaned = self.clean_sensor_data(data)
        self.scaler.fit(cleaned[self.sensor_columns])
        
        # Determine final feature columns
        dummy_transformed = self.transform(data.head(window + 5), window=window)
        return self

    def create_rolling_features(self, data: pd.DataFrame, window=10):
        """Creates physical ratio features for anomaly detection."""
        data_rolled = data.copy()
        
        # 1. Speed/RPM Physical Manifold
        if "Driving_Speed" in data.columns and "Motor_RPM" in data.columns:
            # Normal is ~40. Failure is 8500.
            data_rolled["RPM_Speed_Ratio"] = data["Motor_RPM"] / (data["Driving_Speed"] + 1.0)
            
            # Manifold Error: Deviation from expected RPM
            expected_rpm = data["Driving_Speed"] * 40.0
            data_rolled["Manifold_Error"] = abs(data["Motor_RPM"] - expected_rpm)
            
        # 2. Power Efficiency Manifold (RPM * Torque / 9550 = Power)
        if all(col in data.columns for col in ["Motor_RPM", "Motor_Torque", "Power_Consumption"]):
            # Note: Features are SCALED here, so we use raw column names if we want raw ratios,
            # but usually better to use scaled values or calculate from raw before scaling.
            # However, for "Pure ML", the model identifies deviations in the SCALED manifold too.
            # To catch "Drift", we need a tight relationship.
            expected_power = (data["Motor_RPM"] * data["Motor_Torque"]) / 9550.0
            data_rolled["Power_Manifold_Error"] = abs(data["Power_Consumption"] - expected_power)
            
        return data_rolled.ffill().bfill().fillna(0)

    def transform(self, data: pd.DataFrame, window=10, return_df=False) -> np.ndarray:
        """Transforms raw data into feature vectors."""
        # 1. Clean (Clips negative values)
        cleaned = self.clean_sensor_data(data)
        
        # 2. Scale Raw Sensors
        scaled_raw_values = self.scaler.transform(cleaned[self.sensor_columns])
        scaled_raw_df = pd.DataFrame(scaled_raw_values, columns=self.sensor_columns, index=cleaned.index)
        
        # 3. Create Physical Ratios & Manifold Errors
        featured = self.create_rolling_features(scaled_raw_df, window=window)
        
        # 4. Handle feature column selection
        if not hasattr(self, 'feature_columns'):
            self.feature_columns = featured.columns.tolist()
            
        if return_df:
            return featured[self.feature_columns]
            
        return featured[self.feature_columns].values

    def fit_transform(self, data: pd.DataFrame, window=10) -> np.ndarray:
        """Fits and transforms the data."""
        self.fit(data, window=window)
        return self.transform(data, window=window)

    def clean_sensor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enforces physical sanity bounds on raw sensor data."""
        cleaned = data.copy()
        bounds = {
            "Battery_Voltage": (100, 500),
            "Battery_Current": (-500, 500),
            "Battery_Temperature": (-40, 100),
            "Motor_Temperature": (-40, 150),
            "Motor_Vibration": (0, 10),
            "Motor_RPM": (0, 10000),      # Force non-negative
            "Driving_Speed": (0, 250),    # Force non-negative
            "Tire_Pressure": (10, 60),
            "Brake_Pressure": (0, 5000),
            "Motor_Torque": (-1000, 1000),
            "Power_Consumption": (0, 1000)
        }
        
        for col, (min_val, max_val) in bounds.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].clip(lower=min_val, upper=max_val)
        
        return cleaned.ffill().bfill()

    def save(self, filepath: str):
        """Saves the scaler and feature metadata."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_to_save = {
            "scaler": self.scaler,
            "feature_columns": getattr(self, "feature_columns", self.sensor_columns)
        }
        joblib.dump(data_to_save, filepath)

    def load(self, filepath: str):
        """Loads the scaler and feature metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        loaded_data = joblib.load(filepath)
        if isinstance(loaded_data, dict) and "scaler" in loaded_data:
            self.scaler = loaded_data["scaler"]
            self.feature_columns = loaded_data["feature_columns"]
        else:
            self.scaler = loaded_data
            self.feature_columns = self.sensor_columns
        return self
