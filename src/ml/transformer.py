import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

class DataPreprocessor:
    def __init__(self, sensor_columns=None):
        # Raw sensors needed for physical residuals
        if sensor_columns is None:
            self.sensor_columns = [
                "Battery_Voltage", "Battery_Current", "Battery_Temperature",
                "Motor_Temperature", "Motor_Vibration", "Motor_RPM", 
                "Power_Consumption", "Driving_Speed", "Motor_Torque"
            ]
        else:
            self.sensor_columns = sensor_columns
        
        # High-signal feature set: Focus on EXPONENTIAL physical residuals
        # This makes even minor drifts (e.g. 5C) look like massive outliers.
        self.feature_columns = [
            "Vibration_Res_Sq", "Current_Res_Sq", 
            "Manifold_Res_Sq", "Thermal_Res_Sq", "Efficiency_Res_Sq"
        ]
        
        self.scaler = StandardScaler()

    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates squared physical RESIDUAL features from RAW sensor data."""
        df = data.copy()
        
        # 1. Squared Residuals: Amplifies anomaly signal relative to noise.
        # Motor Vibration
        v_res = df["Motor_Vibration"] - (0.1 + df["Motor_RPM"] / 5000.0)
        df["Vibration_Res_Sq"] = v_res**2

        # Battery Current 
        c_res = df["Battery_Current"] - (-(df["Power_Consumption"] * 10.0))
        df["Current_Res_Sq"] = c_res**2

        # Manifold Consistency
        is_idle = df["Driving_Speed"] < 1.0
        expected_rpm = np.where(is_idle, 800.0, df["Driving_Speed"] * 40.0)
        m_res = df["Motor_RPM"] - expected_rpm
        df["Manifold_Res_Sq"] = m_res**2
        
        # Thermal Efficiency
        t_res = df["Motor_Temperature"] - (50.0 + df["Motor_RPM"] / 100.0)
        df["Thermal_Res_Sq"] = t_res**2
        
        # Mechanical Efficiency
        e_res = df["Power_Consumption"] - ((df["Motor_RPM"] * df["Motor_Torque"]) / 9550.0)
        df["Efficiency_Res_Sq"] = e_res**2
        
        return df.fillna(0)

    def fit(self, data: pd.DataFrame):
        """Fits the scaler on residuals of clean training data."""
        cleaned = self.clean_sensor_data(data)
        derived = self.create_derived_features(cleaned)
        self.scaler.fit(derived[self.feature_columns])
        return self

    def transform(self, data: pd.DataFrame, return_df=False) -> np.ndarray:
        # 1. Clean
        cleaned = self.clean_sensor_data(data)
        
        # 2. Derive Squared Residuals
        derived = self.create_derived_features(cleaned)
        
        # 3. Scale
        # StandardScaler is still used to keep features on similar scale (~1.0 for noise)
        scaled_values = self.scaler.transform(derived[self.feature_columns])
        
        if return_df:
            return pd.DataFrame(scaled_values, columns=self.feature_columns, index=data.index)
            
        return scaled_values

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def clean_sensor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enforces physical sanity bounds on raw sensor data."""
        cleaned = data.copy()
        bounds = {
            "Battery_Voltage": (100, 500),
            "Battery_Current": (-1000, 500),
            "Battery_Temperature": (-40, 100),
            "Motor_Temperature": (-40, 150),
            "Motor_Vibration": (0, 10),
            "Motor_RPM": (0, 10000),
            "Driving_Speed": (0, 250),
        }
        for col, (min_val, max_val) in bounds.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].clip(lower=min_val, upper=max_val)
        return cleaned.ffill().bfill()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_columns": self.feature_columns}, filepath)

    def load(self, filepath: str):
        loaded = joblib.load(filepath)
        self.scaler = loaded["scaler"]
        self.feature_columns = loaded["feature_columns"]
        return self
