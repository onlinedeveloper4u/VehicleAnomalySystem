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

    def fit(self, data: pd.DataFrame, window=10):
        """Fits the scaler on CLEANED raw sensor data."""
        # 1. Clean
        cleaned = self.clean_sensor_data(data)
        
        # 2. Fit scaler only on raw sensors
        self.scaler.fit(cleaned[self.sensor_columns])
        
        # 3. Determine final feature columns after a dummy transform
        # This is needed to know the names of the 90+ features (raw + derived)
        dummy_transformed = self.transform(data.head(window + 5), window=window)
        # transform returns a numpy array, but we want the names. 
        # Actually, let's make transform return the DF for a moment or handle names separately.
        return self

    def transform(self, data: pd.DataFrame, window=10, return_df=False) -> np.ndarray:
        """Transforms data: Clean -> Scale Raw -> Window -> Extract Features."""
        # 1. Clean
        cleaned = self.clean_sensor_data(data)
        
        # 2. Scale Raw Sensors
        # We need to maintain indices for windowing
        scaled_raw_values = self.scaler.transform(cleaned[self.sensor_columns])
        scaled_raw_df = pd.DataFrame(scaled_raw_values, columns=self.sensor_columns, index=cleaned.index)
        
        # 3. Create Rolling Features on SCALED data (Window -> Extract)
        featured = self.create_rolling_features(scaled_raw_df, window=window)
        
        # 4. Handle feature column selection (ordering)
        if not hasattr(self, 'feature_columns'):
            self.feature_columns = featured.columns.tolist()
            
        if return_df:
            return featured[self.feature_columns]
            
        return featured[self.feature_columns].values

    def fit_transform(self, data: pd.DataFrame, window=10) -> np.ndarray:
        """Fits and transforms the data with temporal context."""
        self.fit(data, window=window)
        return self.transform(data, window=window)

    def clean_sensor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans sensor data by enforcing sanity bounds and handling noise.
        Part 2 Compliance: "Rule 1 & 2: Clean before training / Filter obvious sensor errors"
        """
        cleaned = data.copy()
        bounds = {
            "Battery_Voltage": (100, 500),
            "Battery_Current": (-500, 500),
            "Battery_Temperature": (-40, 100),
            "Motor_Temperature": (-40, 150),
            "Motor_Vibration": (0, 10),
            "Motor_RPM": (0, 10000),
            "Driving_Speed": (0, 250),
            "Tire_Pressure": (10, 60),
            "Brake_Pressure": (0, 5000),
        }
        
        for col, (min_val, max_val) in bounds.items():
            if col in cleaned.columns:
                # Clip values to bounds instead of dropping to maintain time-series continuity
                cleaned[col] = cleaned[col].clip(lower=min_val, upper=max_val)
        
        # Handle sharp spikes (Rule 1: Sensor Noise)
        # Using a simple median filter or just relying on subsequent rolling features
        return cleaned.ffill().bfill()

    def create_rolling_features(self, data: pd.DataFrame, window=10):
        """
        Creates advanced statistical and trend features for time-window analysis.
        Part 1 Compliance: "Convert time window -> ML features (mean, std, min, max, slope, delta)"
        """
        data_rolled = data.copy()
        
        # We perform rolling on a per-vehicle/per-session basis if session ID exists,
        # but for now, we assume a continuous stream as provided in the dataset.
        for col in self.sensor_columns:
            if col in data.columns:
                rolling = data[col].rolling(window=window)
                data_rolled[f"{col}_mean"] = rolling.mean()
                data_rolled[f"{col}_std"] = rolling.std()
                data_rolled[f"{col}_min"] = rolling.min()
                data_rolled[f"{col}_max"] = rolling.max()
                
                # Delta (Trend)
                data_rolled[f"{col}_delta"] = data[col].diff(periods=window-1)
                
                # Slope approximation (Simple linear slope: current - start) / window
                data_rolled[f"{col}_slope"] = (data[col] - data[col].shift(window-1)) / window
        
        # Keep only the new features + original raw columns? 
        # Requirement 5.2 mentions specific metrics, and Part 1 says "becomes one ML row".
        # We will keep raw + rolled for maximum model sensitivity.
        return data_rolled.ffill().bfill().fillna(0)

    def save(self, filepath: str):
        """Saves the scaler and feature metadata to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_to_save = {
            "scaler": self.scaler,
            "feature_columns": getattr(self, "feature_columns", self.sensor_columns)
        }
        joblib.dump(data_to_save, filepath)

    def load(self, filepath: str):
        """Loads the scaler and feature metadata from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        loaded_data = joblib.load(filepath)
        if isinstance(loaded_data, dict) and "scaler" in loaded_data:
            self.scaler = loaded_data["scaler"]
            self.feature_columns = loaded_data["feature_columns"]
        else:
            # Fallback for old scaler files
            self.scaler = loaded_data
            self.feature_columns = self.sensor_columns
        return self
