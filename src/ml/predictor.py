import pandas as pd
import joblib
import numpy as np
import os
import json
from src.ml.transformer import DataPreprocessor


class AnomalyDetector:
    """
    Anomaly detection system using Isolation Forest.
    
    Supports stateful history buffer for time-window analysis on single records.
    """
    
    def __init__(self, model_dir: str = "models", version: str = "v1", max_history: int = 20):
        """
        Initialize the anomaly detector.
        
        Args:
            model_dir: Base directory for model storage
            version: Model version to load
            max_history: Maximum records to keep in history per vehicle
        """
        self.model_dir = os.path.join(model_dir, version)
        self.model = None
        self.threshold = 0.0
        self.preprocessor = DataPreprocessor()
        self.version = version
        self.history: dict[str, pd.DataFrame] = {}  # Key: Vehicle_ID, Value: pd.DataFrame
        self.max_history = max_history
        
        self.load_model()

    def load_model(self) -> None:
        """Load the Isolation Forest model, scaler, and threshold."""
        # Load Scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.preprocessor.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        # Load Threshold
        thresholds_path = os.path.join(self.model_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                thresholds = json.load(f)
                self.threshold = thresholds.get("isolation_forest", 0.0)
        else:
            print(f"Warning: Thresholds file not found at {thresholds_path}. Using default threshold.")
            self.threshold = 0.0

        # Load Isolation Forest
        model_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Isolation Forest model not found at {model_path}")

    def predict(self, data: pd.DataFrame, threshold_override: float | None = None) -> dict:
        """
        Predict anomalies for the given DataFrame.
        
        Uses a stateful history buffer to support sequence analysis for single records.
        
        Args:
            data: Input DataFrame with sensor readings
            threshold_override: Optional custom threshold to use
            
        Returns:
            Dictionary with anomaly predictions and scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use override threshold if provided
        current_threshold = threshold_override if threshold_override is not None else self.threshold
        
        # 1. Manage history for sequence context
        vehicle_ids = data['Vehicle_ID'].unique() if 'Vehicle_ID' in data.columns else ['default']
        
        # Track samples with their metadata for final aggregation
        # Format: (original_index, scaled_features, vehicle_id, history_length)
        all_samples_meta = []
        
        for vid in vehicle_ids:
            # Filter data for this vehicle
            v_data = data[data['Vehicle_ID'] == vid] if 'Vehicle_ID' in data.columns else data
            v_history = self.history.get(vid, pd.DataFrame())
            
            # Combine history with new data
            combined_v = pd.concat([v_history, v_data], ignore_index=True)
            
            # Update history for this vehicle
            self.history[vid] = combined_v.tail(self.max_history).copy()
            
            # Transform this vehicle's batch (including history context)
            X_v_combined_scaled = self.preprocessor.transform(combined_v)
            
            # Extract only the NEW records from this vehicle
            n_new = len(v_data)
            X_v_new_scaled = X_v_combined_scaled[-n_new:]
            
            # Label them with their original metadata
            indices = v_data.index
            for i, idx in enumerate(indices):
                current_h_len = len(v_history) + i
                all_samples_meta.append((idx, X_v_new_scaled[i], vid, current_h_len))
        
        # Sort back to original request order
        all_samples_meta.sort(key=lambda x: x[0])
        X_scaled = np.array([x[1] for x in all_samples_meta])

        # 2. Isolation Forest Prediction
        # Higher score = more anomalous
        scores = -self.model.score_samples(X_scaled)
        
        # 3. Apply threshold with warm-up grace period
        n_samples = len(X_scaled)
        is_anomaly = []
        
        for i in range(n_samples):
            # Removed grace period as per user request to enable immediate prediction
            # History < 10 will rely on min_periods=1 in preprocessor (delta=0)
            is_anomaly.append(bool(scores[i] > current_threshold))

        # 4. Build response
        return {
            "is_anomaly": is_anomaly,
            "scores": [round(float(s), 6) for s in scores],
            "threshold": current_threshold,
            "version": self.version
        }
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update the anomaly detection threshold."""
        self.threshold = new_threshold
    
    def clear_history(self, vehicle_id: str | None = None) -> None:
        """
        Clear history buffer for a specific vehicle or all vehicles.
        
        Args:
            vehicle_id: If provided, clear only that vehicle's history.
                       If None, clear all history.
        """
        if vehicle_id:
            self.history.pop(vehicle_id, None)
        else:
            self.history.clear()
