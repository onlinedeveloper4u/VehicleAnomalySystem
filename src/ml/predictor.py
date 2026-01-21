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
            max_history: Maximum records to keep in sensor history per vehicle
        """
        self.model_dir = os.path.join(model_dir, version)
        self.model = None
        self.thresholds = {"hard": 0.0, "soft": 0.0}
        self.preprocessor = DataPreprocessor()
        self.version = version
        
        # Key: Vehicle_ID
        self.history: dict[str, pd.DataFrame] = {} 
        self.score_history: dict[str, list[float]] = {} # Track last 5 anomaly scores
        self.max_history = max_history
        self.max_history = max_history
        self.score_window = 5 # Rolling average window for drift detection
        
        self.load_model()

    def load_model(self) -> None:
        """Load the Isolation Forest model, scaler, and dual thresholds."""
        # Load Scaler
        scaler_path = os.path.join(self.model_dir, "preprocessor.joblib")
        if not os.path.exists(scaler_path):
            # Try old name
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            
        if os.path.exists(scaler_path):
            self.preprocessor.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        # Load Thresholds
        thresholds_path = os.path.join(self.model_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)
        else:
            print(f"Warning: Thresholds file not found at {thresholds_path}. Using zero thresholds.")
            self.thresholds = {"hard": 0.0, "soft": 0.0}

        # Load Isolation Forest
        model_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Isolation Forest model not found at {model_path}")

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Predict anomalies using pure ML scores (Isolation Forest).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        vehicle_ids = data['Vehicle_ID'].unique() if 'Vehicle_ID' in data.columns else ['default']
        all_samples_meta = []
        
        for vid in vehicle_ids:
            v_data = data[data['Vehicle_ID'] == vid] if 'Vehicle_ID' in data.columns else data
            v_history = self.history.get(vid, pd.DataFrame())
            
            # Combine history with new data
            combined_v = pd.concat([v_history, v_data], ignore_index=True)
            self.history[vid] = combined_v.tail(self.max_history).copy()
            
            # Transform and get NEW records
            X_v_combined_scaled = self.preprocessor.transform(combined_v)
            X_v_new_scaled = X_v_combined_scaled[-len(v_data):]
            
            for i, idx in enumerate(v_data.index):
                all_samples_meta.append((idx, X_v_new_scaled[i], vid))
        
        # Sort to match request order
        all_samples_meta.sort(key=lambda x: x[0])
        X_scaled = np.array([x[1] for x in all_samples_meta])

        # Inference
        raw_scores = -self.model.score_samples(X_scaled)

        # Threshold Logic
        is_anomaly = []
        anomaly_types = []
        
        for i, (idx, _, vid) in enumerate(all_samples_meta):
            score = raw_scores[i]
            
            # 1. Pure ML Classification
            # No conditional logic, history suppressions, or ground-truth overrides.
            detected = bool(score > self.thresholds["hard"])
            atype = "Anomaly" if detected else "Normal"
            
            is_anomaly.append(detected)
            anomaly_types.append(atype)


        return {
            "is_anomaly": is_anomaly,
            "anomaly_types": anomaly_types,
            "scores": [round(float(s), 6) for s in raw_scores],
            "thresholds": self.thresholds,
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
