import pandas as pd
import joblib
import numpy as np
import os
import json
from src.ml.transformer import DataPreprocessor


class AnomalyDetector:
    """
    Anomaly detection system for NASA CMAPSS turbofan engines.
    Uses Isolation Forest with operating regime normalization.
    """
    
    def __init__(self, model_dir: str = "models", version: str = "v1", max_history: int = 50):
        """
        Initialize the anomaly detector.
        
        Args:
            model_dir: Base directory for model storage
            version: Model version to load
            max_history: Maximum records to keep in sensor history per engine
        """
        self.model_dir = os.path.join(model_dir, version)
        self.model = None
        self.thresholds = {"p95": 0.5, "p99": 0.6}
        self.preprocessor = DataPreprocessor()
        self.version = version
        
        # Buffer for engine history (used for rolling features if enabled)
        # Key: engine_id
        self.history: dict[int, pd.DataFrame] = {} 
        self.max_history = max_history
        
        self.load_model()

    def load_model(self) -> None:
        """Load the Isolation Forest model, scaler, and thresholds."""
        # Load Preprocessor (Scaler + Regime Logic)
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
             # Try legacy name
            preprocessor_path = os.path.join(self.model_dir, "scaler.pkl")
            
        if os.path.exists(preprocessor_path):
            self.preprocessor.load(preprocessor_path)
            print(f"Loaded preprocessor from {preprocessor_path}")
        else:
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

        # Load Thresholds
        thresholds_path = os.path.join(self.model_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)
            print(f"Loaded thresholds: {self.thresholds}")
        else:
            print(f"Warning: Thresholds file not found at {thresholds_path}. Using defaults.")
            self.thresholds = {"p95": 0.55, "p99": 0.60}

        # Load Isolation Forest
        model_path = os.path.join(self.model_dir, "isolation_forest.pkl")
        if not os.path.exists(model_path):
             # Try legacy name
            model_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Isolation Forest model not found at {model_path}")

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Predict anomalies for turbofan engine data.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        # Ensure engine_id is present
        if 'engine_id' not in data.columns:
            data['engine_id'] = 0 # Default ID if missing
            
        engine_ids = data['engine_id'].unique()
        all_results = []
        
        # Process per engine (to maintain history if we add rolling features later)
        for eid in engine_ids:
            engine_data = data[data['engine_id'] == eid]
            
            # Update history (optional for future rolling features)
            current_hist = self.history.get(eid, pd.DataFrame())
            updated_hist = pd.concat([current_hist, engine_data], ignore_index=True).tail(self.max_history)
            self.history[eid] = updated_hist
            
            # Transform data
            X_scaled = self.preprocessor.transform(engine_data)
            
            # Score
            scores = -self.model.score_samples(X_scaled)
            
            # Threshold Check (using P95 as main trigger based on optimization)
            threshold = self.thresholds.get("p95", 0.55)
            is_anomaly = scores > threshold
            
            for i, idx in enumerate(engine_data.index):
                all_results.append({
                    "index": idx,
                    "score": scores[i],
                    "is_anomaly": bool(is_anomaly[i]),
                    "type": "Degraded" if is_anomaly[i] else "Normal"
                })
        
        # Sort back to original order
        all_results.sort(key=lambda x: x["index"])
        
        return {
            "is_anomaly": [r["is_anomaly"] for r in all_results],
            "anomaly_types": [r["type"] for r in all_results],
            "scores": [round(float(r["score"]), 6) for r in all_results],
            "thresholds": self.thresholds,
            "version": self.version
        }
    
    def clear_history(self, engine_id: int | None = None) -> None:
        """Clear history buffer."""
        if engine_id is not None:
            self.history.pop(engine_id, None)
        else:
            self.history.clear()
