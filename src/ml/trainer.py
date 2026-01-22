import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
import os
import json
from typing import Dict, Any


class ModelTrainer:
    """
    Trainer for CMAPSS Anomaly Detection Model.
    Supports separate training for single-condition (FD001/3) and multi-condition (FD002/4) datasets.
    """
    
    def __init__(self, model_dir: str = "models", version: str = "v1"):
        self.version = version
        self.model_dir = os.path.join(model_dir, version)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_isolation_forest(self, X: np.ndarray, contamination: float = 0.01) -> Dict[str, float]:
        """
        Train Isolation Forest and calculate robust thresholds.
        
        Args:
            X: Feature matrix
            contamination: Expected anomaly rate in training data (usually low for normal data)
            
        Returns:
            Dictionary of calculated thresholds
        """
        print(f"Training Isolation Forest on {X.shape[0]} samples...")
        
        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            max_samples=min(512, len(X)),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)
        
        # Calculate scores on training data
        scores = -model.score_samples(X)
        
        # Calculate statistical thresholds
        thresholds = {
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, "isolation_forest.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        return thresholds

    def save_thresholds(self, thresholds: Dict[str, float]):
        """Save thresholds to JSON."""
        path = os.path.join(self.model_dir, "thresholds.json")
        with open(path, "w") as f:
            json.dump(thresholds, f, indent=4)
        print(f"Thresholds saved to {path}")

    def train(self, X: np.ndarray, contamination: float = 0.01) -> Dict[str, float]:
        """Main training workflow."""
        thresholds = self.train_isolation_forest(X, contamination)
        self.save_thresholds(thresholds)
        return thresholds
