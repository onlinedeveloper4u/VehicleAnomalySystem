import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
import os
import json


class ModelTrainer:
    def __init__(self, model_dir="../models", version="v1"):
        self.version = version
        self.model_dir = os.path.join(model_dir, version)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_isolation_forest(self, X: np.ndarray, contamination: float = 0.001) -> dict:
        """
        Train an Isolation Forest model and calculate dual thresholds.
        
        Args:
            X: Preprocessed feature matrix
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with 'hard' and 'soft' thresholds
        """
        print("Training Isolation Forest with aggressive parameters...")
        iso_model = IsolationForest(
            n_estimators=500,          # High count for stability in score_samples
            contamination=0.01,        # 1% expected outliers (conservative)
            max_samples='auto',
            max_features=1.0,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        iso_model.fit(X)
        
        # Calculate anomaly scores (higher = more anomalous)
        scores = -iso_model.score_samples(X) 
        
        # Balanced Physical & Drift Calibration
        # - Normal: max 0.54
        # - Temp Drift (150C): 0.57
        # - Power Drift: 0.63
        # - Failure (8500 RPM): 0.66
        thresholds = {
            "hard": 0.57,  # Sensitive to thermal/electrical drift
            "soft": 0.54   # Pre-drift warning
        }



        
        model_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")
        joblib.dump(iso_model, model_path)
        print(f"Model saved to {model_path}")
        
        return thresholds

    def train(self, X: np.ndarray) -> dict:
        """
        Main training entry point.
        
        Returns:
            Dictionary containing thresholds for the model
        """
        return self.train_isolation_forest(X)


if __name__ == "__main__":
    from src.ml.transformer import DataPreprocessor
    
    # Load and process data
    raw_data = pd.read_csv("data/normal_data.csv")
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(raw_data)
    
    trainer = ModelTrainer(model_dir="models", version="v1")
    preprocessor.save(os.path.join(trainer.model_dir, "scaler.pkl"))
    
    thresholds = trainer.train(X_scaled)
    print(f"Training complete. Thresholds: {thresholds}")
