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

    def train_isolation_forest(self, X: np.ndarray, contamination: float = 0.001) -> float:
        """
        Train an Isolation Forest model for anomaly detection.
        
        Args:
            X: Preprocessed feature matrix
            contamination: Expected proportion of outliers (default 0.1%)
            
        Returns:
            Threshold value at 99.5th percentile of anomaly scores
        """
        print("Training Isolation Forest...")
        iso_model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        iso_model.fit(X)
        
        # Calculate anomaly scores (higher = more anomalous)
        scores = -iso_model.score_samples(X) 
        threshold = float(np.percentile(scores, 99.5))
        
        model_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")
        joblib.dump(iso_model, model_path)
        print(f"Model saved to {model_path}")
        
        return threshold

    def train(self, X: np.ndarray) -> dict:
        """
        Main training entry point.
        
        Args:
            X: Preprocessed feature matrix
            
        Returns:
            Dictionary containing threshold for the model
        """
        thresholds = {}
        thresholds["isolation_forest"] = self.train_isolation_forest(X)
        return thresholds


if __name__ == "__main__":
    from src.preprocessing.transformer import DataPreprocessor
    
    # Load and process data
    raw_data = pd.read_csv("data/normal_data.csv")
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(raw_data)
    
    trainer = ModelTrainer(model_dir="models", version="v1")
    preprocessor.save(os.path.join(trainer.model_dir, "scaler.pkl"))
    
    thresholds = trainer.train(X_scaled)
    print(f"Training complete. Thresholds: {thresholds}")
