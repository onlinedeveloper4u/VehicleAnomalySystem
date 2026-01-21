import pandas as pd
import numpy as np
from src.ml.trainer import ModelTrainer

from src.ml.transformer import DataPreprocessor
import os
import json


def main():
    """Train the Isolation Forest anomaly detection model."""
    # Load physically consistent healthy baseline data
    data_path = "data/normal_data_consistent.csv"
    if not os.path.exists(data_path):
        data_path = "data/normal_data_augmented.csv" # Fallback



    print(f"Loading normal data from {data_path}...")
    healthy_data = pd.read_csv(data_path) 
    print(f"Healthy Baseline size: {len(healthy_data)}")
    
    print("Preprocessing healthy data...")
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(healthy_data)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    nan_count = np.isnan(X_scaled).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaNs. Filling with 0.")
        X_scaled = np.nan_to_num(X_scaled)
    
    print("Starting model training...")

    trainer = ModelTrainer(model_dir="models", version="v1")
    
    # Save scaler into the versioned directory
    preprocessor.save(os.path.join(trainer.model_dir, "scaler.pkl"))
    
    # Train the model and get thresholds
    thresholds = trainer.train(X_scaled)
    
    # Save thresholds
    thresholds_path = os.path.join(trainer.model_dir, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=4)
    
    print(f"Models and thresholds saved to {trainer.model_dir}")
    print(f"Thresholds: {thresholds}")
    print("Training complete.")


if __name__ == "__main__":
    main()
