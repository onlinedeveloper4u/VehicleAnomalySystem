import pandas as pd
from src.ml.trainer import ModelTrainer
from src.ml.transformer import DataPreprocessor
import os
import json


def main():
    """Train the Isolation Forest anomaly detection model."""
    # Load pre-filtered healthy baseline data
    data_path = "data/normal_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Normal data not found at {data_path}. Please run separate_data.py first.")
        return

    print(f"Loading normal data from {data_path}...")
    healthy_data = pd.read_csv(data_path) 
    print(f"Healthy Baseline size: {len(healthy_data)}")
    
    print("Preprocessing healthy data...")
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(healthy_data)
    
    print("Starting model training...")
    trainer = ModelTrainer(model_dir="models", version="v1")
    
    # Save scaler into the versioned directory
    preprocessor.save(os.path.join(trainer.model_dir, "scaler.pkl"))
    
    # Train the model and get threshold
    thresholds = trainer.train(X_scaled)
    
    # Adjusted to 1.15 to balance sensitivity (avoid misses) vs robustness (avoid false positives)
    SAFETY_MULTIPLIER = 1.15
    stable_thresholds = {k: v * SAFETY_MULTIPLIER for k, v in thresholds.items()}
    
    # Save thresholds
    thresholds_path = os.path.join(trainer.model_dir, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(stable_thresholds, f, indent=4)
    
    print(f"Models and thresholds saved to {trainer.model_dir}")
    print(f"Thresholds: {stable_thresholds}")
    print("Training complete.")


if __name__ == "__main__":
    main()
