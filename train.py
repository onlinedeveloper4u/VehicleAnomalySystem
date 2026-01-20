import pandas as pd
from src.models.trainer import ModelTrainer
from src.preprocessing.transformer import DataPreprocessor
import os

def main():
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
    trainer.train_all(X_scaled)
    print("Training complete.")

if __name__ == "__main__":
    main()
