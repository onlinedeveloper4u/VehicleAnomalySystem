import pandas as pd
from src.models.trainer import ModelTrainer
from src.preprocessing.transformer import DataPreprocessor
import os

def main():
    # Load and process data
    data_path = "data/ev_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print("Loading data...")
    raw_data = pd.read_csv(data_path) 
    
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(raw_data)
    preprocessor.save("models/scaler.pkl")
    
    print("Starting model training...")
    trainer = ModelTrainer(model_dir="models")
    trainer.train_all(X_scaled)
    print("Training complete.")

if __name__ == "__main__":
    main()
