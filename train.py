"""
Main Training Script for NASA CMAPSS Anomaly Detection

Trains a unified Isolation Forest model on all provided CMAPSS datasets (FD001-FD004).
Uses Operating Regime Normalization to handle varying conditions across datasets.
"""

import pandas as pd
import numpy as np
import os
from src.ml.trainer import ModelTrainer
from src.ml.transformer import DataPreprocessor

DATA_DIR = "data/CMAPSSData"
MODEL_DIR = "models"
VERSION = "v1"

# CMAPSS Column Headers
COLS = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
       [f's{i}' for i in range(1, 22)]

def load_all_data():
    """Load and concatenate all available training datasets."""
    dfs = []
    print("Loading datasets...")
    for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
        path = os.path.join(DATA_DIR, f'train_{fd}.txt')
        if os.path.exists(path):
            df = pd.read_csv(path, sep=r'\s+', header=None, names=COLS)
            # Add dataset identifier in case we want to debug later (not used for training features)
            df['dataset'] = fd 
            dfs.append(df)
            print(f"  ✓ {fd}: {len(df)} cycles, {df['engine_id'].nunique()} units")
    
    if not dfs:
        raise FileNotFoundError(f"No training data found in {DATA_DIR}")
        
    return pd.concat(dfs, ignore_index=True)

def extract_normal_baseline(df, ratio=0.25):
    """
    Extract the first `ratio` % of cycles from each engine to form the "Normal" baseline.
    Assumption: Engines start healthy and degrade over time.
    """
    print(f"Extracting normal baseline (first {ratio*100}% of life)...")
    normals = []
    for _, group in df.groupby(['dataset', 'engine_id']):
        max_cycle = group['cycle'].max()
        cutoff = int(max_cycle * ratio)
        normals.append(group[group['cycle'] <= cutoff])
    
    result = pd.concat(normals, ignore_index=True)
    print(f"  ✓ Training on {len(result)} normal samples (Total: {len(df)})")
    return result

def main():
    # 1. Load Data
    raw_df = load_all_data()
    
    # 2. Extract Normal Data
    normal_df = extract_normal_baseline(raw_df, ratio=0.25)
    
    # 3. Preprocess (Fit Scaler & Regime Normalizer)
    print("Preprocessing data with Regime Normalization...")
    # Enable regime normalization to handle mixed conditions from FD002/FD004
    preprocessor = DataPreprocessor(use_regime_normalization=True, n_regimes=6)
    X_train = preprocessor.fit_transform(normal_df)
    
    # 4. Train Model
    print("Training Isolation Forest...")
    trainer = ModelTrainer(model_dir=MODEL_DIR, version=VERSION)
    
    # Save the fitted preprocessor first
    preprocessor.save(os.path.join(trainer.model_dir, "preprocessor.pkl"))
    
    # Train
    thresholds = trainer.train(X_train)
    
    print("\n" + "="*30)
    print("TRAINING COMPLETE")
    print("="*30)
    print(f"Model Version: {VERSION}")
    print(f"Thresholds: {thresholds}")
    print(f"Saved to: {trainer.model_dir}")

if __name__ == "__main__":
    main()
