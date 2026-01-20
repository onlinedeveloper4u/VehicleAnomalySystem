import pandas as pd
import numpy as np
from src.models.predictor import AnomalyDetector
import os

def diagnose():
    detector = AnomalyDetector(model_dir="models", version="v1")
    print(f"Current Thresholds: {detector.thresholds}")

    # 1. Check Healthy Data Scores
    healthy = pd.read_csv("data/normal_data.csv").tail(50)
    healthy["Vehicle_ID"] = "diag_healthy"
    res_h = detector.predict(healthy)
    
    # 2. Check Severe Data Scores
    severe = pd.read_csv("data/severe_data.csv").tail(50)
    severe["Vehicle_ID"] = "diag_severe"
    res_s = detector.predict(severe)

    print("\n--- SCORE AUDIT (First 5 records of each) ---")
    
    print("\nHEALTHY SECTOR:")
    for i in range(5):
        scores = res_h['details'][i]['model_scores']
        flags = res_h['is_anomaly'][i]
        print(f"Row {i}: Anomaly={flags} | Scores: {scores}")

    print("\nSEVERE SECTOR:")
    for i in range(5):
        scores = res_s['details'][i]['model_scores']
        flags = res_s['is_anomaly'][i]
        print(f"Row {i}: Anomaly={flags} | Scores: {scores}")

if __name__ == "__main__":
    diagnose()
