import pandas as pd
from src.models.predictor import AnomalyDetector
import os

def demo_temporal():
    # 1. Initialize detector
    print("Initializing Anomaly Detector...")
    detector = AnomalyDetector(model_dir="models", version="v1")
    
    # --- REALISTIC FLIGHT SIMULATION ---
    print("\n--- TEST: REALISTIC DEGRADATION FLIGHT ---")
    
    # Load baseline healthy row
    base_df = pd.read_csv("data/normal_data.csv").head(1)
    row = base_df.to_dict('records')[0]
    row["Vehicle_ID"] = "simulation_v1"
    
    # 1. WARM UP (10 Steps of Healthy Operation)
    print("Simulating 10s of Healthy Operation (Warm-up)...")
    for _ in range(10):
        detector.predict(pd.DataFrame([row]))
    
    # 2. STEADY STATE (5 Steps, should be NORMAL)
    print("\nPhase: Steady State (Health 100%)")
    for i in range(5):
        res = detector.predict(pd.DataFrame([row]))
        print(f"Step {i+1}: Anomaly={res['is_anomaly'][0]}, Votes={res['votes'][0]}")
        
    # 3. SUDDEN OVERHEAT (The Anomaly)
    print("\nPhase: Sudden Component Overheat (+40°C)")
    fault_row = row.copy()
    fault_row["Battery_Temperature"] += 40
    res = detector.predict(pd.DataFrame([fault_row]))
    print(f"ALARM STEP: Anomaly={res['is_anomaly'][0]}, Votes={res['votes'][0]}")

if __name__ == "__main__":
    demo_temporal()
