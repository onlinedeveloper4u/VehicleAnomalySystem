"""
Real-time CMAPSS Data Replay Simulator

This script simulates a real-time stream of sensor data by replaying
trajectories from the NASA CMAPSS test set. It picks a random engine
and streams its lifecycle data to the API.
"""

import time
import requests
import pandas as pd
import numpy as np
import argparse
import sys
import os

# Configuration
API_URL = "http://127.0.0.1:8000/predict"
API_KEY = "dev-api-key"
DATA_FILE = "data/CMAPSSData/test_FD001.txt"

# Column definitions
COLS = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
       [f's{i}' for i in range(1, 22)]

def load_random_trajectory(filepath):
    """Load a random engine's full trajectory from the dataset."""
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        sys.exit(1)
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLS)
    
    # Pick a random engine
    engine_ids = df['engine_id'].unique()
    selected_id = np.random.choice(engine_ids)
    
    trajectory = df[df['engine_id'] == selected_id].sort_values('cycle')
    print(f"Selected Engine ID: {selected_id} ({len(trajectory)} cycles)")
    
    return trajectory

def run_simulation(interval=1.0, loop=False):
    """Stream data to the API."""
    print("-" * 50)
    print(f"Target API: {API_URL}")
    print(f"Update Interval: {interval}s")
    print("-" * 50)
    
    while True:
        trajectory = load_random_trajectory(DATA_FILE)
        
        for _, row in trajectory.iterrows():
            # Convert row to dictionary matches API schema
            payload = row.to_dict()
            
            # API expects a list of records
            request_data = [payload]
            
            try:
                start_time = time.time()
                response = requests.post(
                    API_URL,
                    headers={"X-API-Key": API_KEY},
                    json=request_data,
                    timeout=2
                )
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    res_json = response.json()
                    is_anomaly = res_json['is_anomaly'][0]
                    score = res_json['scores'][0]
                    
                    status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
                    print(f"Cycle {int(payload['cycle']):3d} | "
                          f"Score: {score:.4f} | {status} | Latency: {latency:.0f}ms")
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"Connection Failed: {e}")
                
            time.sleep(interval)
            
        if not loop:
            print("Trajectory complete.")
            break
        print("\nStarting new trajectory...\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAPSS Data Stream Simulator")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between updates")
    parser.add_argument("--loop", action="store_true", help="Loop indefinitely with new engines")
    args = parser.parse_args()
    
    try:
        run_simulation(args.interval, args.loop)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
