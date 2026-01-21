import numpy as np
import pandas as pd
import requests
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Configuration
API_URL = "http://127.0.0.1:8001/predict"
API_KEY = "dev-api-key"  # Matches .env

def generate_scenario(name, duration_sec=60, anomaly_type=None, magnitude=2.0):
    """
    Generates a sequence of vehicle data with specific anomaly characteristics.
    """
    time_steps = np.arange(0, duration_sec, 0.1)  # 10Hz
    n = len(time_steps)
    
    # Base signals (Normal)
    # Using mostly constant or plausible baselines + noise
    driving_speed = 60 + 5 * np.sin(time_steps * 0.1) + np.random.normal(0, 0.5, n)
    motor_rpm = 2500 + 100 * np.sin(time_steps * 0.1) + np.random.normal(0, 20, n)
    motor_temp = 55 + 2 * np.sin(time_steps * 0.01) + np.random.normal(0, 0.1, n)
    
    # Other fields (Normal noise matching training stats)
    battery_voltage = np.random.normal(350, 20, n)
    battery_current = np.random.normal(-40, 10, n)
    battery_temp = np.random.normal(30, 5, n)
    motor_vibration = np.random.normal(0.6, 0.2, n)
    motor_torque = np.random.normal(80, 10, n)
    power_consumption = np.random.normal(15, 2, n)
    brake_pressure = np.random.normal(0, 0.1, n)
    tire_pressure = np.random.normal(32, 0.5, n)
    tire_temp = np.random.normal(40, 1, n)
    suspension_load = np.random.normal(500, 10, n)
    ambient_temp = np.random.normal(15, 5, n)
    ambient_humidity = np.random.normal(50, 5, n)

    is_anomaly = np.zeros(n, dtype=int)
    
    # Inject Anomalies
    if anomaly_type == "spike": # Sudden Spike (RPM + Vibration)
        idx = n // 2
        motor_rpm[idx:idx+5] += 6000 * magnitude
        motor_vibration[idx:idx+5] += 5.0 * magnitude
        is_anomaly[idx:idx+5] = 1
        
    elif anomaly_type == "drift": # Gradual Drift (Motor Temp + Power)
        drift_start = n // 2
        drift = np.linspace(0, 70 * magnitude, n - drift_start)
        motor_temp[drift_start:] += drift
        power_consumption[drift_start:] += (drift * 0.2)
        is_anomaly[drift_start + len(drift)//2 :] = 1
        
    elif anomaly_type == "failure": # Severe Failure (RPM Mismatch + Speed 0 + High Current)
        idx = n // 2
        driving_speed[idx:idx+20] = 0
        motor_rpm[idx:idx+20] = 9000
        battery_current[idx:idx+20] = 400
        motor_vibration[idx:idx+20] = 8.0
        is_anomaly[idx:idx+20] = 1
        
    data = []
    for i in range(n):
        record = {
            "Battery_Voltage": float(battery_voltage[i]),
            "Battery_Current": float(battery_current[i]),
            "Battery_Temperature": float(battery_temp[i]),
            "Motor_Temperature": float(motor_temp[i]),
            "Motor_Vibration": float(motor_vibration[i]),
            "Motor_Torque": float(motor_torque[i]),
            "Motor_RPM": float(motor_rpm[i]),
            "Power_Consumption": float(power_consumption[i]),
            "Brake_Pressure": float(brake_pressure[i]),
            "Tire_Pressure": float(tire_pressure[i]),
            "Tire_Temperature": float(tire_temp[i]),
            "Suspension_Load": float(suspension_load[i]),
            "Ambient_Temperature": float(ambient_temp[i]),
            "Ambient_Humidity": float(ambient_humidity[i]),
            "Driving_Speed": float(driving_speed[i]),
            "Vehicle_ID": "test_vehicle"
        }
        data.append(record)
        
    return data, is_anomaly

def test_scenario(name, anomaly_type=None):
    print(f"\nTesting Scenario: {name}...")
    data, labels = generate_scenario(name, anomaly_type=anomaly_type)
    
    # Split into batches of 100 to simulate API usage
    predictions = []
    batch_size = 100
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        try:
            response = requests.post(
                API_URL, 
                headers={"X-API-Key": API_KEY},
                json=batch
            )
            response.raise_for_status()
            results = response.json()
            # API returns PredictionResponse with 'is_anomaly' list
            batch_preds = results['is_anomaly']
            predictions.extend(batch_preds)
        except Exception as e:
            print(f"Prediction failed: {e}")
            # Fill with False or skip
            predictions.extend([False] * len(batch))
            
    # Convert bool to int
    y_pred = [1 if p else 0 for p in predictions]
    y_true = labels[:len(y_pred)] # Align lengths
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall: {rec*100:.1f}%")
    
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "count": len(y_true)}

def run_verification():
    scenarios = [
        ("Normal Driving", None),
        ("Engine RPM Spike", "spike"),
        ("Overheating Drift", "drift"),
        ("Sensor Mismatch", "failure")
    ]
    
    total_results = []
    
    print("Starting Comprehensive Scenario Verification...")
    print(f"Target API: {API_URL}")
    
    for name, type_ in scenarios:
        res = test_scenario(name, type_)
        total_results.append(res)
        
    print("\n" + "="*40)
    print("FINAL SUMMARY REPORT")
    print("="*40)
    print(f"{'Scenario':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 58)
    
    avg_acc = 0
    for res in total_results:
        print(f"{res['name']:<20} | {res['accuracy']*100:5.1f}%     | {res['precision']*100:5.1f}%     | {res['recall']*100:5.1f}%")
        avg_acc += res['accuracy']
        
    print("-" * 58)
    print(f"{'OVERALL AVERAGE':<20} | {avg_acc/len(total_results)*100:5.1f}%")
    print("="*40)

if __name__ == "__main__":
    try:
        run_verification()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is it running?")
