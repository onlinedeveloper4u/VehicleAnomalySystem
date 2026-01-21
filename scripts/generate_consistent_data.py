import numpy as np
import pandas as pd
import random
import os

def generate_consistent_data(output_path, n_samples=300000):
    print(f"Generating {n_samples} high-precision consistent samples...")
    rows = []
    
    # Physical Constants
    gear_ratio = 40.0
    idle_rpm = 800.0
    
    for i in range(n_samples):
        # 1. Driving Speed (Normal distribution with some idle states)
        if random.random() < 0.15: # 15% Idle
            speed = 0.0
        else:
            speed = max(0.1, np.random.normal(60.0, 30.0))
            
        # 2. Correlated RPM (Reduced noise for high precision)
        if speed == 0.0:
            rpm = idle_rpm + np.random.normal(0, 2)
            torque = np.random.normal(0, 0.1)
        else:
            rpm = (speed * gear_ratio) + np.random.normal(0, 5)
            torque = 150.0 + 50.0 * np.random.normal(0, 1) 
            
        # 3. Consumptions & Power
        power = max(0, (rpm * torque) / 9550.0 + np.random.normal(0, 0.2))
        current = -(power * 10.0) + np.random.normal(0, 1)
        voltage = 350.0 + (current / 10.0) + np.random.normal(0, 0.5)
        battery_temp = 30.0 + (abs(current) / 50.0) + np.random.normal(0, 0.2)
        motor_temp = 50.0 + (rpm / 100.0) + np.random.normal(0, 0.5)
        
        # 4. Mechanical & Environmental
        vibration = 0.1 + (rpm / 5000.0) + np.random.normal(0, 0.01)
        brake_pressure = np.random.normal(46.0, 0.5)
        tire_pressure = 32.0 + (motor_temp / 20.0) + np.random.normal(0, 0.1)
        tire_temp = 30.0 + (speed / 5.0) + np.random.normal(0, 0.5)
        suspension_load = 200.0 + np.random.normal(0, 2)
        ambient_temp = 14.1 + np.random.normal(0, 0.1)
        ambient_humidity = 47.1 + np.random.normal(0, 0.1)
        
        rows.append({
            "Battery_Voltage": voltage,
            "Battery_Current": current,
            "Battery_Temperature": battery_temp,
            "Motor_Temperature": motor_temp,
            "Motor_Vibration": vibration,
            "Motor_Torque": torque,
            "Motor_RPM": rpm,
            "Power_Consumption": power,
            "Brake_Pressure": brake_pressure,
            "Tire_Pressure": tire_pressure,
            "Tire_Temperature": tire_temp,
            "Suspension_Load": suspension_load,
            "Ambient_Temperature": ambient_temp,
            "Ambient_Humidity": ambient_humidity,
            "Driving_Speed": speed,
            "Vehicle_ID": "SYNTH-NORMAL-HP"
        })
        
        if i % 100000 == 0 and i > 0:
            print(f"Generated {i} samples...")
            
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved high-precision consistent data to {output_path}")

if __name__ == "__main__":
    generate_consistent_data("data/normal_data_consistent.csv")
