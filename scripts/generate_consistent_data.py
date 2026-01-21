import numpy as np
import pandas as pd
import random
import os

def generate_consistent_data(output_path, n_samples=300000):
    print(f"Generating {n_samples} physically consistent samples...")
    rows = []
    
    # Physical Constants
    gear_ratio = 40.0
    idle_rpm = 800.0
    
    for i in range(n_samples):
        # 1. Driving Speed
        if random.random() < 0.20: # 20% Idle/Stationary
            speed = 0.0
        else:
            speed = max(0.1, np.random.normal(60.0, 30.0))
            
        # 2. Correlated RPM & Torque
        if speed == 0.0:
            rpm = idle_rpm + np.random.normal(0, 15)
            torque = np.random.normal(0, 1.0) # Minimal torque at idle
        else:
            rpm = (speed * gear_ratio) + np.random.normal(0, 50)
            # Torque is higher when accelerating or maintaining speed
            # We'll use a range to simulate different loads
            torque = 100.0 + 100.0 * np.random.random() + np.random.normal(0, 10)
            
        # 3. Consumptions & Power (Physical Law: P = T * RPM / 9550)
        # We enforce this law CRYSTAL CLEAR in the training data
        power_consumption = (rpm * torque) / 9550.0 + np.random.normal(0, 0.5)
        
        # 4. Derived Temperatures
        # Motor temp increases with power and RPM
        motor_temp = 40.0 + (power_consumption * 2.0) + (rpm / 500.0) + np.random.normal(0, 2)
        battery_temp = 30.0 + (power_consumption / 1.5) + np.random.normal(0, 1)
        
        # 5. Electrical
        current = -(power_consumption * 10.0) + np.random.normal(0, 5)
        voltage = 350.0 + (current / 20.0) + np.random.normal(0, 2)
        
        # 6. Mechanical & Environmental
        vibration = 0.1 + (rpm / 4000.0) + (torque / 500.0) + np.random.normal(0, 0.05)
        brake_pressure = np.random.normal(46.0, 1.0)
        tire_pressure = 32.0 + (motor_temp / 10.0) + np.random.normal(0, 0.5)
        tire_temp = 30.0 + (speed / 4.0) + np.random.normal(0, 1)
        suspension_load = 200.0 + np.random.normal(0, 10)
        ambient_temp = 14.1 + np.random.normal(0, 1)
        ambient_humidity = 47.1 + np.random.normal(0, 1)
        
        rows.append({
            "Battery_Voltage": voltage,
            "Battery_Current": current,
            "Battery_Temperature": battery_temp,
            "Motor_Temperature": motor_temp,
            "Motor_Vibration": vibration,
            "Motor_Torque": torque,
            "Motor_RPM": rpm,
            "Power_Consumption": power_consumption,
            "Brake_Pressure": brake_pressure,
            "Tire_Pressure": tire_pressure,
            "Tire_Temperature": tire_temp,
            "Suspension_Load": suspension_load,
            "Ambient_Temperature": ambient_temp,
            "Ambient_Humidity": ambient_humidity,
            "Driving_Speed": speed,
            "Vehicle_ID": "PHYS-CONSISTENT-V3"
        })
        
        if i % 100000 == 0 and i > 0:
            print(f"Generated {i} samples...")
            
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved physically consistent data to {output_path}")

if __name__ == "__main__":
    generate_consistent_data("data/normal_data_consistent.csv")
