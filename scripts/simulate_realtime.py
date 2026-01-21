import time
import requests
import numpy as np
import json
import argparse
from datetime import datetime

# Configuration
API_URL = "http://127.0.0.1:8000/predict"
API_KEY = "dev-api-key"

class VehicleSimulator:
    def __init__(self, vehicle_id="SIM-101"):
        self.vehicle_id = vehicle_id
        self.step = 0
        self.anomaly_mode = None
        self.anomaly_step = 0
        
    def set_anomaly(self, mode):
        print(f"\n[ALERT] Injecting Anomaly Mode: {mode.upper()}")
        self.anomaly_mode = mode
        self.anomaly_step = 0
        
    def generate_record(self):
        self.step += 1
        
    def generate_record(self):
        self.step += 1
        
        # 1. Primary Driver: Driving_Speed
        if self.anomaly_mode == "failure":
            driving_speed = 0.0
        else:
            # Slower speed oscillations
            driving_speed = max(0.0, 50.0 + 30.0 * np.sin(self.step / 10.0) + np.random.normal(0, 5))
        
        # 2. Physical Constants
        gear_ratio = 40.0
        idle_rpm = 800.0
        
        # 3. Physically Derived Sensors
        if driving_speed > 0.1:
            motor_rpm = (driving_speed * gear_ratio) + np.random.normal(0, 100)
            motor_torque = 150.0 + 50.0 * np.sin(self.step / 5.0) + np.random.normal(0, 5)
        else:
            motor_rpm = idle_rpm + np.random.normal(0, 20)
            motor_torque = np.random.normal(0, 0.5)
            
        motor_temp = 50.0 + (motor_rpm / 100.0) + np.random.normal(0, 2)
        power_consumption = (motor_rpm * motor_torque) / 9550.0 + np.random.normal(0, 1)
        
        # Battery stats correlated with power
        current = -(power_consumption * 10.0) + np.random.normal(0, 5)
        voltage = 350.0 + (current / 10.0) + np.random.normal(0, 2)
        battery_temp = 30.0 + (abs(current) / 50.0) + np.random.normal(0, 1)
        
        # Others
        vibration = 0.1 + (motor_rpm / 5000.0) + np.random.normal(0, 0.05)
        brake_pressure = np.random.normal(46.0, 2.0)
        tire_pressure = 32.0 + (motor_temp / 20.0) + np.random.normal(0, 0.5)
        tire_temp = 30.0 + (driving_speed / 5.0) + np.random.normal(0, 1)
        suspension_load = 200.0 + 50.0 * np.random.normal(0, 1)
        ambient_temp = 14.1 + np.random.normal(0, 1)
        ambient_humidity = 47.1 + np.random.normal(0, 1)
        
        # Apply Anomalies (Override Physicals)
        if self.anomaly_mode == "spike":
            motor_rpm += 6000
            vibration += 8.0
            self.anomaly_step += 1
            if self.anomaly_step > 5: self.anomaly_mode = None
            
        elif self.anomaly_mode == "drift":
            motor_temp += self.anomaly_step * 2.0
            power_consumption += self.anomaly_step * 0.5
            self.anomaly_step += 1
            if self.anomaly_step > 50: self.anomaly_mode = None
            
        elif self.anomaly_mode == "failure":
            # Driving_Speed set to 0 above
            motor_rpm = 8500
            current = 300
            vibration = 5.0
            self.anomaly_step += 1
            if self.anomaly_step > 15: self.anomaly_mode = None

        return {
            "Battery_Voltage": float(voltage),
            "Battery_Current": float(current),
            "Battery_Temperature": float(battery_temp),
            "Motor_Temperature": float(motor_temp),
            "Motor_Vibration": float(vibration),
            "Motor_Torque": float(motor_torque),
            "Motor_RPM": float(motor_rpm),
            "Power_Consumption": float(power_consumption),

            "Brake_Pressure": float(brake_pressure),
            "Tire_Pressure": float(tire_pressure),
            "Tire_Temperature": float(tire_temp),
            "Suspension_Load": float(suspension_load),
            "Ambient_Temperature": float(ambient_temp),
            "Ambient_Humidity": float(ambient_humidity),
            "Driving_Speed": float(driving_speed),
            "Vehicle_ID": self.vehicle_id
        }

def run_simulation(interval, auto_anomaly=False):
    sim = VehicleSimulator()
    print(f"Starting Real-time Simulation for {sim.vehicle_id}")
    print(f"Target API: {API_URL}")
    print("-" * 50)
    
    try:
        while True:
            # Generate and send data
            record = sim.generate_record()
            
            # Anomaly scheduling (optional)
            if auto_anomaly and sim.step % 50 == 0:
                modes = ["spike", "drift", "failure"]
                sim.set_anomaly(np.random.choice(modes))
            
            try:
                start_time = time.time()
                response = requests.post(
                    API_URL,
                    headers={"X-API-Key": API_KEY},
                    json=[record],  # API expects a list
                    timeout=2
                )
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    res_data = response.json()
                    is_anomaly = res_data["is_anomaly"][0]
                    score = res_data["scores"][0]
                    
                    status_icon = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Speed: {record['Driving_Speed']:.1f} | RPM: {record['Motor_RPM']:.0f} | "
                          f"Score: {score:.3f} | {status_icon} ({latency:.1f}ms)")
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"Request failed: {e}")
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Vehicle Telemetry Simulator")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings")
    parser.add_argument("--auto", action="store_true", help="Automatically inject anomalies periodically")
    args = parser.parse_args()
    
    run_simulation(args.interval, args.auto)
