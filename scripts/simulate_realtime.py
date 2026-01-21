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
        
        # Base realistic distributions (from training data audit)
        # Sinusoidal components for long-term trends
        cycle = np.sin(self.step * 0.1)
        
        speed = 60 + 15 * cycle + np.random.normal(0, 1.0)
        rpm = 2500 + 500 * cycle + np.random.normal(0, 50)
        temp = 55 + 5 * np.sin(self.step * 0.01) + np.random.normal(0, 0.2)
        
        # Other fields
        voltage = np.random.normal(370, 10)
        current = np.random.normal(-35, 5)
        vibration = np.random.normal(0.6, 0.1)
        
        # Apply Anomalies
        if self.anomaly_mode == "spike":
            rpm += 6000
            vibration += 8.0
            self.anomaly_step += 1
            if self.anomaly_step > 5: self.anomaly_mode = None
            
        elif self.anomaly_mode == "drift":
            temp += self.anomaly_step * 2.0
            voltage -= self.anomaly_step * 1.0
            self.anomaly_step += 1
            if self.anomaly_step > 50: self.anomaly_mode = None
            
        elif self.anomaly_mode == "failure":
            speed = 0
            rpm = 8500
            current = 300
            self.anomaly_step += 1
            if self.anomaly_step > 15: self.anomaly_mode = None

        return {
            "Battery_Voltage": float(voltage),
            "Battery_Current": float(current),
            "Battery_Temperature": float(35 + 2 * cycle),
            "Motor_Temperature": float(temp),
            "Motor_Vibration": float(vibration),
            "Motor_Torque": float(100 + 10 * cycle),
            "Motor_RPM": float(rpm),
            "Power_Consumption": float(15 + 5 * cycle),
            "Brake_Pressure": float(max(0, np.random.normal(0, 0.1))),
            "Tire_Pressure": float(32 + np.random.normal(0, 0.2)),
            "Tire_Temperature": float(40 + temp * 0.1),
            "Suspension_Load": float(500 + 20 * cycle),
            "Ambient_Temperature": 22.0,
            "Ambient_Humidity": 45.0,
            "Driving_Speed": float(speed),
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
