# preprocess_data.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load raw dataset
raw_data = pd.read_csv("../data/ev_dataset.csv")  # replace with your dataset

# Sensor columns used by all models
sensor_columns = [
    "Battery_Voltage","Battery_Current","Battery_Temperature",
    "Motor_Temperature","Motor_Vibration","Motor_Torque",
    "Motor_RPM","Power_Consumption","Brake_Pressure",
    "Tire_Pressure","Tire_Temperature","Suspension_Load",
    "Ambient_Temperature","Ambient_Humidity","Driving_Speed"
]

data = raw_data[sensor_columns]

# Fill missing values
data = data.fillna(method='ffill').fillna(method='bfill')

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler and processed data
joblib.dump(scaler, "../models/scaler.pkl")
pd.DataFrame(data_scaled, columns=sensor_columns).to_csv("../data/processed_vehicle_data.csv", index=False)

print("Preprocessing complete. Processed data and scaler saved.")