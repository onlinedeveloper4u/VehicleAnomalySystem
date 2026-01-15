# preprocess.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load CSV file
data = pd.read_csv("../data/ev_dataset.csv")  # adjust path if needed

# Keep only sensor columns
columns_to_keep = [
    "Battery_Voltage","Battery_Current","Battery_Temperature",
    "Motor_Temperature","Motor_Vibration","Motor_Torque",
    "Motor_RPM","Power_Consumption","Brake_Pressure",
    "Tire_Pressure","Tire_Temperature","Suspension_Load",
    "Ambient_Temperature","Ambient_Humidity","Driving_Speed"
]
data = data[columns_to_keep]

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save cleaned & scaled data
pd.DataFrame(data_scaled, columns=columns_to_keep).to_csv("../data/ev_dataset_clean.csv", index=False)

# Save the scaler for later
joblib.dump(scaler, "../models/scaler.pkl")

print("Preprocessing complete. Cleaned data saved as ev_dataset_clean.csv")