# test_model.py
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("../models/isolation_forest_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Example new sensor data (replace with your own readings)
new_data = np.array([[380, -35, 30, 45, 0.4, 180, 1900, 25, 40, 32, 30, 150, 15, 50, 60]])

# Scale using saved scaler
new_data_scaled = scaler.transform(new_data)

# Predict anomaly
prediction = model.predict(new_data_scaled)  # 1 = normal, -1 = anomaly
score = model.decision_function(new_data_scaled)

print("Prediction (1=normal, -1=anomaly):", prediction)
print("Anomaly score:", score)