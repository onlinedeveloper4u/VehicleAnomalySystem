# test_models.py
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# -------------------------------
# Load new sensor data
# -------------------------------
new_data = pd.read_csv("../data/new_sensor_data.csv")

# Sensor columns (same as used in training)
sensor_columns = [
    "Battery_Voltage","Battery_Current","Battery_Temperature",
    "Motor_Temperature","Motor_Vibration","Motor_Torque",
    "Motor_RPM","Power_Consumption","Brake_Pressure",
    "Tire_Pressure","Tire_Temperature","Suspension_Load",
    "Ambient_Temperature","Ambient_Humidity","Driving_Speed"
]

# Fill missing values (forward/backward fill)
new_data = new_data[sensor_columns].ffill().bfill()

# -------------------------------
# Scale data
# -------------------------------
scaler = joblib.load("../models/scaler.pkl")
X_scaled = scaler.transform(new_data)

# ===============================
# Isolation Forest
# ===============================
iso_model = joblib.load("../models/isolation_forest_model.pkl")
iso_pred = iso_model.predict(X_scaled)
iso_score = iso_model.decision_function(X_scaled)

# ===============================
# One-Class SVM
# ===============================
svm_model = joblib.load("../models/one_class_svm_model.pkl")
svm_pred = svm_model.predict(X_scaled)
svm_score = svm_model.decision_function(X_scaled)

# ===============================
# Autoencoder
# ===============================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load trained autoencoder
input_dim = X_scaled.shape[1]
autoencoder = Autoencoder(input_dim)
autoencoder.load_state_dict(torch.load("../models/autoencoder_model.pth"))
autoencoder.eval()

# Convert to tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    X_reconstructed = autoencoder(X_tensor).numpy()

# Compute reconstruction error
recon_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

# Autoencoder anomaly prediction using threshold
threshold = np.mean(recon_error) + 2 * np.std(recon_error)
auto_pred = np.where(recon_error > threshold, -1, 1)

# ===============================
# Save results
# ===============================
results = new_data.copy()
results["IsolationForest_Pred"] = iso_pred
results["IsolationForest_Score"] = iso_score
results["OneClassSVM_Pred"] = svm_pred
results["OneClassSVM_Score"] = svm_score
results["Autoencoder_Pred"] = auto_pred
results["Autoencoder_Score"] = recon_error

results.to_csv("../data/new_sensor_data_results_all_models.csv", index=False)
print("All models tested. Results saved to new_sensor_data_results_all_models.csv")