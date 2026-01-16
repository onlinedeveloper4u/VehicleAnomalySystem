# train_models.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# ===============================
# 0) Load preprocessed data
# ===============================
data_path = "../data/processed_vehicle_data.csv"
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

data = pd.read_csv(data_path)
X = data.values
print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

# ===============================
# 1) Isolation Forest with dynamic contamination
# ===============================
print("\nTraining Isolation Forest...")

iso_model = IsolationForest(random_state=42)
iso_model.fit(X)

# Compute anomaly scores
scores = -iso_model.decision_function(X)  # higher = more abnormal

# Automatic threshold: mean + 3*std
threshold_iso = np.mean(scores) + 3 * np.std(scores)
contamination_iso = np.mean(scores >= threshold_iso)

# Label anomalies
iso_anomalies = X[scores >= threshold_iso]
print(f"Isolation Forest: Detected {len(iso_anomalies)} anomalies ({contamination_iso*100:.2f}% of data)")

# Optional: plot distribution
plt.figure(figsize=(8,5))
plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
plt.axvline(threshold_iso, color='red', linestyle='--', label=f'Threshold ({contamination_iso*100:.2f}%)')
plt.xlabel("Anomaly Score")
plt.ylabel("Number of Samples")
plt.title("Isolation Forest Anomaly Score Distribution")
plt.legend()
plt.savefig(os.path.join(model_dir, "isolation_forest_score_distribution.png"))
plt.close()

# Save model
joblib.dump(iso_model, os.path.join(model_dir, "isolation_forest_model.pkl"))
print("Isolation Forest model trained and saved.")

# ===============================
# 2) One-Class SVM with dynamic nu
# ===============================
print("\nTraining One-Class SVM...")

# Fit temporary SVM with arbitrary nu to compute scores
temp_svm = OneClassSVM(nu=contamination_iso, kernel='rbf', gamma='scale')
temp_svm.fit(X)
svm_scores = temp_svm.decision_function(X)  # higher = normal, lower = abnormal

# Use statistical heuristic for nu: fraction of extreme points
threshold_svm = np.mean(svm_scores) - 3 * np.std(svm_scores)  # lower = anomaly
nu_dynamic = np.mean(svm_scores <= threshold_svm)

# Train final SVM with dynamic nu
svm_model = OneClassSVM(nu=nu_dynamic, kernel='rbf', gamma='scale')
svm_model.fit(X)

# Predict anomalies
svm_labels = svm_model.predict(X)  # -1 = anomaly, 1 = normal
n_svm_anomalies = np.sum(svm_labels == -1)
print(f"One-Class SVM: Detected {n_svm_anomalies} anomalies ({nu_dynamic*100:.2f}% of data)")

# Save model
joblib.dump(svm_model, os.path.join(model_dir, "one_class_svm_model.pkl"))
print("One-Class SVM model trained and saved.")

# ===============================
# 3) Autoencoder (PyTorch)
# ===============================
print("\nTraining Autoencoder...")

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

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare data
X_tensor = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = X.shape[1]
autoencoder = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        batch_data = batch[0].to(device)
        optimizer.zero_grad()
        outputs = autoencoder(batch_data)
        loss = criterion(outputs, batch_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# Save autoencoder
torch.save(autoencoder.state_dict(), os.path.join(model_dir, "autoencoder_model.pth"))
print("Autoencoder trained and saved.")
