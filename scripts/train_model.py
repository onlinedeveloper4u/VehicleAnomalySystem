# train_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load cleaned data
data = pd.read_csv("../data/ev_dataset_clean.csv")

# Train Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(data)

# Save trained model
joblib.dump(model, "../models/isolation_forest_model.pkl")

# Optional: check anomalies on training data
anomalies = model.predict(data)  # 1=normal, -1=anomaly
print("Training complete. Anomalies detected in training data:", sum(anomalies == -1))
print("Model saved as isolation_forest_model.pkl")