import joblib
import torch
import numpy as np
import os
import json
from src.preprocessing.transformer import DataPreprocessor
from src.models.autoencoder import Autoencoder

class AnomalyDetector:
    def __init__(self, model_dir="models", version="v1", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = os.path.join(model_dir, version)
        self.models = {}
        self.thresholds = {}
        self.preprocessor = DataPreprocessor()
        self.version = version
        
        self.load_models()

    def load_models(self):
        # Load Scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.preprocessor.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        # Load Thresholds
        thresholds_path = os.path.join(self.model_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)
        else:
             # Fallback or error? For now, let's warn and use defaults if possible, but better to error.
             print(f"Warning: Thresholds file not found at {thresholds_path}. Using default/zero thresholds.")
             self.thresholds = {"isolation_forest": 0.0, "one_class_svm": 0.0, "autoencoder": 0.0}

        # Load Isolation Forest
        iso_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")
        if os.path.exists(iso_path):
            self.models["isolation_forest"] = joblib.load(iso_path)

        # Load One-Class SVM
        svm_path = os.path.join(self.model_dir, "one_class_svm_model.pkl")
        if os.path.exists(svm_path):
            self.models["one_class_svm"] = joblib.load(svm_path)

        # Load Autoencoder
        ae_path = os.path.join(self.model_dir, "autoencoder_model.pth")
        if os.path.exists(ae_path):
            # Autoencoder training happened on processed features
            input_dim = len(self.preprocessor.feature_columns)
            ae = Autoencoder(input_dim).to(self.device)
            ae.load_state_dict(torch.load(ae_path, map_location=self.device))
            ae.eval()
            self.models["autoencoder"] = ae

    def predict(self, data, threshold_overrides=None):
        """
        Predicts anomalies for the given DataFrame.
        Returns a global anomaly flag and details per model.
        """
        # 1. Start with loaded defaults
        current_thresholds = self.thresholds.copy()
        if threshold_overrides:
            current_thresholds.update(threshold_overrides)
            
        # 2. Preprocess
        X_scaled = self.preprocessor.transform(data)
        
        results = {}
        
        # Isolation Forest
        if "isolation_forest" in self.models:
            model = self.models["isolation_forest"]
            scores = -model.decision_function(X_scaled)
            is_anomaly = scores > current_thresholds.get("isolation_forest", 0)
            results["isolation_forest"] = {
                "score": [round(float(s), 4) for s in scores],
                "is_anomaly": is_anomaly.tolist()
            }

        # One-Class SVM
        if "one_class_svm" in self.models:
            model = self.models["one_class_svm"]
            scores = model.decision_function(X_scaled) 
            is_anomaly = scores < current_thresholds.get("one_class_svm", 0)
            results["one_class_svm"] = {
                "score": [round(float(s), 4) for s in scores],
                "is_anomaly": is_anomaly.tolist()
            }

        # Autoencoder
        if "autoencoder" in self.models:
            model = self.models["autoencoder"]
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                X_recon = model(X_tensor).cpu().numpy()
            
            recon_error = np.mean((X_scaled - X_recon) ** 2, axis=1)
            threshold = current_thresholds.get("autoencoder", 0)
            is_anomaly = recon_error > threshold
            results["autoencoder"] = {
                "score": [round(float(s), 4) for s in recon_error],
                "is_anomaly": is_anomaly.tolist()
            }

        # Aggregate result: Majority Vote (at least 2 models must agree)
        # This makes the system more robust to false positives from a single model.
        # Requirement: "The system shall classify data points as normal or anomalous"
        
        n_samples = len(data)
        anomaly_votes = np.zeros(n_samples, dtype=int)
        
        for key in results:
            anomaly_votes += np.array(results[key]["is_anomaly"], dtype=int)
            
        # If 2 or more models flag it, we consider it a global anomaly
        final_anomaly = anomaly_votes >= 2
            
        return {
            "is_anomaly": final_anomaly.tolist(),
            "votes": anomaly_votes.tolist(),
            "details": results
        }
