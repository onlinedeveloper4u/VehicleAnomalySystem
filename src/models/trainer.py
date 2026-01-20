import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import os
import json
from src.models.autoencoder import Autoencoder

class ModelTrainer:
    def __init__(self, model_dir="../models", version="v1"):
        self.version = version
        self.model_dir = os.path.join(model_dir, version)
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_isolation_forest(self, X):
        print("Training Isolation Forest...")
        iso_model = IsolationForest(random_state=42)
        iso_model.fit(X)
        
        # Calculate Threshold: 99th percentile of normal training scores
        scores = -iso_model.decision_function(X)
        threshold = float(np.percentile(scores, 99))
        
        joblib.dump(iso_model, os.path.join(self.model_dir, "isolation_forest_model.pkl"))
        return threshold

    def train_one_class_svm(self, X, contamination=0.05):
        print("Training One-Class SVM...")
        # Initial fit to estimate nu (using a heuristic or contamination)
        # Using the logic from original script:
        # 1. Fit temp SVM
        temp_svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
        temp_svm.fit(X)
        svm_scores = temp_svm.decision_function(X)
        
        # 2. Dynamic Nu
        threshold_temp = np.mean(svm_scores) - 3 * np.std(svm_scores)
        nu_dynamic = np.mean(svm_scores <= threshold_temp)
        # Ensure nu is valid (0 < nu <= 1)
        nu_dynamic = max(0.001, min(nu_dynamic, 0.999)) # Bounds check
        
        svm_model = OneClassSVM(nu=nu_dynamic, kernel='rbf', gamma='scale')
        svm_model.fit(X)
        
        # Calculate threshold for saving (though SVM predicts -1/1, we might want score threshold logic too?)
        # Standard SVM predict uses 0 as threshold for decision_function. 
        # But let's save the calculated "score threshold" if we were to use decision_function manually.
        # However, one_class_svm.predict() uses 0.
        # But wait, original script used: 
        # threshold_svm = np.mean(svm_scores) - 3 * np.std(svm_scores) 
        # AND THEN RETRAINED. The retrained model has its own decision boundary (0).
        # So for the final model, 'predict' is sufficient? 
        # Actually, for anomaly detection, having a continuous score is nice.
        # Let's simple return 0 as threshold for standard SVM usage, or we can use the distribution of the NEW model scores?
        # Let's use the new model scores.
        
        final_scores = svm_model.decision_function(X)
        # For SVM, decision_function < 0 is anomaly. 
        # But we use scores < threshold logic. 
        # Let's find the 1st percentile (since lower is more anomalous)
        threshold = float(np.percentile(final_scores, 1))
        
        joblib.dump(svm_model, os.path.join(self.model_dir, "one_class_svm_model.pkl"))
        return threshold

    def train_autoencoder(self, X, epochs=50, batch_size=32):
        print("Training Autoencoder...")
        input_dim = X.shape[1]
        autoencoder = Autoencoder(input_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch_data = batch[0].to(self.device)
                optimizer.zero_grad()
                outputs = autoencoder(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # if (epoch+1) % 10 == 0:
            #     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
                
        # Calculate threshold
        autoencoder.eval()
        with torch.no_grad():
            X_recon = autoencoder(X_tensor.to(self.device)).cpu().numpy()
            
        recon_error = np.mean((X - X_recon) ** 2, axis=1)
        # Use 99th percentile of reconstruction errors
        threshold = float(np.percentile(recon_error, 99))
        
        torch.save(autoencoder.state_dict(), os.path.join(self.model_dir, "autoencoder_model.pth"))
        return threshold

    def train_all(self, X):
        thresholds = {}
        
        thresholds["isolation_forest"] = self.train_isolation_forest(X)
        thresholds["one_class_svm"] = self.train_one_class_svm(X) 
        thresholds["autoencoder"] = self.train_autoencoder(X)
        
        # Save thresholds
        with open(os.path.join(self.model_dir, "thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=4)
        print(f"Models and thresholds saved to {self.model_dir}")

if __name__ == "__main__":
    from src.preprocessing.transformer import DataPreprocessor
    
    # Load and process data
    raw_data = pd.read_csv("data/ev_dataset.csv") # Assuming run from root
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(raw_data)
    trainer = ModelTrainer(model_dir="models", version="v1")
    preprocessor.save(os.path.join(trainer.model_dir, "scaler.pkl"))
    trainer.train_all(X_scaled)
