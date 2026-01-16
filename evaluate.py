import pandas as pd
import numpy as np
import os
from src.models.predictor import AnomalyDetector
from src.utils.visualizer import plot_anomaly_distribution, plot_agreement_matrix
from sklearn.metrics import roc_auc_score

def run_evaluation():
    print("Starting Model Evaluation...")
    
    # 1. Load Data
    data_path = "data/ev_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Load Models
    detector = AnomalyDetector(model_dir="models")
    
    # 3. Predict on a sample to speed up (or full dataset if small enough)
    # The dataset has 175k rows. Let's use 20k for evaluation to be fast.
    sample_df = df.sample(n=min(150000, len(df)), random_state=42)
    
    print(f"Running predictions on {len(sample_df)} samples...")
    results = detector.predict(sample_df)
    
    # 4. Generate Visualizations
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    for model_name in results["details"]:
        scores = results["details"][model_name]["score"]
        threshold = detector.thresholds.get(model_name, 0)
        
        plot_anomaly_distribution(
            scores, 
            threshold, 
            model_name.replace("_", " ").title(),
            os.path.join(report_dir, f"{model_name}_distribution.png")
        )
        
    # Agreement plot
    plot_agreement_matrix(
        results["votes"],
        os.path.join(report_dir, "model_agreement.png")
    )
    
    # 5. Accuracy Correlation Analysis
    # We use Failure_Probability as a "proxy" for real labels if available
    if "Failure_Probability" in sample_df.columns:
        print("\n--- Accuracy Performance Analysis ---")
        fail_prob = sample_df["Failure_Probability"].values
        
        for model_name in results["details"]:
            is_anomaly = np.array(results["details"][model_name]["is_anomaly"], dtype=int)
            scores = np.array(results["details"][model_name]["score"])
            
            # Calibration: For OCSVM, scores are signed distance (positive = normal), 
            # so we invert for ROC calculation if needed, but easier to just use binary correlation.
            correlation = np.corrcoef(is_anomaly, fail_prob)[0, 1]
            print(f"{model_name.title()} Correlation with Failure Probability: {correlation:.4f}")

        # Global system correlation
        global_anomaly = np.array(results["is_anomaly"], dtype=int)
        global_corr = np.corrcoef(global_anomaly, fail_prob)[0, 1]
        print(f"Global Multi-Model System Correlation: {global_corr:.4f}")
        
        # Binary Classification Mockup (treating high prob as 1)
        # Assuming Failure_Probability > 0.5 is an anomaly
        binary_labels = (fail_prob > 0.5).astype(int)
        if len(np.unique(binary_labels)) > 1:
            auc = roc_auc_score(binary_labels, global_anomaly)
            print(f"System ROC-AUC (vs Prob > 0.5): {auc:.4f}")
    
    print(f"\nEvaluation complete. Plots generated in '{report_dir}/'")

if __name__ == "__main__":
    run_evaluation()
