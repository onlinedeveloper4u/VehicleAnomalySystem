import pandas as pd
import numpy as np
import os
import json
from src.models.predictor import AnomalyDetector
from src.utils.visualizer import plot_anomaly_distribution, plot_agreement_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

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

        # Binary Classification Mockup (treating high prob as 1)
        # Assuming Failure_Probability > 0.5 is an anomaly
        binary_labels = (fail_prob > 0.5).astype(int)
        
        if len(np.unique(binary_labels)) > 1:
            print(f"\nEvaluating performance (using Failure_Probability > 0.5 as proxy labels):")
            
            # Global system evaluation
            global_anomaly = np.array(results["is_anomaly"], dtype=int)
            print(f"\nGLOBAL MULTI-MODEL SYSTEM:")
            print(f"ROC-AUC:   {roc_auc_score(binary_labels, global_anomaly):.4f}")
            print(f"Precision: {precision_score(binary_labels, global_anomaly, zero_division=0):.4f}")
            print(f"Recall:    {recall_score(binary_labels, global_anomaly, zero_division=0):.4f}")
            print(f"F1-score:  {f1_score(binary_labels, global_anomaly, zero_division=0):.4f}")

            # Individual model evaluation
            model_metrics = {}
            for model_name in results["details"]:
                is_anomaly = np.array(results["details"][model_name]["is_anomaly"], dtype=int)
                m = {
                    "roc_auc": float(roc_auc_score(binary_labels, is_anomaly)),
                    "precision": float(precision_score(binary_labels, is_anomaly, zero_division=0)),
                    "recall": float(recall_score(binary_labels, is_anomaly, zero_division=0)),
                    "f1_score": float(f1_score(binary_labels, is_anomaly, zero_division=0))
                }
                model_metrics[model_name] = m
                print(f"\nMODEL: {model_name.upper()}")
                print(f"ROC-AUC:   {m['roc_auc']:.4f}")
                print(f"Precision: {m['precision']:.4f}")
                print(f"Recall:    {m['recall']:.4f}")
                print(f"F1-score:  {m['f1_score']:.4f}")

            # Save metrics to JSON
            metrics_path = os.path.join(report_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({
                    "timestamp": str(pd.Timestamp.now()),
                    "sample_size": len(sample_df),
                    "global_system": {
                        "roc_auc": float(roc_auc_score(binary_labels, global_anomaly)),
                        "precision": float(precision_score(binary_labels, global_anomaly, zero_division=0)),
                        "recall": float(recall_score(binary_labels, global_anomaly, zero_division=0)),
                        "f1_score": float(f1_score(binary_labels, global_anomaly, zero_division=0))
                    },
                    "individual_models": model_metrics
                }, f, indent=4)
            print(f"\nMetrics saved to {metrics_path}")
    
    print(f"\nEvaluation complete. Plots generated in '{report_dir}/'")

if __name__ == "__main__":
    run_evaluation()
