import pandas as pd
import numpy as np
import os
import json
import argparse
from src.ml.predictor import AnomalyDetector
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def run_evaluation(sample_size: int = 1000, version: str = "v1"):
    """
    Run evaluation on the anomaly detection model.
    
    Args:
        sample_size: Number of samples to evaluate
        version: Model version to evaluate
    """
    print(f"Starting Model Evaluation (version: {version})...")
    
    # 1. Load Data
    data_path = f"data/{'normal' if 'normal' in version else 'abnormal'}_data.csv"
    if not os.path.exists(data_path):
        data_path = "data/normal_data.csv" # Fallback
    
    df = pd.read_csv(data_path)
    
    # 2. Load Model
    detector = AnomalyDetector(model_dir="models", version=version)
    
    # 3. Sample data for evaluation
    actual_sample_size = min(sample_size, len(df))
    sample_df = df.sample(n=actual_sample_size, random_state=42)
    
    print(f"Running predictions on {actual_sample_size} samples...")
    results = detector.predict(sample_df)
    
    n_anomalies = sum(results["is_anomaly"])
    print(f"Detected Anomalies: {n_anomalies} / {actual_sample_size} ({n_anomalies/actual_sample_size*100:.1f}%)")
    
    # 4. Generate report directory
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # 5. Performance Analysis
    # Use Failure_Probability as a proxy for real labels if available
    if "Failure_Probability" not in sample_df.columns:
        print("\nNote: Failure_Probability column not found. Skipping metric computation.")
        return
    
    print("\n--- Anomaly Detection Performance ---")
    fail_prob = sample_df["Failure_Probability"].values
    
    # Correlation analysis
    is_anomaly_array = np.array(results["is_anomaly"], dtype=int)
    scores = np.array(results["scores"])
    
    correlation = np.corrcoef(scores, fail_prob)[0, 1]
    print(f"Score-Failure Probability Correlation: {correlation:.4f}")

    # Binary Classification (treat Failure_Probability > 0.5 as anomaly)
    binary_labels = (fail_prob > 0.5).astype(int)
    
    if len(np.unique(binary_labels)) < 2:
        print("Warning: Only one class present in labels. Cannot compute all metrics.")
        return
        
    print(f"\nEvaluating (using Failure_Probability > 0.5 as proxy labels):")
    
    metrics = {
        "roc_auc": float(roc_auc_score(binary_labels, is_anomaly_array)),
        "precision": float(precision_score(binary_labels, is_anomaly_array, zero_division=0)),
        "recall": float(recall_score(binary_labels, is_anomaly_array, zero_division=0)),
        "f1_score": float(f1_score(binary_labels, is_anomaly_array, zero_division=0))
    }
    
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    # Save metrics to JSON
    metrics_output = {
        "timestamp": str(pd.Timestamp.now()),
        "sample_size": actual_sample_size,
        "model_version": version,
        "thresholds": detector.thresholds,
        "correlation": float(correlation),
        "metrics": metrics
    }
    
    metrics_path = os.path.join(report_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=4)
    
    print(f"\nMetrics saved to {metrics_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--version", type=str, default="v1", help="Model version to evaluate")
    args = parser.parse_args()
    
    run_evaluation(sample_size=args.samples, version=args.version)
