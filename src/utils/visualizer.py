import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_anomaly_distribution(scores, threshold, model_name, output_path):
    """
    Plots the distribution of anomaly scores and the threshold line.
    """
    plt.figure(figsize=(10, 6))
    
    # Use log scale if scores vary wildly
    sns.histplot(scores, kde=True, bins=50)
    
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    
    plt.title(f'{model_name} Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Report saved to {output_path}")

def plot_agreement_matrix(votes, output_path):
    """
    Plots the count of votes (0 to 3) to show how much models agree.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x=votes)
    plt.title('Agreement Between Models (Votes per Record)')
    plt.xlabel('Number of Models Flagging Anomaly')
    plt.ylabel('Record Count')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
