import pandas as pd
import os

def separate_datasets():
    data_path = "data/ev_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print("Loading original dataset...")
    df = pd.read_csv(data_path)

    # Criteria for 'Normal' (Golden Healthy)
    # Failure Probability < 5% and Health Score > 90%
    normal_mask = (df['Failure_Probability'] < 0.05) & (df['Component_Health_Score'] > 0.9)
    
    normal_df = df[normal_mask].copy()
    abnormal_df = df[~normal_mask].copy()

    print(f"Total records: {len(df)}")
    print(f"Normal records: {len(normal_df)}")
    print(f"Abnormal records: {len(abnormal_df)}")

    os.makedirs("data", exist_ok=True)
    
    normal_path = "data/normal_data.csv"
    abnormal_path = "data/abnormal_data.csv"
    
    normal_df.to_csv(normal_path, index=False)
    abnormal_df.to_csv(abnormal_path, index=False)
    
    print(f"Done! Saved to {normal_path} and {abnormal_path}")

if __name__ == "__main__":
    separate_datasets()
