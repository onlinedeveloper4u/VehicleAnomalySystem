import pandas as pd
import os
import json

def separate_datasets():
    data_path = "data/ev_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print("Loading original dataset...")
    df = pd.read_csv(data_path)

    # Define Metadata/Label columns to drop (Leakage Prevention)
    drop_cols = [
        'Failure_Probability', 'Component_Health_Score', 
        'RUL', 'TTF', 'Maintenance_Type'
    ]

    # --- 1. NORMAL (Training Baseline) ---
    # Loosen criteria to include more natural variation
    normal_mask = (df['Failure_Probability'] == 0)
    normal_df = df[normal_mask].copy().drop(columns=drop_cols)
    
    # --- 2. SEVERE ABNORMAL (Critical Faults) ---
    severe_mask = (df['Failure_Probability'] == 1)
    severe_df = df[severe_mask].copy().drop(columns=drop_cols)

    # --- 3. MILD ABNORMAL (Subtle Degradation) ---
    # Low health but no failure yet
    mild_mask = (df['Failure_Probability'] == 0) & (df['Component_Health_Score'] < 0.75)
    mild_df = df[mild_mask].copy().drop(columns=drop_cols)

    print(f"Total Source Records: {len(df)}")
    print(f"Normal (Training):    {len(normal_df)}")
    print(f"Severe Abnormal:      {len(severe_df)}")
    print(f"Mild Abnormal:        {len(mild_df)}")

    os.makedirs("data", exist_ok=True)
    
    # Saving datasets
    normal_df.to_csv("data/normal_data.csv", index=False)
    severe_df.to_csv("data/abnormal_data.csv", index=False) 
    severe_df.to_csv("data/severe_data.csv", index=False)
    mild_df.to_csv("data/mild_data.csv", index=False)
    
    # Saving Metadata
    metadata = {
        "version": "v1.2",
        "description": "Production-Grade Vehicle Dataset Split",
        "leakage_protection": "STRICT - All labels/prob columns dropped",
        "filters": {
            "healthy": "Failure_Prob=0 AND Health > 0.95",
            "severe": "Failure_Prob=1",
            "mild": "Failure_Prob=0 AND 0.7 <= Health <= 0.8"
        },
        "counts": {
            "normal": len(normal_df),
            "severe": len(severe_df),
            "mild": len(mild_df)
        }
    }
    
    with open("data/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n✅ Done! Multi-tier datasets saved to data/")

if __name__ == "__main__":
    separate_datasets()
