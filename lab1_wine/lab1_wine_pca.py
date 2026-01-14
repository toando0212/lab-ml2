# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scikit-learn",
# ]
# ///

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys

def main():
    print("=== WINE DATASET: PCA ANALYSIS ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/WineQT.csv')
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Prepare Features
    cols_to_drop = ['quality', 'Id']
    features_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Feature Scope: All {features_df.shape[1]} Features")
    print(f"Features: {list(features_df.columns)}")
    print(f"Shape: {features_df.shape}\n")
    
    # 3. Standardization (Critical for PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    # 4. Apply PCA
    pca = PCA()
    pca.fit(scaled_data)
    
    # 5. Analyze Variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("[PCA Result Summary]")
    print(f"{'PC #':<6} | {'Exp. Variance':<15} | {'Cumulative':<15}")
    print("-" * 42)
    
    count_95 = 0
    
    for i, (exp_var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        pc_num = i + 1
        print(f"PC{pc_num:<3} | {exp_var:.5f}         | {cum_var:.5f}")
        
        if count_95 == 0 and cum_var >= 0.95:
            count_95 = pc_num

    print("-" * 42)
    print(f"\nINFO: To preserve 95% of information, you need {count_95} Principal Components.")
    print(f"      (Reduced dimensions from {features_df.shape[1]} -> {count_95})")

if __name__ == "__main__":
    main()
