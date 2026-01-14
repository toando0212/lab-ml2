# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///

import pandas as pd
import numpy as np
import os
import sys

def main():
    print("=== WINE DATASET: CORRELATION ANALYSIS ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/WineQT.csv')
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Prepare Features
    # Target is 'quality', Id is metadata
    cols_to_drop = ['quality', 'Id']
    features_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Features analyzed ({features_df.shape[1]}): {list(features_df.columns)}\n")
    
    # 3. Calculate Correlation Matrix
    corr_matrix = features_df.corr().abs()

    # 4. Find Max Correlation (excluding diagonal)
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    masked_corr = corr_matrix * mask
    max_corr_val = masked_corr.max().max()
    
    row_idx, col_idx = np.unravel_index(masked_corr.to_numpy().argmax(), masked_corr.shape)
    
    feature_1 = features_df.columns[row_idx]
    feature_2 = features_df.columns[col_idx]
    
    print("=" * 60)
    print("MOST CORRELATED PAIR")
    print("=" * 60)
    print(f"Feature A: {feature_1}")
    print(f"Feature B: {feature_2}")
    print(f"Correlation: {max_corr_val:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
