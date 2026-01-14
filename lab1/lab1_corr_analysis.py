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
    print("=== LAB 1: CORRELATION ANALYSIS (FIRST 10 DIMENSIONS) ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/data.csv')
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Select First 10 Dimensions
    # Based on standard WBCCD: 0=id, 1=diagnosis, 2-9=features
    # We want columns 0 to 9.
    df_10 = df.iloc[:, :10]
    
    # 3. Filter for Numeric Features only
    # Drop 'id' and 'diagnosis' to focus on measurements
    # 'id' is distinct, 'diagnosis' is object.
    # We expect 8 numeric features.
    numeric_df = df_10.drop(columns=['id', 'diagnosis'], errors='ignore')
    numeric_df = numeric_df.select_dtypes(include=[np.number])

    print(f"Features analyzed ({numeric_df.shape[1]}): {list(numeric_df.columns)}")
    
    if numeric_df.empty:
        print("No numeric features found in the first 10 dimensions.")
        sys.exit(1)

    # 4. Calculate Correlation Matrix
    corr_matrix = numeric_df.corr().abs()

    # 5. Find Max Correlation (excluding diagonal)
    # Mask the diagonal (which is always 1.0)
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    # Get the max correlation value
    # We multiply by mask so diagonal becomes 0
    masked_corr = corr_matrix * mask
    
    max_corr_val = masked_corr.max().max()
    
    # Find the indices of the max value
    row_idx, col_idx = np.unravel_index(masked_corr.to_numpy().argmax(), masked_corr.shape)
    
    feature_1 = numeric_df.columns[row_idx]
    feature_2 = numeric_df.columns[col_idx]
    
    print("\n" + "="*40)
    print("MOST CORRELATED PAIR")
    print("="*40)
    print(f"Feature A: {feature_1}")
    print(f"Feature B: {feature_2}")
    print(f"Correlation: {max_corr_val:.6f}")
    print("="*40)

if __name__ == "__main__":
    main()
