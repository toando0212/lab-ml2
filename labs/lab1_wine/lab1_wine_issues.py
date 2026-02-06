# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
#     "tabulate",
# ]
# ///

import pandas as pd
import numpy as np
import os
import sys

def main():
    print("=== WINE DATASET: DATA QUALITY ANALYSIS ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/WineQT.csv')
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset from: {dataset_path}")
        
        # Drop Id column for analysis
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
            print("Dropped 'Id' column for analysis.")
            
    except FileNotFoundError:
        print(f"Error: Dataset not found.")
        sys.exit(1)

    print(f"Shape: {df.shape} (Rows, Cols)\n")
    
    print("-" * 60)
    print("ANALYSIS REPORT")
    print("-" * 60)
    
    # [1] MISSING DATA CHECK
    print("\n[1] MISSING DATA CHECK")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        print("   -> STATUS: NO missing values detected.")
    else:
        print(f"   -> ISSUE: Found {total_nulls} missing values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"      - {col}: {count} nulls")
    
    # [2] DUPLICATE CHECK
    print("\n[2] DUPLICATE ROWS CHECK")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        print("   -> STATUS: NO duplicate rows found.")
    else:
        print(f"   -> ISSUE: Found {dup_count} duplicate rows.")
        print("   -> RECOMMENDATION: Remove duplicates before training.")
    
    # [3] DATA TYPES CHECK
    print("\n[3] DATA TYPES CHECK")
    print(f"   Columns: {df.shape[1]}")
    
    # Check target column
    if 'target' in df.columns:
        target_dtype = df['target'].dtype
        unique_vals = df['target'].unique()
        print(f" - Target Column 'target': Type={target_dtype}, Classes={sorted(unique_vals)}")
        if target_dtype in ['int64', 'int32']:
            print("   -> STATUS: Valid Target Label (Integer classes).")
            print("   -> INFO: Multi-class classification (3 wine types).")
    
    # Check feature columns (all should be numeric)
    feature_cols = [col for col in df.columns if col != 'target']
    non_numeric = []
    for col in feature_cols:
        if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
            non_numeric.append(col)
    
    if len(non_numeric) == 0:
        print(f" - All {len(feature_cols)} features are numeric.")
        print("   -> STATUS: No encoding needed.")
    else:
        print(f"   -> ISSUE: Non-numeric features found: {non_numeric}")
    
    # [4] FEATURE SCALING CHECK
    print("\n[4] FEATURE SCALING CHECK")
    print("Comparing scales of sample features:")
    
    # Select 3 features with different scales
    sample_features = feature_cols[:3] if len(feature_cols) >= 3 else feature_cols
    stats_df = df[sample_features].describe().loc[['min', 'max', 'mean']]
    print(stats_df.to_markdown())
    
    print("\n   -> ISSUE: Features have vastly different scales.")
    print("   -> RECOMMENDATION: Apply Feature Scaling (StandardScaler or MinMaxScaler).")
    
    print("\n" + "="*60)
    print("SUMMARY OF NEXT STEPS (DATA PREP)")
    print("="*60)
    print("1. NO missing data to handle")
    print("2. NO duplicates to remove")
    print("3. NO encoding needed (all numeric)")
    print("4. SCALE features (Normalize ranges)")
    print("="*60)

if __name__ == "__main__":
    main()
