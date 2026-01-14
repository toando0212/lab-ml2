# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "tabulate",
#     "numpy",
# ]
# ///

import pandas as pd
import numpy as np
import sys
import os

# Configure pandas to display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    print("=== LAB 1.2: DATA ISSUES ANALYSIS & PREPARATION ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/data.csv')
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset from: {dataset_path}")
    except FileNotFoundError:
        print(f"Error: Dataset not found during execution.")
        sys.exit(1)

    print(f"Shape (Full): {df.shape} (Rows, Cols)")
    
    # Slice to first 10 columns as requested
    df = df.iloc[:, :10]
    print(f"Shape (Analyzed): {df.shape} (First 10 Cols)\n")

    print("-" * 60)
    print("ANALYSIS REPORT")
    print("-" * 60)

    # --- 2. Check for Missing Data ---
    print("\n[1] MISSING DATA CHECK")
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    
    if not cols_with_nulls.empty:
        print(f"ISSUE FOUND: specific columns contain Null values:")
        for col, count in cols_with_nulls.items():
            percent = (count / len(df)) * 100
            print(f" - Column '{col}': {count} missing values ({percent:.1f}%)")
            if percent == 100:
                print(f"   -> DIAGNOSIS: This is likely a garbage column (parsing artifact).")
                print(f"   -> RECOMMENDATION: Drop column '{col}'.")
    else:
        print("RESULT: No missing values found in any column.")

    # --- 3. Check for Duplicates ---
    print("\n[2] DUPLICATE RECORDS CHECK")
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"ISSUE FOUND: {dupes} duplicate rows detected.")
        print("   -> RECOMMENDATION: Inspect and drop duplicates.")
    else:
        print("RESULT: No duplicate rows found. Data is unique.")

    # --- 4. Check Data Types & Labels ---
    print("\n[3] DATA TYPES & LABEL CHECK")
    # check diagnosis
    if 'diagnosis' in df.columns:
        diag_dtype = df['diagnosis'].dtype
        unique_vals = df['diagnosis'].unique()
        print(f" - Target Column 'diagnosis': Type={diag_dtype}, Unique Values={unique_vals}")
        if diag_dtype == 'object' or diag_dtype == 'O':
            print("   -> STATUS: Valid Target Label (Categorical).")
            print("   -> PREPARATION: Encode to Binary (e.g., M=1, B=0) for model training.")
    
    # check ID
    if 'id' in df.columns:
        print(f" - Column 'id': Detected. ID columns usually carry no predictive signal.")
        print("   -> RECOMMENDATION: Drop 'id' column to prevent overfitting.")

    # --- 5. Check Feature Scales (Preparation Check) ---
    print("\n[4] FEATURE SCALING CHECK")
    # Select a few representative numeric columns to compare scales
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove id/unnamed if present for this check
    cols_to_check = [c for c in ['area_mean', 'smoothness_mean', 'texture_se'] if c in numeric_cols]
    
    if cols_to_check:
        stats = df[cols_to_check].agg(['min', 'max', 'mean'])
        print("Comparing scales of sample features:")
        print(stats.to_markdown())
        
        max_vals = stats.loc['max']
        min_in_group = max_vals.min()
        max_in_group = max_vals.max()
        
        if max_in_group / (min_in_group + 1e-9) > 100: # Heuristic threshold
            print("\n   -> ISSUE: Features have vastly different scales (e.g. Area ~2500 vs Smoothness ~0.16).")
            print("   -> RECOMMENDATION: Apply Feature Scaling (StandardScaler or MinMaxScaler).")
    
    print("\n" + "="*30)
    print("SUMMARY OF NEXT STEPS (DATA PREP)")
    print("="*30)
    print("1. DROP 'Unnamed: 32' (Missing Data)")
    print("2. DROP 'id' (Irrelevant Label)")
    print("3. ENCODE 'diagnosis' (String -> Number)")
    print("4. SCALE features (Normalize ranges)")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
