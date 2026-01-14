# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
# ]
# ///

import pandas as pd
import sys
import os

# Configure pandas to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    print("=== LAB 1: DATASET ANALYSIS ===\n")
    
    # Path to dataset -> Resolve relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/data.csv')
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {dataset_path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)

    # --- PART 1: Study the dataset (First 10 Dimensions) ---
    print("--- PART 1: First 10 Dimensions Inspection ---")
    
    # Select only the first 10 columns
    df_10 = df.iloc[:, :10]
    
    # Display info just like the notebook
    print("Info for first 10 columns:")
    print(df_10.info())
    
    print("\nFirst 5 rows of first 10 columns:")
    print(df_10.head())
    print("\n" + "="*50 + "\n")

    # --- PART 2: Analyze Data Issues ---
    print("--- PART 2: Data Quality Issues (Whole Dataset) ---")
    
    # 1. Check for Missing Data (NaNs)
    print("\n[Missing Values Check]")
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    
    if not cols_with_nulls.empty:
        print("Columns with null values:")
        print(cols_with_nulls)
    else:
        print("No missing values found.")

    # 2. Check for Duplicate Rows
    print("\n[Duplicate Rows Check]")
    duplicates = df.duplicated().sum()
    print(f"Count of duplicate rows: {duplicates}")

    # 3. Check for Anomalies (Statistics)
    print("\n[Descriptive Statistics]")
    print(df.describe())

if __name__ == "__main__":
    main()
