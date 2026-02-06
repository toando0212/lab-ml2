# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "tabulate",
# ]
# ///

import pandas as pd
import sys
import os

# Configure pandas to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    print("=== LAB 1: FULL DATASET STATISTICAL ANALYSIS ===\n")
    
    # Path to dataset relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/data.csv')
    output_path = os.path.join(script_dir, '1.5_stats_results.md')
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {dataset_path}")
        sys.exit(1)

    print(f"Dataset Shape: {df.shape}")
    
    # Pre-processing: Select only numeric columns (30 features)
    numeric_df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    numeric_df = numeric_df.select_dtypes(include=['float64', 'int64'])
    
    print(f"Analyzing {numeric_df.shape[1]} numeric features.")

    # Calculate Stats
    mean_val = numeric_df.mean().rename("Mean")
    var_val = numeric_df.var().rename("Variance")
    
    # Combine Mean and Variance into one DataFrame for the first table
    summary_stats = pd.concat([mean_val, var_val], axis=1)

    # Matrices
    cov_matrix = numeric_df.cov()
    corr_matrix = numeric_df.corr()

    # Write to Markdown file
    print(f"Writing full results to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("# 1.5 Statistical Analysis Results (Full Dataset)\n\n")
        f.write("Below are the comprehensive statistical measures for all 30 numeric features.\n\n")
        
        f.write("## 1. Mean & Variance Table\n\n")
        f.write(summary_stats.round(4).to_markdown())
        f.write("\n\n")
        
        f.write("## 2. Covariance Matrix (Full 30x30)\n")
        f.write("*Note: This matrix shows the joint variability between all pairs of features.*\n\n")
        f.write(cov_matrix.round(4).to_markdown())
        f.write("\n\n")

        f.write("## 3. Correlation Matrix (Full 30x30)\n")
        f.write("*Note: Values range from -1 to 1. 1 indicates perfect positive correlation.*\n\n")
        f.write(corr_matrix.round(4).to_markdown())
        f.write("\n")

    print("Done! File generated successfully.")

if __name__ == "__main__":
    main()
