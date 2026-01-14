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

def analyze_feature_type(series):
    """
    Rigorously determine feature type based on data.
    Returns: (Discrete/Continuous, Quantitative/Qualitative, Explanation)
    """
    # 1. Quantitative vs Qualitative
    if pd.api.types.is_numeric_dtype(series):
        quant_qual = "Quantitative"
    else:
        quant_qual = "Qualitative"
        return "Discrete", "Qualitative", "Non-numeric data type"

    # 2. Discrete vs Continuous (Logic for Numerics)
    n_unique = series.nunique()
    n_total = len(series)
    ratio = n_unique / n_total
    
    # Logic: 
    # - Float type is usually Continuous.
    # - Integer type with FEW unique values is Discrete.
    # - Integer type with MANY unique values (e.g. ID, large counts) can be considered Continuous for stats, 
    #   but technically Discrete. However, in ML context:
    #   Threshold: If unique values < 15 -> Distinct categories -> Discrete
    #   Else -> Treated as Continuous spectrum.
    
    is_float = pd.api.types.is_float_dtype(series)
    
    if is_float:
        dist_cont = "Continuous"
        explanation = f"Float type, {n_unique} unique values."
    else:
        # Integer
        if n_unique < 20:
            dist_cont = "Discrete"
            explanation = f"Integer, only {n_unique} distinct values (Categories)."
        else:
            dist_cont = "Continuous" # Effectively continuous
            explanation = f"Integer, but High Cardinality ({n_unique} distinct values)."

    return dist_cont, quant_qual, explanation

def main():
    print("=== WINEQT DATASET: RIGOROUS FEATURE ANALYSIS ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/WineQT.csv')
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)
        
    # Drop Id if present (it's just an index)
    if 'Id' in df.columns:
        print("Note: Dropping 'Id' column from analysis (Metadata).\n")
        df_analysis = df.drop(columns=['Id'])
    else:
        df_analysis = df

    # 2. Basic Counts
    n_samples, n_dims = df_analysis.shape
    print(f"Samples (Rows): {n_samples}")
    print(f"Dimensions (Columns): {n_dims}")
    print("-" * 60)

    # 3. Analyze Each Feature
    results = []
    
    for col in df_analysis.columns:
        d_c, q_q, expl = analyze_feature_type(df_analysis[col])
        results.append({
            "Feature": col,
            "Discrete/Continuous": d_c,
            "Quantitative/Qualitative": q_q,
            "Explanation": expl
        })
        
    # 4. Print Table
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    
    # 5. Save to File for Documentation
    output_path = os.path.join(script_dir, '1.1_analysis_report.txt')
    with open(output_path, 'w') as f:
        f.write(df_results.to_markdown(index=False))
    print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    main()
