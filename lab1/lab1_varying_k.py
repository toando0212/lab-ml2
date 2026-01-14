# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
# ]
# ///

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys

def main():
    print("=== BREAST CANCER: Varying K Analysis ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/data.csv')
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Slice to first 10 columns (8 features)
    df = df.iloc[:, :10]
    features = df.drop(columns=['id', 'diagnosis'], errors='ignore')
    
    print(f"Features: {features.shape[1]}")
    
    # 3. Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    # 4. Apply PCA
    pca = PCA()
    pca.fit(scaled_data)
    
    # 5. Get cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    k_values = np.arange(1, len(cumulative_variance) + 1)
    
    # 6. Create plot
    output_dir = os.path.join(script_dir, 'assets')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cumulative_variance, 'bo-', linewidth=2, markersize=10, label='Cumulative Variance')
    
    # Highlight key points
    plt.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Threshold')
    
    # Mark important k values
    important_k = [2, 3, 4]
    important_var = [cumulative_variance[k-1] for k in important_k]
    plt.scatter(important_k, important_var, color='red', s=200, zorder=5, edgecolors='black', linewidth=2)
    
    # Annotate
    plt.annotate(f'k=2: {cumulative_variance[1]:.1%}', 
                xy=(2, cumulative_variance[1]), xytext=(2.5, 0.75),
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'k=4: {cumulative_variance[3]:.1%}', 
                xy=(4, cumulative_variance[3]), xytext=(5, 0.92),
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.title('Impact of Number of Components on Variance Retention (Breast Cancer)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Number of Principal Components (k)', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0.6, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    
    output_path = os.path.join(output_dir, 'pca_varying_k.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()
    
    # 7. Print table data
    print("\nCumulative Variance by k:")
    print(f"{'k':<4} | {'Cumulative Variance':<20}")
    print("-" * 30)
    for k, var in zip(k_values, cumulative_variance):
        print(f"{k:<4} | {var:.3f} ({var*100:.1f}%)")

if __name__ == "__main__":
    main()
