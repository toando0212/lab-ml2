# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
#     "seaborn",
# ]
# ///

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def main():
    print("=== WINE DATASET: PCA VISUALIZATION (Elbow, Kaiser, 2D) ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/WineQT.csv')
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        # Try finding it relative to the lab1 folder if adjacent
        alternative_path = os.path.join(script_dir, '../../Dataset/WineQT.csv') 
        if os.path.exists(alternative_path):
             df = pd.read_csv(alternative_path)
        else:
             print(f"Could not find dataset at {dataset_path}")
             sys.exit(1)

    # 2. Prepare Data
    labels = df['quality']
    cols_to_drop = ['quality', 'Id']
    features = df.drop(columns=cols_to_drop, errors='ignore')
    feature_names = features.columns.tolist()

    print(f"Features: {len(feature_names)}")

    # 3. Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    # 4. Apply Full PCA
    pca_full = PCA()
    pca_full.fit(scaled_data)
    
    # Get Eigenvalues and Variance info
    eigenvalues = pca_full.explained_variance_
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    components = np.arange(1, len(eigenvalues) + 1)
    
    output_dir = os.path.join(script_dir, 'assets')
    os.makedirs(output_dir, exist_ok=True)

    # --- PLOT 1: Scree Plot (Elbow Method) + Kaiser's Rule ---
    plt.figure(figsize=(10, 6))
    plt.plot(components, eigenvalues, 'bo-', linewidth=2, markersize=8, label='Eigenvalues')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label="Kaiser's Threshold (Eigenvalue = 1.0)")
    
    plt.title('Scree Plot & Kaiser\'s Rule', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component Number', fontsize=12)
    plt.ylabel('Eigenvalue (Explained Variance)', fontsize=12)
    plt.xticks(components)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    scree_path = os.path.join(output_dir, 'pca_scree_kaiser.png')
    plt.savefig(scree_path, dpi=300, bbox_inches='tight')
    print(f"Saved Scree Plot to: {scree_path}")
    plt.close()

    # --- PLOT 2: Cumulative Explained Variance ---
    plt.figure(figsize=(10, 6))
    plt.plot(components, cumulative_variance, 'g-s', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Threshold')
    
    # Find components needed for 95%
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    plt.axvline(x=n_components_95, color='orange', linestyle=':', label=f'95% at {n_components_95} Components')

    plt.title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Variance Explained', fontsize=12)
    plt.xticks(components)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    
    cum_path = os.path.join(output_dir, 'pca_cumulative.png')
    plt.savefig(cum_path, dpi=300, bbox_inches='tight')
    print(f"Saved Cumulative Plot to: {cum_path}")
    plt.close()

    # --- PLOT 3: 2D Projection (PC1 vs PC2) ---
    pca_2d = PCA(n_components=2)
    principal_components_2d = pca_2d.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(data=principal_components_2d, columns=['PC1', 'PC2'])
    pca_df['Wine Quality'] = labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='Wine Quality', 
        data=pca_df, 
        palette='viridis',
        alpha=0.7,
        s=80
    )
    
    plt.title('2D PCA Projection of WineQT', fontsize=15, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca_full.explained_variance_ratio_[0]:.1%} Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca_full.explained_variance_ratio_[1]:.1%} Variance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title='Wine Class', loc='best')
    
    proj_path = os.path.join(output_dir, 'pca_2d.png')
    plt.savefig(proj_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D Projection to: {proj_path}")
    plt.close()
    
    print("\nAll visualizations generated successfully.")

if __name__ == "__main__":
    main()
