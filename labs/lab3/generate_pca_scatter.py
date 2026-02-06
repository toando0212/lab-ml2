import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import os

# Cấu hình
MICE_PATH = "/Volumes/Toan/ML2/Dataset/Data_Cortex_Nuclear.csv"
ISOLET_TRAIN_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet1234.data"
ISOLET_TEST_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet5.data"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mice_protein():
    df = pd.read_csv(MICE_PATH)
    df = df.fillna(0)
    drop_cols = ['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['class']
    return X, y

def load_isolet():
    df_train = pd.read_csv(ISOLET_TRAIN_PATH, header=None)
    df_test = pd.read_csv(ISOLET_TEST_PATH, header=None)
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    return X_train, y_train, X_test, y_test

def plot_pca_2d(X, y, dataset_name, n_components_95):
    """Generate 2D scatter plot of first 2 principal components"""
    # Normalize first
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA with 2 components for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Get variance explained by first 2 components
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Get unique classes and create color map
    unique_classes = sorted(y.unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    
    # Plot each class
    for idx, cls in enumerate(unique_classes):
        mask = y == cls
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[idx]], label=f'Class {cls}', 
                   alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
    
    plt.xlabel(f'PC1 ({var_pc1:.1f}% variance)', fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({var_pc2:.1f}% variance)', fontsize=12, fontweight='bold')
    plt.title(f'PCA Visualization - {dataset_name}\n({n_components_95} components needed for 95% variance)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1 if len(unique_classes) <= 10 else 2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dataset_name}_pca_scatter.png", dpi=150, bbox_inches='tight')
    print(f"Saved PCA scatter plot to {OUTPUT_DIR}/{dataset_name}_pca_scatter.png")
    plt.close()
    
    print(f"\n{dataset_name} PCA 2D Projection:")
    print(f"PC1 explains: {var_pc1:.2f}% variance")
    print(f"PC2 explains: {var_pc2:.2f}% variance")
    print(f"Total (PC1+PC2): {var_pc1 + var_pc2:.2f}% variance")
    print(f"Number of classes: {len(unique_classes)}")

# Mice Protein
print("=== Mice Protein Dataset ===")
X_mice, y_mice = load_mice_protein()
X_mice_train, X_mice_test, y_mice_train, y_mice_test = train_test_split(
    X_mice, y_mice, test_size=0.3, random_state=42)

# Get number of components for 95% variance
scaler_mice = MinMaxScaler()
X_mice_scaled = scaler_mice.fit_transform(X_mice_train)
pca_mice_full = PCA(n_components=0.95)
pca_mice_full.fit(X_mice_scaled)
n_comp_mice = pca_mice_full.n_components_

plot_pca_2d(X_mice_train, y_mice_train, "Mice_Protein", n_comp_mice)

# Isolet
print("\n=== Isolet Dataset ===")
X_iso_train, y_iso_train, X_iso_test, y_iso_test = load_isolet()

# Get number of components for 95% variance
scaler_iso = MinMaxScaler()
X_iso_scaled = scaler_iso.fit_transform(X_iso_train)
pca_iso_full = PCA(n_components=0.95)
pca_iso_full.fit(X_iso_scaled)
n_comp_iso = pca_iso_full.n_components_

plot_pca_2d(X_iso_train, y_iso_train, "Isolet", n_comp_iso)

print("\n✓ All PCA scatter plots generated successfully!")
