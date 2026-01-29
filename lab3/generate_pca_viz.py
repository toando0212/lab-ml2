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

def plot_pca_analysis(X, dataset_name, n_components_95=None):
    """Generate PCA visualization: scree plot and cumulative variance"""
    # Normalize first
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA with all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Get variance explained
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    n_components = min(50, len(explained_variance_ratio))  # Show first 50 components
    ax1.bar(range(1, n_components + 1), explained_variance_ratio[:n_components], 
            alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variance Explained Ratio', fontsize=12, fontweight='bold')
    ax1.set_title(f'Scree Plot - {dataset_name}', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative variance plot
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='-', color='darkgreen', linewidth=2, markersize=4)
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Variance')
    if n_components_95:
        ax2.axvline(x=n_components_95, color='orange', linestyle='--', linewidth=2, 
                   label=f'{n_components_95} components')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cumulative Variance - {dataset_name}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dataset_name}_pca_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved PCA analysis plot to {OUTPUT_DIR}/{dataset_name}_pca_analysis.png")
    plt.close()
    
    # Print statistics
    print(f"\n{dataset_name} PCA Statistics:")
    print(f"Total features: {len(explained_variance_ratio)}")
    print(f"Components for 95% variance: {n_components_95 if n_components_95 else np.argmax(cumulative_variance >= 0.95) + 1}")
    print(f"Components for 99% variance: {np.argmax(cumulative_variance >= 0.99) + 1}")
    print(f"Top 5 components explain: {cumulative_variance[4]:.2%} variance")

# Mice Protein
print("=== Mice Protein Dataset ===")
X_mice, y_mice = load_mice_protein()
X_mice_train, X_mice_test, y_mice_train, y_mice_test = train_test_split(
    X_mice, y_mice, test_size=0.3, random_state=42)

# Normalize and get PCA components
scaler_mice = MinMaxScaler()
X_mice_scaled = scaler_mice.fit_transform(X_mice_train)
pca_mice = PCA(n_components=0.95)
pca_mice.fit(X_mice_scaled)
n_comp_mice = pca_mice.n_components_

plot_pca_analysis(X_mice_train, "Mice_Protein", n_comp_mice)

# Isolet
print("\n=== Isolet Dataset ===")
X_iso_train, y_iso_train, X_iso_test, y_iso_test = load_isolet()

# Normalize and get PCA components
scaler_iso = MinMaxScaler()
X_iso_scaled = scaler_iso.fit_transform(X_iso_train)
pca_iso = PCA(n_components=0.95)
pca_iso.fit(X_iso_scaled)
n_comp_iso = pca_iso.n_components_

plot_pca_analysis(X_iso_train, "Isolet", n_comp_iso)

print("\n✓ All PCA visualizations generated successfully!")
