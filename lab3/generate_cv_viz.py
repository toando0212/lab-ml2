import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
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

def plot_cv_results(mice_scores, isolet_scores):
    """Generate box plot for K-Fold CV results"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data
    data = [mice_scores, isolet_scores]
    labels = ['Mice Protein', 'Isolet']
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none'))
    
    # Add individual points
    for i, scores in enumerate(data, 1):
        x = np.random.normal(i, 0.04, size=len(scores))
        ax.scatter(x, scores, alpha=0.6, s=80, color='darkblue', edgecolors='black', linewidth=0.5, zorder=3)
    
    # Add mean markers
    means = [np.mean(scores) for scores in data]
    ax.scatter([1, 2], means, marker='D', s=150, color='green', edgecolors='black', 
               linewidth=1.5, zorder=4, label='Mean')
    
    # Formatting
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('5-Fold Cross-Validation Results (K-NN, k=5)', fontsize=14, fontweight='bold')
    ax.set_ylim([0.95, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # Add statistics text
    for i, (scores, label) in enumerate(zip(data, labels), 1):
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        ax.text(i, 0.951, f'μ={mean_val:.4f}\nσ={std_val:.4f}', 
               ha='center', va='bottom', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kfold_cv_boxplot.png", dpi=150, bbox_inches='tight')
    print(f"Saved K-Fold CV box plot to {OUTPUT_DIR}/kfold_cv_boxplot.png")
    plt.close()

# Mice Protein
print("=== Mice Protein 5-Fold CV ===")
X_mice, y_mice = load_mice_protein()
X_mice_train, X_mice_test, y_mice_train, y_mice_test = train_test_split(
    X_mice, y_mice, test_size=0.3, random_state=42)

# Normalize
scaler_mice = MinMaxScaler()
X_mice_train_norm = scaler_mice.fit_transform(X_mice_train)

# 5-Fold CV
knn_mice = KNeighborsClassifier(n_neighbors=5)
mice_cv_scores = cross_val_score(knn_mice, X_mice_train_norm, y_mice_train, cv=5)

print(f"Scores: {mice_cv_scores}")
print(f"Mean: {mice_cv_scores.mean():.4f}")
print(f"Std: {mice_cv_scores.std():.4f}")
print(f"Min: {mice_cv_scores.min():.4f}")
print(f"Max: {mice_cv_scores.max():.4f}")

# Isolet
print("\n=== Isolet 5-Fold CV ===")
X_iso_train, y_iso_train, X_iso_test, y_iso_test = load_isolet()

# Normalize
scaler_iso = MinMaxScaler()
X_iso_train_norm = scaler_iso.fit_transform(X_iso_train)

# 5-Fold CV
knn_iso = KNeighborsClassifier(n_neighbors=5)
isolet_cv_scores = cross_val_score(knn_iso, X_iso_train_norm, y_iso_train, cv=5)

print(f"Scores: {isolet_cv_scores}")
print(f"Mean: {isolet_cv_scores.mean():.4f}")
print(f"Std: {isolet_cv_scores.std():.4f}")
print(f"Min: {isolet_cv_scores.min():.4f}")
print(f"Max: {isolet_cv_scores.max():.4f}")

# Generate visualization
plot_cv_results(mice_cv_scores, isolet_cv_scores)

print("\n✓ K-Fold CV visualization generated successfully!")
