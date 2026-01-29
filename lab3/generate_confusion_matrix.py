import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import os

# Cấu hình
MICE_PATH = "/Volumes/Toan/ML2/Dataset/Data_Cortex_Nuclear.csv"
ISOLET_TRAIN_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet1234.data"
ISOLET_TEST_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet5.data"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mice_protein():
    print("--- Loading Mice Protein Dataset ---")
    df = pd.read_csv(MICE_PATH)
    df = df.fillna(0)
    drop_cols = ['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['class']
    print(f"Mice Shape: {X.shape}, Classes: {y.nunique()}")
    return X, y

def load_isolet():
    print("--- Loading Isolet Dataset ---")
    df_train = pd.read_csv(ISOLET_TRAIN_PATH, header=None)
    df_test = pd.read_csv(ISOLET_TEST_PATH, header=None)
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    print(f"Isolet Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, dataset_name, class_labels=None):
    """Generate and save confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    # For datasets with many classes (like Isolet), use smaller figure
    if len(np.unique(y_true)) > 15:
        figsize = (12, 10)
        fontsize = 8
    else:
        figsize = (10, 8)
        fontsize = 10
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels if class_labels else 'auto',
                yticklabels=class_labels if class_labels else 'auto',
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': fontsize})
    plt.title(f'Confusion Matrix - {dataset_name} (k=5)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dataset_name}_confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {OUTPUT_DIR}/{dataset_name}_confusion_matrix.png")
    plt.close()

def main():
    # 1. Mice Protein
    X_mice, y_mice = load_mice_protein()
    X_mice_train, X_mice_test, y_mice_train, y_mice_test = train_test_split(
        X_mice, y_mice, test_size=0.3, random_state=42)
    
    print("\n[Mice_Protein] Running KNN for Confusion Matrix...")
    knn_mice = KNeighborsClassifier(n_neighbors=5)
    knn_mice.fit(X_mice_train, y_mice_train)
    y_mice_pred = knn_mice.predict(X_mice_test)
    
    # Get unique class labels for Mice
    mice_classes = sorted(y_mice.unique())
    plot_confusion_matrix(y_mice_test, y_mice_pred, "Mice_Protein", class_labels=mice_classes)
    
    # 2. Isolet
    X_iso_train, y_iso_train, X_iso_test, y_iso_test = load_isolet()
    
    print("\n[Isolet] Running KNN for Confusion Matrix...")
    knn_iso = KNeighborsClassifier(n_neighbors=5)
    knn_iso.fit(X_iso_train, y_iso_train)
    y_iso_pred = knn_iso.predict(X_iso_test)
    
    # For Isolet, labels are 1-26 (letters A-Z)
    plot_confusion_matrix(y_iso_test, y_iso_pred, "Isolet")
    
    print("\n✓ All confusion matrices generated successfully!")

if __name__ == "__main__":
    main()
