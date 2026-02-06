
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import os

# Configuration
MICE_PATH = "/Volumes/Toan/ML2/Dataset/Data_Cortex_Nuclear.csv"
ISOLET_TRAIN_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet1234.data"
ISOLET_TEST_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet5.data"
OUTPUT_DIR = "/Volumes/Toan/ML2/lab4/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mice_protein():
    print("--- Loading Mice Protein Dataset ---")
    df = pd.read_csv(MICE_PATH)
    # Simple imputation
    df = df.fillna(0)
    
    # Drop metadata cols
    drop_cols = ['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['class']
    
    print(f"Mice Shape: {X.shape}, Classes: {y.nunique()}")
    return X, y

def load_isolet():
    print("--- Loading Isolet Dataset ---")
    # Load and combine to split 80/20 manually as per instruction
    df_train = pd.read_csv(ISOLET_TRAIN_PATH, header=None)
    df_test = pd.read_csv(ISOLET_TEST_PATH, header=None)
    
    df_all = pd.concat([df_train, df_test], axis=0)
    X = df_all.iloc[:, :-1]
    y = df_all.iloc[:, -1]
    
    print(f"Isolet Total Shape: {X.shape}, Classes: {y.nunique()}")
    return X, y

def plot_confusion_matrix_custom(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_pca(X, y, title, filename):
    # Standardize first usually helps PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', legend='full', alpha=0.7)
    plt.title(title)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_feature_importance(model, feature_names, title, filename, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    
    # Check if feature_names is list-like or index
    if hasattr(feature_names, 'shape'): # numpy or pandas index
        names = [feature_names[i] for i in indices]
    else: # list
        names = [feature_names[i] for i in indices]
        
    plt.yticks(range(len(indices)), names)
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis() # Highest importance on top
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_class_distribution(y, title, filename):
    plt.figure(figsize=(10, 6))
    # Check if y is numeric to sort; otherwise keep as is
    unique_labels = np.unique(y)
    
    sns.countplot(x=y, order=unique_labels, palette='viridis')
    plt.title(title)
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    if len(unique_labels) > 10: # Rotate labels if many classes
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_tree_structure(model, feature_names, class_names, title, filename):
    plt.figure(figsize=(20, 10))
    # Limit max_depth for visualization readability
    plot_tree(model, max_depth=3, feature_names=feature_names, 
              class_names=[str(c) for c in class_names], 
              filled=True, rounded=True, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_rf_convergence(X_train, y_train, X_test, y_test, title, filename):
    # Using warm_start to efficiently calculate error vs n_estimators
    # n_jobs=-1 for speed
    rf = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42, n_jobs=-1)
    
    errors = []
    # Check every 5 trees to speed up plot generation, make sure 100 is included
    n_trees_range = list(range(1, 101, 5)) 
    if 100 not in n_trees_range: n_trees_range.append(100)
    
    for n in n_trees_range:
        rf.n_estimators = n
        rf.fit(X_train, y_train)
        acc = rf.score(X_test, y_test)
        errors.append(1 - acc)
        
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees_range, errors, marker='o', linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('Classification Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def run_experiment(X, y, dataset_name):
    print(f"\n{'='*20} Processing {dataset_name} {'='*20}")
    
    # 1. Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    results_md = f"\n### {dataset_name}\n\n"
    results_md += f"- **Dataset**: {dataset_name}\n"
    results_md += f"- **Train/Test Split**: 80/20\n"
    results_md += f"- **Train Size**: {X_train.shape[0]}, **Test Size**: {X_test.shape[0]}\n\n"
    
    # --- Visualization: Class Distribution ---
    print(f"[{dataset_name}] Generating Class Distribution plot...")
    plot_class_distribution(y, f'{dataset_name} - Class Distribution', f'{dataset_name}_dist.png')
    
    # --- Visualization: PCA ---
    print(f"[{dataset_name}] Generating PCA plot...")
    plot_pca(X, y, f'{dataset_name} - PCA Projection (2D)', f'{dataset_name}_pca.png')
    
    results_md += f"#### Data Analysis\n"
    results_md += f"| Class Distribution | PCA Projection |\n"
    results_md += f"| :---: | :---: |\n"
    results_md += f"| ![Dist](results/{dataset_name}_dist.png) | ![PCA](results/{dataset_name}_pca.png) |\n\n"
    
    # Get feature/class names
    if hasattr(X, 'columns'):
        feat_names = list(X.columns)
    else:
        feat_names = [f'Feat_{i}' for i in range(X.shape[1])]
    
    class_names = np.unique(y)
    
    # --- Decision Tree ---
    print(f"[{dataset_name}] Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    acc_dt = accuracy_score(y_test, y_pred_dt)
    err_dt = 1 - acc_dt
    prec_dt, rec_dt, f1_dt, _ = precision_recall_fscore_support(y_test, y_pred_dt, average='weighted')
    
    print(f"DT Accuracy: {acc_dt:.4f}, Error: {err_dt:.4f}")
    
    plot_confusion_matrix_custom(y_test, y_pred_dt, 
                                 f'{dataset_name} - Decision Tree Confusion Matrix', 
                                 f'{dataset_name}_dt_cm.png')
    
    # Plot Tree Structure
    print(f"[{dataset_name}] Generating Tree Structure plot...")
    plot_tree_structure(dt, feat_names, class_names, 
                        f'{dataset_name} - Decision Tree Structure (Depth=3)', 
                        f'{dataset_name}_dt_struct.png')
    
    results_md += "#### 1. Decision Tree (DT)\n"
    results_md += f"- **Accuracy**: {acc_dt:.4f}\n"
    results_md += f"- **Classification Error**: {err_dt:.4f}\n"
    results_md += f"- **Weighted Precision**: {prec_dt:.4f}\n"
    results_md += f"- **Weighted Recall**: {rec_dt:.4f}\n"
    results_md += f"- **Weighted F1-Score**: {f1_dt:.4f}\n"
    results_md += f"![DT Structure](results/{dataset_name}_dt_struct.png)\n"
    results_md += f"*Figure: First few layers of the Decision Tree.*\n\n"
    results_md += f"![DT Confusion Matrix](results/{dataset_name}_dt_cm.png)\n\n"
    
    # --- Random Forest ---
    print(f"[{dataset_name}] Training Random Forest (K=100)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    err_rf = 1 - acc_rf
    prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')

    print(f"RF Accuracy: {acc_rf:.4f}, Error: {err_rf:.4f}")
    
    plot_confusion_matrix_custom(y_test, y_pred_rf, 
                                 f'{dataset_name} - Random Forest Confusion Matrix', 
                                 f'{dataset_name}_rf_cm.png')
                                 
    # Convergence Plot
    print(f"[{dataset_name}] Generating RF Convergence plot...")
    plot_rf_convergence(X_train, y_train, X_test, y_test, 
                        f'{dataset_name} - RF Error Rate vs Trees', 
                        f'{dataset_name}_rf_conv.png')
    
    results_md += "#### 2. Random Forest (RF)\n"
    results_md += f"- **K (Trees)**: 100\n"
    results_md += f"- **Accuracy**: {acc_rf:.4f}\n"
    results_md += f"- **Classification Error**: {err_rf:.4f}\n"
    results_md += f"- **Weighted Precision**: {prec_rf:.4f}\n"
    results_md += f"- **Weighted Recall**: {rec_rf:.4f}\n"
    results_md += f"- **Weighted F1-Score**: {f1_rf:.4f}\n"
    results_md += f"![RF Convergence](results/{dataset_name}_rf_conv.png)\n"
    results_md += f"*Figure: Error rate decreases as number of trees increases.*\n\n"
    results_md += f"![RF Confusion Matrix](results/{dataset_name}_rf_cm.png)\n\n"
    
    # --- Feature Importance ---
    print(f"[{dataset_name}] Generating Feature Importance plot...")
    # Get feature names
    # Reuse feat_names from above
        
    plot_feature_importance(rf, feat_names, 
                            f'{dataset_name} - Top 20 Feature Importance (RF)', 
                            f'{dataset_name}_rf_features.png')
    results_md += f"#### Top Features (Random Forest)\n"
    results_md += f"![Feature Importance](results/{dataset_name}_rf_features.png)\n\n"
    
    # --- Comparison ---
    improvement = acc_rf - acc_dt
    results_md += "#### Comparison\n"
    results_md += f"- RF Improvement over DT: {improvement * 100:.2f}%\n"
    if improvement > 0:
        results_md += "- Random Forest outperformed Decision Tree as expected, reducing the variance and overfitting often seen in single decision trees.\n"
    else:
        results_md += "- Random Forest did not significantly outperform Decision Tree in this specific run.\n"
        
    return results_md

def main():
    md_output = "# Lab 4: Decision Tree & Random Forest Results\n\n"
    
    # Mice Protein
    X_mice, y_mice = load_mice_protein()
    md_output += run_experiment(X_mice, y_mice, "Mice_Protein")
    
    # Isolet
    X_iso, y_iso = load_isolet()
    md_output += run_experiment(X_iso, y_iso, "Isolet")
    
    # Write to MD file
    with open("/Volumes/Toan/ML2/lab4/lab4_results.md", "w") as f:
        f.write(md_output)
    print("\nResults saved to /Volumes/Toan/ML2/lab4/lab4_results.md")

if __name__ == "__main__":
    main()
