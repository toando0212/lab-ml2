import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Cấu hình
MICE_PATH = "/Volumes/Toan/ML2/Dataset/Data_Cortex_Nuclear.csv"
ISOLET_TRAIN_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet1234.data"
ISOLET_TEST_PATH = "/Volumes/Toan/ML2/Dataset/archivelab2/isolet5.data"

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

def generate_k_table(X_train, X_test, y_train, y_test, dataset_name):
    """Generate table of accuracy/error for different k values"""
    ks = list(range(1, 21, 2))
    results = []
    
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        error = 1 - acc
        results.append({
            'k': k,
            'Accuracy': acc,
            'Error': error
        })
    
    df_results = pd.DataFrame(results)
    print(f"\n{dataset_name} Results:")
    print(df_results.to_string(index=False))
    
    # Manual LaTeX table generation
    print(f"\nLaTeX Table for {dataset_name}:")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("\\textbf{k} & \\textbf{Accuracy} & \\textbf{Error} \\\\")
    print("\\midrule")
    for _, row in df_results.iterrows():
        print(f"{int(row['k'])} & {row['Accuracy']:.4f} & {row['Error']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Classification performance for varying $k$ values on {dataset_name} dataset.}}")
    print(f"\\label{{tab:{dataset_name.lower().replace(' ', '_')}_k_variation}}")
    print("\\end{table}")
    
    return df_results

# Mice Protein
X_mice, y_mice = load_mice_protein()
X_mice_train, X_mice_test, y_mice_train, y_mice_test = train_test_split(
    X_mice, y_mice, test_size=0.3, random_state=42)
mice_results = generate_k_table(X_mice_train, X_mice_test, y_mice_train, y_mice_test, "Mice Protein")

# Isolet
X_iso_train, y_iso_train, X_iso_test, y_iso_test = load_isolet()
isolet_results = generate_k_table(X_iso_train, X_iso_test, y_iso_train, y_iso_test, "Isolet")
