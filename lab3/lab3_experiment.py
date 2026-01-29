
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    # Xử lý missing values: fill mean hoặc drop. Ở đây chọn fill 0 cho đơn giản hoặc mean
    df = df.fillna(0) # Simple imputation approach for this lab
    
    # Xác định features và labels
    # Mice dataset thường có cột class ở cuối hoặc 'class', 'Genotype' etc.
    # Trong file Data_Cortex_Nuclear.csv, cột phân loại thường là 'class'
    
    # Kiểm tra các cột không phải số
    # Các cột đầu cuối thường là ID, Genotype, Treatment, Behavior, class
    # Ta sẽ lấy features là các cột protein (số) và target là 'class'
    
    features = df.select_dtypes(include=[np.number]).columns
    # Loại bỏ các cột không phải protein expression nếu có (ví dụ MouseID nếu nó được nhận diện là số - thường ko)
    # Nhưng tốt nhất drop các cột known metadata: MouseID, Genotype, Treatment, Behavior, class
    
    drop_cols = ['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Vì file có thể chứa cột object, ta convert X sang numeric error='coerce' cho chắc
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    y = df['class']
    print(f"Mice Shape: {X.shape}, Classes: {y.nunique()}")
    return X, y

def load_isolet():
    print("--- Loading Isolet Dataset ---")
    # Isolet data không có header, cột cuối là label
    df_train = pd.read_csv(ISOLET_TRAIN_PATH, header=None)
    df_test = pd.read_csv(ISOLET_TEST_PATH, header=None)
    
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    
    print(f"Isolet Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def run_knn(X_train, X_test, y_train, y_test, dataset_name="Dataset"):
    print(f"\n[{dataset_name}] Running KNN...")
    
    # 2. Basic KNN
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    error = 1 - acc
    print(f"Default k={k}: Accuracy={acc:.4f}, Error={error:.4f}")
    
    # 3. Vary k
    print(f"[{dataset_name}] Varying k...")
    ks = list(range(1, 21, 2))
    errors = []
    for k_i in ks:
        knn_i = KNeighborsClassifier(n_neighbors=k_i)
        knn_i.fit(X_train, y_train)
        score = knn_i.score(X_test, y_test)
        errors.append(1 - score)
    
    plt.figure()
    plt.plot(ks, errors, marker='o')
    plt.title(f'{dataset_name} - KNN Error vs K')
    plt.xlabel('k')
    plt.ylabel('Classification Error')
    plt.savefig(f"{OUTPUT_DIR}/{dataset_name}_vary_k.png")
    print(f"Saved plot to {OUTPUT_DIR}/{dataset_name}_vary_k.png")
    
    # 4. Normalization
    print(f"[{dataset_name}] Normalization...")
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    knn_norm = KNeighborsClassifier(n_neighbors=5)
    knn_norm.fit(X_train_norm, y_train)
    acc_norm = knn_norm.score(X_test_norm, y_test)
    print(f"Normalized (MinMax) k=5: Accuracy={acc_norm:.4f}, Error={1-acc_norm:.4f}")
    
    # 5. PCA & SVD
    print(f"[{dataset_name}] PCA & SVD...")
    # PCA
    pca = PCA(n_components=0.95) # Keep 95% variance
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    
    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    acc_pca = knn_pca.score(X_test_pca, y_test)
    print(f"PCA (95% var) k=5: Accuracy={acc_pca:.4f}, Error={1-acc_pca:.4f}, Components={pca.n_components_}")
    
    # 6. Improved Approach: CV (Just demo on Train set)
    print(f"[{dataset_name}] 5-Fold Cross Validation (on Train)...")
    knn_cv = KNeighborsClassifier(n_neighbors=5)
    cv_scores = cross_val_score(knn_cv, X_train_norm, y_train, cv=5)
    print(f"CV Scores: {cv_scores}, Mean: {cv_scores.mean():.4f}")
    
    # 7. Leave One Out (Only run on subset of Mice, skip Isolet full for speed if needed)
    # Isolet is large, LOO is very slow. Run on small sample for demo.
    if len(X_train) > 1000:
        print(f"[{dataset_name}] Data too large for full LOO. Running on subset of 200 samples...")
        X_sub = X_train_norm[:200]
        y_sub = y_train[:200]
    else:
        X_sub = X_train_norm
        y_sub = y_train
        
    loo = LeaveOneOut()
    knn_loo = KNeighborsClassifier(n_neighbors=5)
    # Using cross_val_score with LOO
    # Note: this might be slow, so we just run prediction manually or use cross_val_predict
    # For speed in this environment, I will skip strict LOO on full set if > 200 samples
    # Actually just run on the subset defined above
    loo_scores = cross_val_score(knn_loo, X_sub, y_sub, cv=loo)
    print(f"LOO Mean Accuracy (Subset): {loo_scores.mean():.4f}")

def run_svm(X_train, X_test, y_train, y_test, dataset_name="Dataset"):
    print(f"\n[{dataset_name}] Running SVM...")
    # Analyze distribution (Just check shape and linearity assumption by result)
    # Linearly separable? Hard to tell in high dim. We usually try Linear kernel vs RBF.
    
    # Normalize is crucial for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear SVM
    print("Fitting Linear SVM...")
    svm_lin = SVC(kernel='linear')
    svm_lin.fit(X_train_scaled, y_train)
    acc_lin = svm_lin.score(X_test_scaled, y_test)
    print(f"Linear SVM Accuracy: {acc_lin:.4f}")
    
    # RBF SVM
    print("Fitting RBF SVM...")
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train_scaled, y_train)
    acc_rbf = svm_rbf.score(X_test_scaled, y_test)
    print(f"RBF SVM Accuracy: {acc_rbf:.4f}")
    
    # Multi-class handling
    # sklearn SVC uses One-vs-One by default for multi-class. LinearSVC uses One-vs-Rest.
    # We can inspect decision_function_shape
    print(f"SVM handling multi-class strategy: {svm_rbf.decision_function_shape}")
    print("Classification Report (RBF):")
    print(classification_report(y_test, svm_rbf.predict(X_test_scaled)))

def main():
    # 1. Mice Protein
    # Cần split ra train/test vì file csv gộp chung
    X_mice, y_mice = load_mice_protein()
    X_mice_train, X_mice_test, y_mice_train, y_mice_test = train_test_split(X_mice, y_mice, test_size=0.3, random_state=42)
    
    run_knn(X_mice_train, X_mice_test, y_mice_train, y_mice_test, dataset_name="Mice_Protein")
    
    # Mice SVM
    run_svm(X_mice_train, X_mice_test, y_mice_train, y_mice_test, dataset_name="Mice_Protein")
    
    # 2. Isolet
    X_iso_train, y_iso_train, X_iso_test, y_iso_test = load_isolet()
    
    # Run KNN Isolet
    run_knn(X_iso_train, X_iso_test, y_iso_train, y_iso_test, dataset_name="Isolet")
    
    # Run SVM Isolet
    run_svm(X_iso_train, X_iso_test, y_iso_train, y_iso_test, dataset_name="Isolet")

if __name__ == "__main__":
    main()
