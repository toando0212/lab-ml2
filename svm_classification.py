import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def run_svm_pipeline(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load dữ liệu
    print(f"Đang đọc dữ liệu từ: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X = df[feature_cols].values
    y = df['ClassId'].values
    
    # 2. Chuẩn hóa dữ liệu (BẮT BUỘC cho SVM)
    print("Đang chuẩn hóa dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Khởi tạo mô hình SVM
    # kernel='rbf' để xử lý phi tuyến, class_weight='balanced' để xử lý mất cân bằng lớp
    model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    
    # ---------------------------------------------------------
    # 4. K-Fold Cross Validation (Đánh giá chi tiết)
    # ---------------------------------------------------------
    print("\n--- Thực hiện Stratified 5-Fold Cross Validation (Full Metrics) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    from sklearn.model_selection import cross_validate
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(model, X_scaled, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.4f} (+/- {np.std(cv_results['test_accuracy']):.4f})")
    print(f"Precision (Macro): {np.mean(cv_results['test_precision_macro']):.4f}")
    print(f"Recall (Macro): {np.mean(cv_results['test_recall_macro']):.4f}")
    print(f"F1-score (Macro): {np.mean(cv_results['test_f1_macro']):.4f}")
    
    # ---------------------------------------------------------
    # 5. Huấn luyện chi tiết trên một lần Split (Để lấy Report & Confusion Matrix)
    # ---------------------------------------------------------
    print("\n--- Huấn luyện chi tiết trên tập Train/Test (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Báo cáo phân loại
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Lưu report vào file txt
    with open(os.path.join(output_dir, "svm_report.txt"), "w") as f:
        f.write("SVM CLASSIFICATION REPORT\n")
        f.write("=========================\n")
        f.write("1. K-FOLD CROSS VALIDATION (ON WHOLE DATASET):\n")
        f.write(f"   - Accuracy:  {np.mean(cv_results['test_accuracy']):.4f}\n")
        f.write(f"   - Precision: {np.mean(cv_results['test_precision_macro']):.4f}\n")
        f.write(f"   - Recall:    {np.mean(cv_results['test_recall_macro']):.4f}\n")
        f.write(f"   - F1-score:  {np.mean(cv_results['test_f1_macro']):.4f}\n\n")
        
        f.write("2. DETAILED METRICS (ON 20% TEST SET):\n")
        f.write(report)
    
    # Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('SVM Confusion Matrix (ResNet50 Features)')
    plt.savefig(os.path.join(output_dir, "svm_confusion_matrix.png"))
    print(f"Đã lưu ma trận nhầm lẫn tại: {os.path.join(output_dir, 'svm_confusion_matrix.png')}")
    
    # Lưu mô hình và scaler để sử dụng sau này (nếu cần)
    joblib.dump(model, os.path.join(output_dir, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "svm_scaler.pkl"))
    print("\nĐã lưu mô hình SVM và Scaler thành công.")

if __name__ == "__main__":
    CSV_PATH = "/Volumes/Toan/ML2/Features/resnet50_features.csv"
    OUTPUT_DIR = "/Volumes/Toan/ML2/Results_SVM"
    run_svm_pipeline(CSV_PATH, OUTPUT_DIR)
