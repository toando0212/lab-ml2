import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_name):
    print(f"\n📊 Đang đánh giá mô hình {model_name}...")
    
    model_dir = "/Volumes/Toan/ML2/models/final_svm"
    test_data_dir = "/Volumes/Toan/ML2/data/Features/TestSplit"
    report_dir = "/Volumes/Toan/ML2/reports/performance"
    os.makedirs(report_dir, exist_ok=True)

    # 1. Nạp mô hình và dữ liệu test
    svm = joblib.load(f"{model_dir}/{model_name.lower()}_linear_svm.pkl")
    scaler = joblib.load(f"{model_dir}/{model_name.lower()}_scaler.pkl")
    X_test = np.load(f"{test_data_dir}/{model_name.lower()}_X_test.npy")
    y_test = np.load(f"{test_data_dir}/{model_name.lower()}_y_test.npy")

    # 2. Tiền xử lý tập test
    X_test_scaled = scaler.transform(X_test)

    # 3. Dự đoán
    y_pred = svm.predict(X_test_scaled)

    # 4. Tính toán Metric
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    # 5. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name} + Linear SVM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    cm_path = f"{report_dir}/{model_name.lower()}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"💾 Đã lưu Confusion Matrix tại: {cm_path}")

    return {
        "Model": model_name,
        "Accuracy": acc
    }

def main():
    models = ["ResNet50", "InceptionV3"]
    results = []
    
    for m in models:
        try:
            res = evaluate_model(m)
            results.append(res)
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {m}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv("/Volumes/Toan/ML2/reports/final_comparison_summary.csv", index=False)
        print("\n✅ Đã lưu bảng tổng hợp tại: /Volumes/Toan/ML2/reports/final_comparison_summary.csv")

if __name__ == "__main__":
    main()
