import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_rbf_model(model_name):
    print(f"\n📊 Đang đánh giá mô hình RBF-SVM {model_name}...")
    
    model_dir = "/Volumes/Toan/ML2/models/rbf_svm"
    test_data_dir = "/Volumes/Toan/ML2/data/Features/TestSplit"
    report_dir = "/Volumes/Toan/ML2/reports/performance"
    os.makedirs(report_dir, exist_ok=True)

    # 1. Nạp mô hình và dữ liệu test
    svm = joblib.load(f"{model_dir}/{model_name.lower()}_rbf_svm.pkl")
    scaler = joblib.load(f"{model_dir}/{model_name.lower()}_scaler.pkl")
    X_test = np.load(f"{test_data_dir}/{model_name.lower()}_X_test_rbf.npy")
    y_test = np.load(f"{test_data_dir}/{model_name.lower()}_y_test_rbf.npy")

    # 2. Tiền xử lý tập test
    X_test_scaled = scaler.transform(X_test)

    # 3. Dự đoán
    y_pred = svm.predict(X_test_scaled)

    # 4. Tính toán Metric
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    
    # Lấy classification report dạng dict để trích xuất metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # In ra terminal
    report_str = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report_str)

    # 5. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f"Confusion Matrix: {model_name} + RBF-SVM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    cm_path = f"{report_dir}/{model_name.lower()}_rbf_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"💾 Đã lưu Confusion Matrix tại: {cm_path}")

    return {
        "model_name": model_name,
        "report_dict": report_dict,
        "accuracy": acc
    }

def main():
    models = ["ResNet50", "InceptionV3"]
    all_results = []
    
    for m in models:
        try:
            res = evaluate_rbf_model(m)
            model_name = res["model_name"]
            report_dict = res["report_dict"]
            
            # Tạo bảng tổng hợp cho từng lớp
            for class_id, metrics in report_dict.items():
                if class_id in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                
                all_results.append({
                    "Model": model_name,
                    "SVM_Type": "RBF",
                    "Class": class_id,
                    "Precision": metrics['precision'],
                    "Recall": metrics['recall'],
                    "F1-Score": metrics['f1-score'],
                    "Support": int(metrics['support'])
                })
            
            # Thêm các dòng tổng hợp
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report_dict:
                    all_results.append({
                        "Model": model_name,
                        "SVM_Type": "RBF",
                        "Class": avg_type,
                        "Precision": report_dict[avg_type]['precision'],
                        "Recall": report_dict[avg_type]['recall'],
                        "F1-Score": report_dict[avg_type]['f1-score'],
                        "Support": int(report_dict[avg_type]['support'])
                    })
            
            # Thêm dòng Accuracy
            all_results.append({
                "Model": model_name,
                "SVM_Type": "RBF",
                "Class": "accuracy",
                "Precision": res["accuracy"],
                "Recall": res["accuracy"],
                "F1-Score": res["accuracy"],
                "Support": int(report_dict['macro avg']['support'])
            })
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {m}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = "/Volumes/Toan/ML2/reports/rbf_svm_comparison_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Đã lưu bảng tổng hợp RBF-SVM chi tiết tại: {csv_path}")
        print(f"   📋 Tổng số dòng: {len(df)}")

if __name__ == "__main__":
    main()
