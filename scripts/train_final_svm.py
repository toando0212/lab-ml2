import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

def train_and_evaluate(model_name, feat_path, lbl_path):
    print(f"\n🚀 Đang huấn luyện SVM cho {model_name}...")
    
    # 1. Nạp dữ liệu
    X = np.load(feat_path)
    y = np.load(lbl_path)
    
    # 2. Chia tập dữ liệu (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Chuẩn hóa dữ liệu (Quan trọng cho SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4. Huấn luyện SVM (Kernel RBF - Mặc định và mạnh mẽ)
    start_time = time.time()
    # Bật verbose=True để thấy log của LibSVM vì fit() không hỗ trợ tqdm trực tiếp
    svm = SVC(kernel='rbf', probability=True, verbose=True, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"✅ Hoàn thành huấn luyện {model_name} trong {train_time:.2f}s")
    
    # 5. Dự đoán và Đánh giá
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Lưu mô hình và scaler
    model_dir = "/Volumes/Toan/ML2/models/final_svm"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(svm, f"{model_dir}/{model_name.lower()}_svm.pkl")
    joblib.dump(scaler, f"{model_dir}/{model_name.lower()}_scaler.pkl")
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "F1-Score": f1,
        "Train Time (s)": train_time
    }

def main():
    variants = [
        ("ResNet50", "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_features.npy", 
                     "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_labels.npy"),
        ("InceptionV3", "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_features.npy", 
                       "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_labels.npy")
    ]
    
    summary_results = []
    
    for name, f_path, l_path in variants:
        res = train_and_evaluate(name, f_path, l_path)
        summary_results.append(res)
    
    # Xuất bảng tổng hợp
    df = pd.DataFrame(summary_results)
    print("\n📊 KẾT QUẢ ĐỐI CHỨNG CUỐI CÙNG (SVM ON 2048D)")
    print("-" * 65)
    print(df.to_string(index=False))
    print("-" * 65)
    
    df.to_csv("/Volumes/Toan/ML2/reports/final_performance_comparison.csv", index=False)
    print("\n✅ Đã lưu báo cáo tại: /Volumes/Toan/ML2/reports/final_performance_comparison.csv")

if __name__ == "__main__":
    main()
