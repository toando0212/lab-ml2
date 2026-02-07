import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time

def train_and_save(model_name, feat_path, lbl_path):
    print(f"\n🚀 Đang huấn luyện Linear SVM cho {model_name}...")
    
    # 1. Nạp dữ liệu
    X = np.load(feat_path)
    y = np.load(lbl_path)
    
    # 2. Chia tập dữ liệu (80/20) để lưu đúng scaler dùng cho test sau này
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # 4. Huấn luyện SVM sử dụng Stochastic Gradient Descent (SGD)
    # Đây là giải pháp "thần tốc" cho dữ liệu lớn.
    # SGDClassifier với loss='hinge' chính là một Linear SVM.
    from sklearn.linear_model import SGDClassifier
    
    start_time = time.time()
    print(f"   → Đang huấn luyện SGD-SVM (max_iter=1000)...")
    
    # Tắt verbose để tránh spam terminal
    svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, 
                        max_iter=1000, tol=1e-3, class_weight='balanced',
                        random_state=42, verbose=0, n_jobs=-1)
    
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"✅ Hoàn thành huấn luyện {model_name} sử dụng SGD trong {train_time:.2f}s")
    
    # Lưu mô hình, scaler và cả tập test để code evaluate dùng
    model_dir = "/Volumes/Toan/ML2/models/final_svm"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(svm, f"{model_dir}/{model_name.lower()}_linear_svm.pkl")
    joblib.dump(scaler, f"{model_dir}/{model_name.lower()}_scaler.pkl")
    
    # Lưu tập test để đảm bảo tính nhất quán khi đánh giá
    test_data_dir = "/Volumes/Toan/ML2/data/Features/TestSplit"
    os.makedirs(test_data_dir, exist_ok=True)
    np.save(f"{test_data_dir}/{model_name.lower()}_X_test.npy", X_test)
    np.save(f"{test_data_dir}/{model_name.lower()}_y_test.npy", y_test)
    
    return train_time

def main():
    variants = [
        ("ResNet50", "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_features.npy", 
                     "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_labels.npy"),
        ("InceptionV3", "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_features.npy", 
                       "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_labels.npy")
    ]
    
    for name, f_path, l_path in variants:
        train_and_save(name, f_path, l_path)

if __name__ == "__main__":
    main()
