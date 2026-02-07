import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time

def train_rbf_svm(model_name):
    print(f"\n🚀 Đang huấn luyện RBF-SVM cho {model_name}...")
    
    # 1. Nạp dữ liệu
    data_root = f"/Volumes/Toan/ML2/data/Features/Variants/{model_name.lower().replace('v', '_v')}"
    feat_path = f"{data_root}/{model_name.lower().replace('v', '_v')}_features.npy"
    lbl_path = f"{data_root}/{model_name.lower().replace('v', '_v')}_labels.npy"
    
    X = np.load(feat_path)
    y = np.load(lbl_path)
    
    # 2. Chia tập dữ liệu (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # 4. Huấn luyện RBF-SVM
    start_time = time.time()
    print(f"   → Đang huấn luyện RBF-SVM (kernel='rbf', probability=True)...")
    
    svm = SVC(kernel='rbf', probability=True, verbose=True, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"✅ Hoàn thành huấn luyện {model_name} sử dụng RBF-SVM trong {train_time:.2f}s")
    
    # Lưu mô hình, scaler và cả tập test để code evaluate dùng
    model_dir = "/Volumes/Toan/ML2/models/rbf_svm"
    test_data_dir = "/Volumes/Toan/ML2/data/Features/TestSplit"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    
    joblib.dump(svm, f"{model_dir}/{model_name.lower()}_rbf_svm.pkl")
    joblib.dump(scaler, f"{model_dir}/{model_name.lower()}_scaler.pkl")
    
    # Lưu tập test
    np.save(f"{test_data_dir}/{model_name.lower()}_X_test_rbf.npy", X_test)
    np.save(f"{test_data_dir}/{model_name.lower()}_y_test_rbf.npy", y_test)
    
    print(f"💾 Đã lưu mô hình tại: {model_dir}/{model_name.lower()}_rbf_svm.pkl")

def main():
    models = ["ResNet50", "InceptionV3"]
    
    for m in models:
        try:
            train_rbf_svm(m)
        except Exception as e:
            print(f"❌ Lỗi khi huấn luyện {m}: {e}")

if __name__ == "__main__":
    main()
