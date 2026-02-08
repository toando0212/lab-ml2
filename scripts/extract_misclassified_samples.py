import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_misclassified(model_name):
    print(f"\n🔍 Đang trích xuất các mẫu bị dự đoán sai cho mô hình: {model_name}...")
    
    # Đường dẫn
    folder_name = model_name.lower().replace('inceptionv3', 'inception_v3')
    data_root = f"/Volumes/Toan/ML2/data/Features/Variants/{folder_name}"
    feat_path = f"{data_root}/{folder_name}_features.npy"
    lbl_path = f"{data_root}/{folder_name}_labels.npy"
    paths_path = f"{data_root}/{folder_name}_paths.npy"
    
    model_dir = "/Volumes/Toan/ML2/models/rbf_svm"
    report_dir = "/Volumes/Toan/ML2/reports/performance"
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Nạp toàn bộ dữ liệu để tái lập việc chia tập
    X = np.load(feat_path)
    y = np.load(lbl_path)
    image_paths = np.load(paths_path)
    
    # 2. Tái lập việc chia tập dữ liệu (phải khớp hoàn toàn với train_rbf_svm.py)
    # Train/Test 80/20, random_state=42, stratify=y
    indices = np.arange(len(y))
    _, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    paths_test = image_paths[test_idx]
    
    # 3. Nạp mô hình và scaler
    svm_path = f"{model_dir}/{model_name.lower()}_rbf_svm.pkl"
    scaler_path = f"{model_dir}/{model_name.lower()}_scaler.pkl"
    
    if not os.path.exists(svm_path) or not os.path.exists(scaler_path):
        print(f"❌ Không tìm thấy mô hình hoặc scaler tại {model_dir}")
        return

    svm = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    
    # 4. Tiền xử lý và Dự đoán
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm.predict(X_test_scaled)
    
    # 5. Lọc các mẫu sai
    misclassified_mask = y_pred != y_test
    mis_paths = paths_test[misclassified_mask]
    mis_actual = y_test[misclassified_mask]
    mis_predicted = y_pred[misclassified_mask]
    
    # 6. Tạo DataFrame và lưu
    df_mis = pd.DataFrame({
        'image_path': mis_paths,
        'actual_class': mis_actual,
        'predicted_class': mis_predicted
    })
    
    output_path = f"{report_dir}/{model_name.lower()}_rbf_misclassified.csv"
    df_mis.to_csv(output_path, index=False)
    
    print(f"✅ Đã tìm thấy {len(df_mis)} mẫu bị dự đoán sai.")
    print(f"💾 Kết quả đã được lưu tại: {output_path}")

def main():
    models = ["ResNet50", "InceptionV3"]
    for m in models:
        try:
            extract_misclassified(m)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {m}: {e}")

if __name__ == "__main__":
    main()
