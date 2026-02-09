import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import json

# Thêm path gốc vào hệ thống
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_confusion_matrix(y_true, y_pred, model_name, backbone, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} on {backbone}')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_experiment(backbone_name, feat_path, lbl_path):
    print(f"\n--- 🧪 Khởi động thực nghiệm trên {backbone_name} ---")
    
    # 1. Nạp dữ liệu
    X = np.load(feat_path)
    y = np.load(lbl_path)
    
    # 2. Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Định nghĩa mô hình
    models = {
        "RFC": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SGDC": SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, random_state=42, n_jobs=-1)
    }
    
    results = []
    output_base_dir = f"/Volumes/Toan/ML2/reports/experiments/{backbone_name.lower()}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    for m_name, model in models.items():
        print(f"  > Đang huấn luyện {m_name}...")
        start_time = time.time()
        
        # RFC không nhất thiết cần scaled data nhưng SGDC thì có
        # Để công bằng ta dùng scaled cho cả hai hoặc tùy chọn
        if m_name == "RFC":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
        elapsed = time.time() - start_time
        acc = accuracy_score(y_test, y_pred)
        
        print(f"    ✓ Hoàn thành trong {elapsed:.2f}s | Accuracy: {acc:.4f}")
        
        # Lưu classification report dạng dict
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"{output_base_dir}/{m_name.lower()}_report.csv")
        
        # Lưu Confusion Matrix
        cm_path = f"{output_base_dir}/{m_name.lower()}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, m_name, backbone_name, cm_path)
        
        results.append({
            "Backbone": backbone_name,
            "Model": m_name,
            "Accuracy": acc,
            "Time": elapsed
        })
        
    return results

def main():
    experiments = [
        ("ResNet50", "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_features.npy", 
                     "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_labels.npy"),
        ("InceptionV3", "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_features.npy", 
                       "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_labels.npy")
    ]
    
    all_summaries = []
    for name, f_path, l_path in experiments:
        if os.path.exists(f_path):
            summary = run_experiment(name, f_path, l_path)
            all_summaries.extend(summary)
        else:
            print(f"⚠️ Không tìm thấy tệp đặc trưng cho {name} tại {f_path}")
            
    # Lưu tổng hợp kết quả
    summary_df = pd.DataFrame(all_summaries)
    print("\n📊 TỔNG HỢP KẾT QUẢ THỰC NGHIỆM BASLINE")
    print(summary_df.to_string(index=False))
    
    summary_path = "/Volumes/Toan/ML2/reports/experiments/baseline_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Đã lưu kết quả tại /Volumes/Toan/ML2/reports/experiments/")

if __name__ == "__main__":
    main()
