import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.analysis import calculate_cluster_metrics

def main():
    # 1. Đường dẫn dữ liệu
    data_root = "/Volumes/Toan/ML2/data/Features/Variants"
    models = ["resnet50", "inception_v3"]
    
    results = []

    print("🚀 Bắt đầu đánh giá định lượng đặc trưng...")
    
    for model in models:
        feat_path = os.path.join(data_root, f"{model}/{model}_features.npy")
        lbl_path = os.path.join(data_root, f"{model}/{model}_labels.npy")
        
        print(f"⏳ Đang xử lý {model}...")
        features = np.load(feat_path)
        labels = np.load(lbl_path)
        
        # Tính toán metric (sử dụng mẫu 5000 để đảm bảo tốc độ)
        s_score, db_score, ari_score, nmi_score = calculate_cluster_metrics(features, labels, sample_size=5000)
        
        results.append({
            "Model": model,
            "Silhouette (↑)": s_score,
            "Davies-Bouldin (↓)": db_score,
            "ARI (↑)": ari_score,
            "NMI (↑)": nmi_score
        })

    # 2. Hiển thị báo cáo
    df = pd.DataFrame(results)
    print("\n📊 BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG (CLUSTERING QUALITY)")
    print("-" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    print("📌 Giải thích:")
    print("  - Silhouette Score: Càng gần 1 càng tốt (các cụm tách rời và đặc).")
    print("  - Davies-Bouldin: Càng nhỏ càng tốt (các cụm cách xa nhau và gọn).")

    # Lưu kết quả
    output_path = "/Volumes/Toan/ML2/reports/cluster_metrics_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Kết quả đã được lưu tại: {output_path}")

if __name__ == "__main__":
    main()
