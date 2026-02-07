import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import plotly.express as px
import pandas as pd
from src.analysis import run_pca, run_umap, calculate_cluster_metrics

def main():
    variants = [
        ("ResNet50", "/Volumes/Toan/ML2/data/Features/Variants/resnet50"),
        ("InceptionV3", "/Volumes/Toan/ML2/data/Features/Variants/inception_v3")
    ]
    
    report_dir = "/Volumes/Toan/ML2/reports/interactive"
    os.makedirs(report_dir, exist_ok=True)
    all_results = []

    for model_name, data_dir in variants:
        print(f"\n⏳ Đang xử lý đặc trưng {model_name}...")
        feat_path = os.path.join(data_dir, f"{model_name.lower().replace('v3', '_v3') if 'Inception' in model_name else model_name.lower()}_features.npy")
        lbl_path = os.path.join(data_dir, f"{model_name.lower().replace('v3', '_v3') if 'Inception' in model_name else model_name.lower()}_labels.npy")
        
        features = np.load(feat_path)
        labels = np.load(lbl_path).astype(str)

        # 1. Chạy PCA 3D
        pca_3d = run_pca(features, n_components=3)
        s_pca, db_pca = calculate_cluster_metrics(pca_3d, labels)

        # 2. Chạy UMAP 3D
        umap_3d = run_umap(features, n_components=3)
        s_umap, db_umap = calculate_cluster_metrics(umap_3d, labels)

        # Lưu kết quả
        all_results.append({"Model": model_name, "Method": "PCA (3D)", "Silhouette (↑)": s_pca, "DB Index (↓)": db_pca})
        all_results.append({"Model": model_name, "Method": "UMAP (3D)", "Silhouette (↑)": s_umap, "DB Index (↓)": db_umap})

        # 3. Xuất HTML (Chỉ lấy UMAP 3D để giữ báo cáo gọn nhẹ, hoặc cả 2 nếu cần)
        for method_name, embedding in [("PCA", pca_3d), ("UMAP", umap_3d)]:
            fig = px.scatter_3d(
                x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
                color=labels,
                title=f"{method_name} 3D: {model_name} Features",
                opacity=0.7
            )
            fig.update_traces(marker=dict(size=3))
            output_file = os.path.join(report_dir, f"compare_3d_{model_name.lower()}_{method_name.lower()}.html")
            fig.write_html(output_file)

    # 4. In bảng so sánh tổng hợp
    df = pd.DataFrame(all_results)
    print("\n📊 BẢNG SO SÁNH TỔNG HỢP: PCA vs UMAP (3D Projection)")
    print("-" * 75)
    print(df.to_string(index=False))
    print("-" * 75)
    
    df.to_csv("/Volumes/Toan/ML2/reports/pca_vs_umap_full_3d_metrics.csv", index=False)

if __name__ == "__main__":
    main()
