import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis import run_umap

def main():
    # 1. Đường dẫn dữ liệu
    data_root = "/Volumes/Toan/ML2/data/Features/Variants"
    resnet_feat_path = os.path.join(data_root, "resnet50/resnet50_features.npy")
    resnet_lbl_path = os.path.join(data_root, "resnet50/resnet50_labels.npy")
    
    inception_feat_path = os.path.join(data_root, "inception_v3/inception_v3_features.npy")
    inception_lbl_path = os.path.join(data_root, "inception_v3/inception_v3_labels.npy")
    
    report_dir = "/Volumes/Toan/ML2/reports/figures"
    os.makedirs(report_dir, exist_ok=True)

    print("⏳ Đang nạp dữ liệu...")
    res_feat = np.load(resnet_feat_path)
    res_lbl = np.load(resnet_lbl_path)
    inc_feat = np.load(inception_feat_path)
    inc_lbl = np.load(inception_lbl_path)

    # 2. Chạy UMAP (Lấy mẫu nếu dữ liệu quá lớn, nhưng 14k vẫn ổn)
    print("⏳ Đang chạy UMAP cho ResNet50...")
    res_umap = run_umap(res_feat)
    
    print("⏳ Đang chạy UMAP for InceptionV3...")
    inc_umap = run_umap(inc_feat)

    # 3. Vẽ đồ thị Side-by-Side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ResNet Plot
    scatter1 = ax1.scatter(res_umap[:, 0], res_umap[:, 1], c=res_lbl, cmap='tab10', s=5, alpha=0.6)
    ax1.set_title("UMAP Visualization: ResNet50 Features", fontsize=15)
    plt.colorbar(scatter1, ax=ax1, label='Class ID')
    
    # Inception Plot
    scatter2 = ax2.scatter(inc_umap[:, 0], inc_umap[:, 1], c=inc_lbl, cmap='tab10', s=5, alpha=0.6)
    ax2.set_title("UMAP Visualization: InceptionV3 Features", fontsize=15)
    plt.colorbar(scatter2, ax=ax2, label='Class ID')

    plt.tight_layout()
    output_path = os.path.join(report_dir, "umap_side_by_side_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Đồ thị so sánh UMAP đã được lưu tại: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
