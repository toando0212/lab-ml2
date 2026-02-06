import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.analysis import get_pca_variance_ratio, plot_variance_comparison

def main():
    # 1. Đường dẫn dữ liệu
    data_root = "/Volumes/Toan/ML2/data/Features/Variants"
    resnet_path = os.path.join(data_root, "resnet50/resnet50_features.npy")
    inception_path = os.path.join(data_root, "inception_v3/inception_v3_features.npy")
    
    # Tạo thư mục báo cáo nếu chưa có
    report_dir = "/Volumes/Toan/ML2/reports/figures"
    os.makedirs(report_dir, exist_ok=True)

    print("⏳ Đang nạp dữ liệu đặc trưng...")
    res_feat = np.load(resnet_path)
    inc_feat = np.load(inception_path)
    
    print(f"✅ ResNet50: {res_feat.shape}")
    print(f"✅ InceptionV3: {inc_feat.shape}")

    # 2. Tính toán phương sai (Lấy 200 components đầu tiên)
    print("⏳ Đang tính toán PCA Variance...")
    res_ind, res_cum = get_pca_variance_ratio(res_feat, n_components=200)
    inc_ind, inc_cum = get_pca_variance_ratio(inc_feat, n_components=200)
    
    # 3. Tìm điểm 90%
    def find_90_idx(var_array):
        idx = np.where(var_array >= 0.9)[0]
        return idx[0] + 1 if len(idx) > 0 else f"> {len(var_array)}"

    res_90 = find_90_idx(res_cum)
    inc_90 = find_90_idx(inc_cum)
    
    # In thông số PC1-PC10
    print("-" * 30)
    print(f"📌 PC1 Variance: ResNet={res_ind[0]*100:.2f}%, Inception={inc_ind[0]*100:.2f}%")
    print(f"📌 PC1-PC10 Total: ResNet={res_cum[9]*100:.2f}%, Inception={inc_cum[9]*100:.2f}%")
    print(f"📌 Ngưỡng 90%: ResNet cần {res_90} comp, Inception cần {inc_90} comp")
    print("-" * 30)

    # 4. Vẽ biểu đồ và Lưu CSV để trace
    variance_dict = {
        f"ResNet50 (90% @ {res_90})": res_cum,
        f"InceptionV3 (90% @ {inc_90})": inc_cum
    }
    
    # Tạo DataFrame để lưu CSV đầy đủ
    df_trace = pd.DataFrame({
        'n_components': range(1, len(res_cum) + 1),
        'resnet50_individual': res_ind,
        'resnet50_cumulative': res_cum,
        'inceptionv3_individual': inc_ind,
        'inceptionv3_cumulative': inc_cum
    })
    trace_path = os.path.join(report_dir, "pca_variance_trace.csv")
    df_trace.to_csv(trace_path, index=False)
    print(f"📄 File trace đầy đủ đã được lưu tại: {trace_path}")

    plot_variance_comparison(
        variance_dict, 
        output_path=os.path.join(report_dir, "pca_variance_comparison.png")
    )

if __name__ == "__main__":
    main()
