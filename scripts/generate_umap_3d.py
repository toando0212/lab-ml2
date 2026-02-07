import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.analysis import run_umap

def main():
    # 1. Đường dẫn dữ liệu
    data_root = "/Volumes/Toan/ML2/data/Features/Variants"
    resnet_feat_path = os.path.join(data_root, "resnet50/resnet50_features.npy")
    resnet_lbl_path = os.path.join(data_root, "resnet50/resnet50_labels.npy")
    resnet_path_path = os.path.join(data_root, "resnet50/resnet50_paths.npy")
    
    inception_feat_path = os.path.join(data_root, "inception_v3/inception_v3_features.npy")
    inception_lbl_path = os.path.join(data_root, "inception_v3/inception_v3_labels.npy")
    
    report_dir = "/Volumes/Toan/ML2/reports/interactive"
    os.makedirs(report_dir, exist_ok=True)

    print("⏳ Đang nạp dữ liệu...")
    res_feat = np.load(resnet_feat_path)
    res_lbl = np.load(resnet_lbl_path).astype(str) # Chord Plotly color
    res_paths = np.load(resnet_path_path)
    
    inc_feat = np.load(inception_feat_path)
    inc_lbl = np.load(inception_lbl_path).astype(str)

    # 2. Chạy UMAP 3D
    print("⏳ Đang chạy UMAP 3D cho ResNet50...")
    res_umap = run_umap(res_feat, n_components=3)
    
    print("⏳ Đang chạy UMAP 3D for InceptionV3...")
    inc_umap = run_umap(inc_feat, n_components=3)

    # 3. Tạo biểu đồ tương tác 3D (Subplots 3D hơi phức tạp trong Plotly, nên làm 2 file riêng hoặc 1 file view chung)
    # Chúng ta sẽ tạo 2 HTML riêng để có trải nghiệm xoay/zoom mượt nhất
    
    for name, embedding, labels, paths in [("ResNet50", res_umap, res_lbl, res_paths), 
                                           ("InceptionV3", inc_umap, inc_lbl, res_paths)]:
        print(f"📊 Đang tạo HTML cho {name}...")
        fig = px.scatter_3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            color=labels,
            hover_name=paths,
            title=f"UMAP 3D Interactive Visualization: {name} Features",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            opacity=0.7,
            size_max=10
        )
        fig.update_traces(marker=dict(size=3))
        output_file = os.path.join(report_dir, f"umap_3d_{name.lower()}.html")
        fig.write_html(output_file)
        print(f"✅ Đã lưu: {output_file}")

if __name__ == "__main__":
    main()
