import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
import umap
import os
import time

def run_visualization(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Bắt đầu đọc dữ liệu từ: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Tách đặc trưng và nhãn
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X = df[feature_cols].values
    y = df['ClassId'].values
    
    # 1. PCA Denoising (Giảm xuống 50 chiều trước khi chạy UMAP hoặc PCA chi tiết)
    print("Đang chạy PCA (Denoising to 50 dims)...")
    pca_50 = PCA(n_components=50, random_state=42)
    X_pca_50 = pca_50.fit_transform(X)
    
    # ---------------------------------------------------------
    # 2. PCA 2D & 3D
    # ---------------------------------------------------------
    print("Đang thực hiện PCA 2D & 3D...")
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_pca_50) # Chạy trên kết quả 50 dims để ổn định
    
    # Lưu kết quả 2D (lấy 2 cột đầu của 3D)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca_3d[:, 0], y=X_pca_3d[:, 1], hue=y, palette='tab10', legend='full', alpha=0.6)
    plt.title("PCA 2D Visualization (GTSRB 10 Classes)")
    plt.savefig(os.path.join(output_dir, "pca_2d.png"))
    print("Đã lưu PCA 2D (PNG)")
    
    # Lưu kết quả 3D Interactive
    fig_pca_3d = px.scatter_3d(
        x=X_pca_3d[:, 0], y=X_pca_3d[:, 1], z=X_pca_3d[:, 2],
        color=y.astype(str), title="PCA 3D Visualization",
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
    )
    fig_pca_3d.write_html(os.path.join(output_dir, "pca_3d.html"))
    print("Đã lưu PCA 3D (HTML)")

    # Lưu kết quả 3D Tĩnh (PNG) bằng matplotlib để đưa vào báo cáo
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='tab10', alpha=0.6)
    ax.set_title("PCA 3D Static Visualization")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter, label='ClassId')
    plt.savefig(os.path.join(output_dir, "pca_3d.png"))
    plt.close()
    print("Đã lưu PCA 3D Tĩnh (PNG)")

    # ---------------------------------------------------------
    # 3. UMAP 2D & 3D
    # ---------------------------------------------------------
    print("Đang thực hiện UMAP 2D & 3D (Việc này có thể tốn vài phút)...")
    start_time = time.time()
    
    # UMAP 2D
    reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap_2d = reducer_2d.fit_transform(X_pca_50)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_umap_2d[:, 0], y=X_umap_2d[:, 1], hue=y, palette='tab10', legend='full', alpha=0.6)
    plt.title("UMAP 2D Visualization (GTSRB 10 Classes)")
    plt.savefig(os.path.join(output_dir, "umap_2d.png"))
    print("Đã lưu UMAP 2D (PNG)")
    
    # UMAP 3D
    reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap_3d = reducer_3d.fit_transform(X_pca_50)
    
    fig_umap_3d = px.scatter_3d(
        x=X_umap_3d[:, 0], y=X_umap_3d[:, 1], z=X_umap_3d[:, 2],
        color=y.astype(str), title="UMAP 3D Visualization"
    )
    fig_umap_3d.write_html(os.path.join(output_dir, "umap_3d.html"))
    print(f"Đã lưu UMAP 3D (HTML). Tổng thời gian UMAP: {time.time() - start_time:.2f}s")

    # Lưu kết quả UMAP 3D Tĩnh (PNG)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=y, cmap='tab10', alpha=0.6)
    ax.set_title("UMAP 3D Static Visualization")
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_zlabel('U3')
    plt.colorbar(scatter, label='ClassId')
    plt.savefig(os.path.join(output_dir, "umap_3d.png"))
    plt.close()
    print("Đã lưu UMAP 3D Tĩnh (PNG)")

if __name__ == "__main__":
    CSV_PATH = "/Volumes/Toan/ML2/Features/resnet50_features.csv"
    OUTPUT_DIR = "/Volumes/Toan/ML2/Visuals"
    run_visualization(CSV_PATH, OUTPUT_DIR)
