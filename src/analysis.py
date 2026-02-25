import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

def calculate_cluster_metrics(features, labels, sample_size=5000):
    """Tính toán các chỉ số định lượng cho độ sắc nét của các cụm đặc trưng."""
    
    # Nếu dữ liệu quá lớn, lấy mẫu để tính toán nhanh hơn
    if features.shape[0] > sample_size:
        indices = np.random.choice(features.shape[0], sample_size, replace=False)
        X_sample = features[indices]
        y_sample = labels[indices]
    else:
        X_sample = features
        y_sample = labels

    s_score = silhouette_score(X_sample, y_sample)
    db_index = davies_bouldin_score(X_sample, y_sample)
    
    # Calculate Supervised Clustering Metrics (ARI, NMI) using K-Means
    # We use K-Means to find "natural" clusters and see how well they match ground truth
    n_classes = len(np.unique(y_sample))
    kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_sample)
    
    ari_score = adjusted_rand_score(y_sample, cluster_labels)
    nmi_score = normalized_mutual_info_score(y_sample, cluster_labels)
    
    return s_score, db_index, ari_score, nmi_score

def run_pca(features, n_components=3):
    """Giảm chiều dữ liệu sử dụng PCA (Mặc định 3D)."""
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(features)
    return embedding

def run_umap(features, n_neighbors=15, min_dist=0.1, n_components=3, random_state=42):
    """Giảm chiều dữ liệu sử dụng UMAP (Mặc định 3D)."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    embedding = reducer.fit_transform(features)
    return embedding

def get_pca_variance_ratio(features, n_components=100):
    """Tính toán tỷ lệ phương sai đơn lẻ và tích lũy của PCA."""
    pca = PCA(n_components=min(n_components, features.shape[0], features.shape[1]))
    pca.fit(features)
    individual_ratios = pca.explained_variance_ratio_
    cumulative_ratios = np.cumsum(individual_ratios)
    return individual_ratios, cumulative_ratios

def plot_variance_comparison(variance_dict, output_path=None):
    """Vẽ biểu đồ so sánh phương sai tích lũy giữa các mô hình."""
    plt.figure(figsize=(10, 6))
    
    for name, cumulative_variance in variance_dict.items():
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=name, linewidth=2)
    
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Comparison (Ablation Study)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Biểu đồ đã được lưu tại: {output_path}")
    
    plt.show()
