import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
