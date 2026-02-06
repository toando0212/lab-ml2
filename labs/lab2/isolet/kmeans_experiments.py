"""
K-means Clustering Experiments on ISOLET Dataset
=================================================
Dataset: ISOLET (Isolated Letter Speech Recognition)
- 617 features (continuous)
- 26 classes (letters A-Z)
- Training set: isolet1234.data (6239 samples)
- Test set: isolet5.data

Experimental Protocol:
1. Load and preprocess data
2. Apply standardization
3. Run k-means with different k values
4. Evaluate clustering quality using multiple metrics
5. Analyze results and visualize
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')


class ISOLETKMeansExperiment:
    """
    Class để thực hiện các thử nghiệm k-means trên dataset ISOLET
    """
    
    def __init__(self, data_path):
        """
        Khởi tạo thử nghiệm
        
        Parameters:
        -----------
        data_path : str
            Đường dẫn đến file dữ liệu ISOLET
        """
        self.data_path = data_path
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = None
        self.results = []
        
    def load_data(self):
        """
        Tải dữ liệu từ file
        
        Format: 617 features + 1 label (cuối cùng)
        """
        print("=" * 80)
        print("BƯỚC 1: TẢI DỮ LIỆU")
        print("=" * 80)
        
        # Đọc dữ liệu
        data = pd.read_csv(self.data_path, header=None)
        print(f"Kích thước dữ liệu: {data.shape}")
        
        # Tách features và labels
        self.X = data.iloc[:, :-1].values  # 617 features
        self.y = data.iloc[:, -1].values   # Labels (1-26)
        
        print(f"Số lượng mẫu: {self.X.shape[0]}")
        print(f"Số lượng đặc trưng: {self.X.shape[1]}")
        print(f"Số lượng lớp: {len(np.unique(self.y))}")
        print(f"Phân bố các lớp:")
        unique, counts = np.unique(self.y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Lớp {int(label)}: {count} mẫu")
        print()
        
    def preprocess_data(self):
        """
        Chuẩn hóa dữ liệu sử dụng StandardScaler
        
        Lý do: K-means nhạy cảm với scale của dữ liệu
        """
        print("=" * 80)
        print("BƯỚC 2: CHUẨN HÓA DỮ LIỆU")
        print("=" * 80)
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print("Đã chuẩn hóa dữ liệu (mean=0, std=1)")
        print(f"Mean sau chuẩn hóa: {np.mean(self.X_scaled, axis=0)[:5]} ...")
        print(f"Std sau chuẩn hóa: {np.std(self.X_scaled, axis=0)[:5]} ...")
        print()
        
    def run_kmeans_experiment(self, k_values, n_init=10, max_iter=300, random_state=42):
        """
        Chạy thử nghiệm k-means với nhiều giá trị k khác nhau
        
        Parameters:
        -----------
        k_values : list
            Danh sách các giá trị k cần thử nghiệm
        n_init : int
            Số lần chạy k-means với các centroid khởi tạo khác nhau
        max_iter : int
            Số vòng lặp tối đa
        random_state : int
            Seed cho reproducibility
        """
        print("=" * 80)
        print("BƯỚC 3: CHẠY THỬ NGHIỆM K-MEANS")
        print("=" * 80)
        print(f"Tham số:")
        print(f"  - Giá trị k: {k_values}")
        print(f"  - n_init: {n_init}")
        print(f"  - max_iter: {max_iter}")
        print(f"  - random_state: {random_state}")
        print()
        
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"Thử nghiệm với k = {k}")
            print(f"{'='*60}")
            
            # Bắt đầu đo thời gian
            start_time = time.time()
            
            # Khởi tạo và huấn luyện k-means
            kmeans = KMeans(
                n_clusters=k,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
                verbose=0
            )
            
            # Fit và predict
            labels = kmeans.fit_predict(self.X_scaled)
            
            # Tính thời gian
            elapsed_time = time.time() - start_time
            
            # Tính các metrics
            result = {
                'k': k,
                'inertia': kmeans.inertia_,
                'n_iter': kmeans.n_iter_,
                'time': elapsed_time,
                'labels': labels,
                'cluster_centers': kmeans.cluster_centers_
            }
            
            # Internal metrics (không cần ground truth)
            result['silhouette'] = silhouette_score(self.X_scaled, labels)
            result['davies_bouldin'] = davies_bouldin_score(self.X_scaled, labels)
            result['calinski_harabasz'] = calinski_harabasz_score(self.X_scaled, labels)
            
            # External metrics (so sánh với ground truth)
            result['adjusted_rand'] = adjusted_rand_score(self.y, labels)
            result['nmi'] = normalized_mutual_info_score(self.y, labels)
            result['homogeneity'] = homogeneity_score(self.y, labels)
            result['completeness'] = completeness_score(self.y, labels)
            result['v_measure'] = v_measure_score(self.y, labels)
            
            self.results.append(result)
            
            # In kết quả
            print(f"\nKết quả:")
            print(f"  Thời gian: {elapsed_time:.2f}s")
            print(f"  Số vòng lặp: {kmeans.n_iter_}")
            print(f"  Inertia (SSE): {kmeans.inertia_:.2f}")
            print(f"\nInternal Metrics:")
            print(f"  Silhouette Score: {result['silhouette']:.4f} (cao hơn tốt hơn, [-1, 1])")
            print(f"  Davies-Bouldin Index: {result['davies_bouldin']:.4f} (thấp hơn tốt hơn)")
            print(f"  Calinski-Harabasz Index: {result['calinski_harabasz']:.2f} (cao hơn tốt hơn)")
            print(f"\nExternal Metrics (so với ground truth):")
            print(f"  Adjusted Rand Index: {result['adjusted_rand']:.4f} ([-1, 1], cao hơn tốt hơn)")
            print(f"  Normalized Mutual Info: {result['nmi']:.4f} ([0, 1], cao hơn tốt hơn)")
            print(f"  Homogeneity: {result['homogeneity']:.4f} ([0, 1], cao hơn tốt hơn)")
            print(f"  Completeness: {result['completeness']:.4f} ([0, 1], cao hơn tốt hơn)")
            print(f"  V-measure: {result['v_measure']:.4f} ([0, 1], cao hơn tốt hơn)")
            
    def visualize_results(self, save_dir='./'):
        """
        Visualize experimental results and save measurements.
        """
        print("\n" + "=" * 80)
        print("STEP 4: VISUALIZING RESULTS")
        print("=" * 80)
        
        # Create DataFrame
        df_results = pd.DataFrame([
            {
                'k': r['k'],
                'Inertia': r['inertia'],
                'Silhouette': r['silhouette'],
                'Davies-Bouldin': r['davies_bouldin'],
                'Calinski-Harabasz': r['calinski_harabasz'],
                'Adjusted Rand': r['adjusted_rand'],
                'NMI': r['nmi'],
                'Homogeneity': r['homogeneity'],
                'Completeness': r['completeness'],
                'V-measure': r['v_measure'],
                'Time (s)': r['time']
            }
            for r in self.results
        ])
        
        # Save results to CSV
        csv_path = f"{save_dir}/kmeans_results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # 1. Elbow curve (Inertia)
        plt.figure(figsize=(8, 5))
        plt.plot(df_results['k'], df_results['Inertia'], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia (WCSS)', fontsize=12)
        plt.title('Elbow Method - Inertia vs k', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/elbow_curve.png", dpi=300)
        plt.close()
        
        # 2. Internal Metrics (Silhouette)
        plt.figure(figsize=(8, 5))
        plt.plot(df_results['k'], df_results['Silhouette'], 'go-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/silhouette_score.png", dpi=300)
        plt.close()

         # 3. Internal Metrics (Davies-Bouldin)
        plt.figure(figsize=(8, 5))
        plt.plot(df_results['k'], df_results['Davies-Bouldin'], 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Davies-Bouldin Index', fontsize=12)
        plt.title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/davies_bouldin.png", dpi=300)
        plt.close()
        
        # 4. External Metrics (NMI & ARI)
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['k'], df_results['NMI'], 'b^-', label='NMI', linewidth=2, markersize=8)
        plt.plot(df_results['k'], df_results['Adjusted Rand'], 'rs-', label='ARI', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('External Metrics: NMI and ARI', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/external_metrics.png", dpi=300)
        plt.close()
        
    def print_summary(self):
        """
        In tóm tắt kết quả
        """
        print("\n" + "=" * 80)
        print("TÓM TẮT KẾT QUẢ THỬ NGHIỆM")
        print("=" * 80)
        
        df_results = pd.DataFrame([
            {
                'k': r['k'],
                'Silhouette': r['silhouette'],
                'Davies-Bouldin': r['davies_bouldin'],
                'NMI': r['nmi'],
                'ARI': r['adjusted_rand']
            }
            for r in self.results
        ])
        
        print("\nBảng tóm tắt các metrics chính:")
        print(df_results.to_string(index=False))
        
        # Tìm k tốt nhất theo từng metric
        print("\n" + "-" * 80)
        print("K tốt nhất theo từng metric:")
        print("-" * 80)
        print(f"Silhouette Score (cao nhất): k = {df_results.loc[df_results['Silhouette'].idxmax(), 'k']:.0f}")
        print(f"Davies-Bouldin Index (thấp nhất): k = {df_results.loc[df_results['Davies-Bouldin'].idxmin(), 'k']:.0f}")
        print(f"NMI (cao nhất): k = {df_results.loc[df_results['NMI'].idxmax(), 'k']:.0f}")
        print(f"ARI (cao nhất): k = {df_results.loc[df_results['ARI'].idxmax(), 'k']:.0f}")
        

def main():
    """
    Hàm chính để chạy thử nghiệm
    """
    # Đường dẫn đến dữ liệu
    data_path = '/Volumes/Toan/ML2/Dataset/archivelab2/isolet1234.data'
    
    # Khởi tạo thử nghiệm
    experiment = ISOLETKMeansExperiment(data_path)
    
    # Tải và tiền xử lý dữ liệu
    experiment.load_data()
    experiment.preprocess_data()
    
    # Chạy thử nghiệm với các giá trị k khác nhau
    # Thử nghiệm với k từ 2 đến 40 (bao gồm k=26 là số lớp thực tế)
    k_values = [2, 5, 10, 15, 20, 26, 30, 35, 40]
    
    experiment.run_kmeans_experiment(
        k_values=k_values,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    # Trực quan hóa kết quả
    experiment.visualize_results(save_dir='/Volumes/Toan/ML2/lab2/isolet')
    
    # In tóm tắt
    experiment.print_summary()
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH THỬ NGHIỆM!")
    print("=" * 80)


if __name__ == "__main__":
    main()
