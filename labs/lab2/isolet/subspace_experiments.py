"""
Subspace Clustering Experiments on ISOLET Dataset
=================================================
Part 2 of Labwork 2

Requirements:
1. Select high-dim dataset (>100 features) -> ISOLET (617 features)
2. PCA/SVD Visualization (2D/3D)
3. Clustering on PCA data (2D/3D) & Comparison
4. Clustering on Random Subspace & Comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import time
import warnings
import random

warnings.filterwarnings('ignore')

class SubspaceExperiment:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X = None
        self.y = None
        self.X_scaled = None
        self.k_true = 26  # Known ground truth for ISOLET
        
    def load_and_preprocess(self):
        print("LOADING DATA...")
        data = pd.read_csv(self.data_path, header=None)
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values
        
        # Standardize
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        print(f"Data Loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")

    def run_kmeans(self, data, n_clusters):
        """Helper to run K-means and return metrics"""
        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(data)
        elapsed = time.time() - start
        
        ari = adjusted_rand_score(self.y, labels)
        nmi = normalized_mutual_info_score(self.y, labels)
        sil = silhouette_score(data, labels)
        
        return labels, ari, nmi, sil, elapsed

    def experiment_pca_visualization(self, save_dir='./isolet'):
        print("\n=== STEP 2.1: PCA VISUALIZATION ===")
        
        # PCA 2D
        pca2 = PCA(n_components=2, random_state=42)
        X_pca2 = pca2.fit_transform(self.X_scaled)
        var2 = pca2.explained_variance_ratio_
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=self.y, cmap='tab20', s=10, alpha=0.6)
        plt.title(f'PCA 2D Visualization (Total Variance: {sum(var2):.2%})\nGround Truth Labels', fontsize=14)
        plt.xlabel(f'PC1 ({var2[0]:.2%})')
        plt.ylabel(f'PC2 ({var2[1]:.2%})')
        plt.colorbar(label='Class')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pca_2d_ground_truth.png", dpi=300)
        plt.close()
        print("Saved PCA 2D plot.")

        # PCA 3D
        pca3 = PCA(n_components=3, random_state=42)
        X_pca3 = pca3.fit_transform(self.X_scaled)
        var3 = pca3.explained_variance_ratio_
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=self.y, cmap='tab20', s=10, alpha=0.6)
        ax.set_title(f'PCA 3D Visualization (Total Variance: {sum(var3):.2%})\nGround Truth Labels', fontsize=14)
        ax.set_xlabel(f'PC1 ({var3[0]:.2%})')
        ax.set_ylabel(f'PC2 ({var3[1]:.2%})')
        ax.set_zlabel(f'PC3 ({var3[2]:.2%})')
        plt.colorbar(sc, label='Class')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pca_3d_ground_truth.png", dpi=300)
        plt.close()
        print("Saved PCA 3D plot.")
        
        return X_pca2, X_pca3

    def experiment_clustering_pca(self, X_pca2, X_pca3):
        print("\n=== STEP 2.2: CLUSTERING ON PCA REDUCED DATA ===")
        
        # Run on PCA 2D
        labels_2d, ari_2d, nmi_2d, sil_2d, t_2d = self.run_kmeans(X_pca2, self.k_true)
        print(f"PCA 2D -> NMI: {nmi_2d:.4f}, ARI: {ari_2d:.4f}, Time: {t_2d:.2f}s")
        
        # Run on PCA 3D
        labels_3d, ari_3d, nmi_3d, sil_3d, t_3d = self.run_kmeans(X_pca3, self.k_true)
        print(f"PCA 3D -> NMI: {nmi_3d:.4f}, ARI: {ari_3d:.4f}, Time: {t_3d:.2f}s")
        
        return {
            'PCA 2D': {'NMI': nmi_2d, 'ARI': ari_2d, 'Time': t_2d},
            'PCA 3D': {'NMI': nmi_3d, 'ARI': ari_3d, 'Time': t_3d}
        }

    def experiment_random_subspace(self):
        print("\n=== STEP 2.3: CLUSTERING ON RANDOM SUBSPACE ===")
        
        # Select random 10% of features (approx 62 features)
        n_features = self.X_scaled.shape[1]
        n_subspace = int(n_features * 0.1)
        random_indices = sorted(random.sample(range(n_features), n_subspace))
        
        X_subspace = self.X_scaled[:, random_indices]
        print(f"Selected {n_subspace} random features indices: {random_indices[:5]}...")
        
        labels_rand, ari_rand, nmi_rand, sil_rand, t_rand = self.run_kmeans(X_subspace, self.k_true)
        print(f"Random Subspace -> NMI: {nmi_rand:.4f}, ARI: {ari_rand:.4f}, Time: {t_rand:.2f}s")
        
        return {'Random Subspace': {'NMI': nmi_rand, 'ARI': ari_rand, 'Time': t_rand}}

    def print_comparison(self, results):
        print("\n=== FINAL COMPARISON ===")
        # Original results (hardcoded from Part 1 execution for comparison)
        # NMI: 0.7308, ARI: 0.5225 (from k=26)
        results['Original (Full)'] = {'NMI': 0.7308, 'ARI': 0.5225, 'Time': 1.5} 
        
        df = pd.DataFrame(results).T
        print(df)
        df.to_csv('isolet/subspace_results.csv')

def main():
    exp = SubspaceExperiment('/Volumes/Toan/ML2/Dataset/archivelab2/isolet1234.data')
    exp.load_and_preprocess()
    
    # 2.1
    X_pca2, X_pca3 = exp.experiment_pca_visualization(save_dir='./isolet')
    
    # 2.2
    results = {}
    pca_results = exp.experiment_clustering_pca(X_pca2, X_pca3)
    results.update(pca_results)
    
    # 2.3
    rand_results = exp.experiment_random_subspace()
    results.update(rand_results)
    
    exp.print_comparison(results)

if __name__ == "__main__":
    main()
