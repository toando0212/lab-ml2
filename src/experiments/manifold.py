import numpy as np
import umap
import os
import joblib

class FeatureProjector:
    """Xử lý việc giảm chiều từ High-D (2048D) sang Low-D (3D/n-D)."""
    
    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        
    def fit_transform(self, X):
        print(f"  > Đang thực hiện UMAP projection sang {self.n_components} chiều...")
        return self.reducer.fit_transform(X)
    
    def save_features(self, features, labels, output_dir, name):
        os.makedirs(output_dir, exist_ok=True)
        feat_path = os.path.join(output_dir, f"{name}_umap_{self.n_components}d.npy")
        lbl_path = os.path.join(output_dir, f"{name}_labels.npy")
        
        np.save(feat_path, features)
        np.save(lbl_path, labels)
        
        # Lưu cả model UMAP nếu cần dùng cho inference sau này
        model_path = os.path.join(output_dir, f"{name}_umap_model.pkl")
        joblib.dump(self.reducer, model_path)
        
        print(f"  ✅ Đã lưu đặc trưng mới tại: {feat_path}")
        return feat_path, lbl_path
