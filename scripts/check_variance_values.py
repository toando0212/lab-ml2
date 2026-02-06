import numpy as np
from src.analysis import get_pca_variance_ratio

def check():
    data_root = "/Volumes/Toan/ML2/data/Features/Variants"
    resnet_path = os.path.join(data_root, "resnet50/resnet50_features.npy")
    inc_feat = np.load(os.path.join(data_root, "inception_v3/inception_v3_features.npy"))
    res_feat = np.load(resnet_path)
    
    from sklearn.decomposition import PCA
    pca_res = PCA(n_components=10).fit(res_feat)
    pca_inc = PCA(n_components=10).fit(inc_feat)
    
    print("ResNet Explained Variance Ratio:", pca_res.explained_variance_ratio_)
    print("Inception Explained Variance Ratio:", pca_inc.explained_variance_ratio_)

if __name__ == "__main__":
    import os
    check()
