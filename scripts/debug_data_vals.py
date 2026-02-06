import numpy as np
import os

def check():
    data_root = "/Volumes/Toan/ML2/data/Features/Variants"
    resnet_path = os.path.join(data_root, "resnet50/resnet50_features.npy")
    res_feat = np.load(resnet_path)
    
    print("ResNet features shape:", res_feat.shape)
    print("ResNet features mean:", np.mean(res_feat))
    print("ResNet features std:", np.std(res_feat))
    print("ResNet features min:", np.min(res_feat))
    print("ResNet features max:", np.max(res_feat))
    print("First 5 elements of first row:", res_feat[0, :5])
    
    unique_rows = len(np.unique(res_feat, axis=0))
    print("Unique rows:", unique_rows)

if __name__ == "__main__":
    check()
