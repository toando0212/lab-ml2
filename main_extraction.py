import os
import numpy as np
import torch
from data_loader import get_dataloader
from feature_extractor import FeatureExtractor

def main():
    # Cấu hình đường dẫn
    csv_train = "/Volumes/Toan/ML2/Dataset/GTRSB/Train.csv"
    root_dir = "/Volumes/Toan/ML2/Dataset/GTRSB/"
    output_dir = "/Volumes/Toan/ML2/Features"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Trích xuất bằng ResNet50
    print("\n--- Trích xuất bằng ResNet50 ---")
    loader_resnet = get_dataloader(csv_train, root_dir, img_size=224)
    extractor_resnet = FeatureExtractor(model_name='resnet50', device=device)
    feat_resnet, labels_resnet, paths_resnet = extractor_resnet.extract(loader_resnet)
    
    # Tạo DataFrame để lưu CSV
    import pandas as pd
    cols = [f"feat_{i}" for i in range(feat_resnet.shape[1])]
    df = pd.DataFrame(feat_resnet, columns=cols)
    df['ClassId'] = labels_resnet
    df['Path'] = paths_resnet
    
    csv_path = os.path.join(output_dir, "resnet50_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Đã lưu ResNet50 features ra CSV: {csv_path}")

    # Ghi log kết quả vào file txt
    log_file = os.path.join(output_dir, "extraction_summary.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY LOG: FEATURE EXTRACTION (RESNET50 ONLY) ===\n")
        f.write(f"Thiết bị sử dụng: {device}\n\n")
        
        f.write("1. ResNet50:\n")
        f.write(f"   - Feature shape: {feat_resnet.shape}\n")
        f.write(f"   - Số lượng mẫu: {len(labels_resnet)}\n")
        f.write(f"   - File CSV: {os.path.basename(csv_path)}\n\n")
        
        f.write("2. Thống kê theo lớp (ClassId 0-9):\n")
        unique, counts = np.unique(labels_resnet, return_counts=True)
        for cls, count in zip(unique, counts):
            f.write(f"   - Class {int(cls)}: {count} mẫu\n")
            
    print(f"\nĐã ghi log kết quả vào: {log_file}")

if __name__ == "__main__":
    main()
