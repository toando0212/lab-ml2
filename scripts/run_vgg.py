import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import time
from src.data_loader import get_dataloader
from src.extractors.vgg import VGG16Extractor

def run():
    csv_train = "/Volumes/Toan/ML2/data/Dataset/GTRSB/Train.csv"
    root_dir = "/Volumes/Toan/ML2/data/Dataset/GTRSB/"
    output_dir = "/Volumes/Toan/ML2/data/Features/Variants/vgg16"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Bắt đầu trích xuất: VGG16")
    loader = get_dataloader(csv_train, root_dir, img_size=224, batch_size=32)
    extractor = VGG16Extractor()
    
    start_time = time.time()
    features, labels, paths = extractor.extract(loader)
    duration = time.time() - start_time
    
    np.save(os.path.join(output_dir, "vgg16_features.npy"), features)
    np.save(os.path.join(output_dir, "vgg16_labels.npy"), labels)
    np.save(os.path.join(output_dir, "vgg16_paths.npy"), paths)
    
    # Lưu Debug CSV (Lấy 5 mẫu mỗi lớp)
    debug_indices = []
    for cid in range(10):
        idx = np.where(labels == cid)[0]
        debug_indices.extend(idx[:5])
        
    df_debug = pd.DataFrame(features[debug_indices], columns=[f"feat_{i}" for i in range(features.shape[1])])
    df_debug['ClassId'] = labels[debug_indices]
    df_debug['Path'] = paths[debug_indices]
    df_debug.to_csv(os.path.join(output_dir, "vgg16_debug.csv"), index=False)
    
    print(f"✅ Hoàn thành VGG16. Thời gian: {duration:.2f}s")

if __name__ == "__main__":
    run()
