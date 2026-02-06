import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import time
from src.data_loader import get_dataloader
from src.extractors.inception import InceptionV3Extractor

def run():
    csv_train = "/Volumes/Toan/ML2/data/Dataset/GTRSB/Train.csv"
    root_dir = "/Volumes/Toan/ML2/data/Dataset/GTRSB/"
    output_dir = "/Volumes/Toan/ML2/data/Features/Variants/inception_v3"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Bắt đầu trích xuất: INCEPTION_V3")
    # Inception yêu cầu size 299
    loader = get_dataloader(csv_train, root_dir, img_size=299, batch_size=32)
    extractor = InceptionV3Extractor()
    
    start_time = time.time()
    features, labels, paths = extractor.extract(loader)
    duration = time.time() - start_time
    
    np.save(os.path.join(output_dir, "inception_v3_features.npy"), features)
    np.save(os.path.join(output_dir, "inception_v3_labels.npy"), labels)
    np.save(os.path.join(output_dir, "inception_v3_paths.npy"), paths)
    
    # Lưu Debug CSV (Lấy 5 mẫu mỗi lớp)
    debug_indices = []
    for cid in range(10):
        idx = np.where(labels == cid)[0]
        debug_indices.extend(idx[:5])
        
    df_debug = pd.DataFrame(features[debug_indices], columns=[f"feat_{i}" for i in range(features.shape[1])])
    df_debug['ClassId'] = labels[debug_indices]
    df_debug['Path'] = paths[debug_indices]
    df_debug.to_csv(os.path.join(output_dir, "inception_v3_debug.csv"), index=False)
    
    print(f"✅ Hoàn thành InceptionV3. Thời gian: {duration:.2f}s")

if __name__ == "__main__":
    run()
