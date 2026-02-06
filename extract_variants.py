import os
import pandas as pd
import time
from data_loader import get_dataloader
from extractors.resnet_extractor import ResNet50Extractor
from extractors.inception_extractor import InceptionV3Extractor
from extractors.vgg_extractor import VGG16Extractor
from extractors.mobilenet_extractor import MobileNetV2Extractor

def run_extraction_pipeline():
    # Cấu hình
    csv_train = "/Volumes/Toan/ML2/Dataset/GTRSB/Train.csv"
    root_dir = "/Volumes/Toan/ML2/Dataset/GTRSB/"
    output_dir = "/Volumes/Toan/ML2/Features/Variants"
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách các extractor cần chạy
    variant_configs = [
        {'name': 'resnet50', 'class': ResNet50Extractor, 'img_size': 224},
        {'name': 'inception_v3', 'class': InceptionV3Extractor, 'img_size': 299},
        {'name': 'vgg16', 'class': VGG16Extractor, 'img_size': 224},
        {'name': 'mobilenet_v2', 'class': MobileNetV2Extractor, 'img_size': 224},
    ]
    
    summary_results = []

    for cfg in variant_configs:
        print(f"\n🚀 Bắt đầu trích xuất với: {cfg['name'].upper()}")
        
        # 1. Khởi tạo dataloader với img_size tương ứng
        # Mặc định num_classes=10 trong get_dataloader (đã có sẵn trong data_loader.py)
        loader = get_dataloader(csv_train, root_dir, img_size=cfg['img_size'], batch_size=32)
        
        # 2. Khởi tạo Extractor
        extractor = cfg['class']()
        
        # 3. Trích xuất và đo thời gian
        start_time = time.time()
        features, labels, paths = extractor.extract(loader)
        duration = time.time() - start_time
        
        # 4. Lưu dữ liệu tối ưu (.npy)
        # Tạo thư mục con cho từng model để gọn gàng
        model_dir = os.path.join(output_dir, cfg['name'])
        os.makedirs(model_dir, exist_ok=True)
        
        # Lưu toàn bộ đặc trưng nhị phân
        np.save(os.path.join(model_dir, "features.npy"), features)
        np.save(os.path.join(model_dir, "labels.npy"), labels)
        np.save(os.path.join(model_dir, "paths.npy"), paths)
        
        # 5. Lưu 50 dòng đầu ra CSV để debug (Xem bằng mắt)
        debug_size = min(50, len(labels))
        cols = [f"feat_{i}" for i in range(features.shape[1])]
        df_debug = pd.DataFrame(features[:debug_size], columns=cols)
        df_debug['ClassId'] = labels[:debug_size]
        df_debug['Path'] = paths[:debug_size]
        
        debug_csv_path = os.path.join(model_dir, f"debug_first_{debug_size}.csv")
        df_debug.to_csv(debug_csv_path, index=False)
        
        print(f"✅ Hoàn thành {cfg['name']}. Thời gian: {duration:.2f}s.")
        print(f"   - Toàn bộ data lưu tại: {model_dir}/*.npy (Binary)")
        print(f"   - File debug: {os.path.basename(debug_csv_path)} (CSV)")
        
        summary_results.append({
            'Model': cfg['name'],
            'Dimensions': features.shape[1],
            'Time (s)': round(duration, 2),
            'Samples': len(labels)
        })

    # 6. Lưu báo cáo tổng tóm tắt
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "="*50)
    print("TỔNG KẾT TRÍCH XUẤT ĐA KIẾN TRÚC")
    print(summary_df)
    print("="*50)

if __name__ == "__main__":
    run_extraction_pipeline()
