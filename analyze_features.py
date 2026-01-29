import pandas as pd
import numpy as np

def analyze_csv(file_path):
    print(f"Đang đọc tệp: {file_path}...")
    # Chỉ đọc head để lấy info nhanh nếu file quá lớn, nhưng ở đây cần thống kê nên đọc hết
    df = pd.read_csv(file_path)
    
    # 1. Số lượng sample
    num_samples = len(df)
    
    # 2. Số lượng class
    unique_classes = df['ClassId'].unique()
    num_classes = len(unique_classes)
    
    # 3. Số lượng feature
    # Feature columns là các cột bắt đầu bằng 'feat_'
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    num_features = len(feature_cols)
    
    # 4. Kiểu dữ liệu của feature
    # Lấy kiểu dữ liệu của cột feature đầu tiên làm đại diện
    feature_dtype = df[feature_cols[0]].dtype
    
    print("\n" + "="*40)
    print("KẾT QUẢ PHÂN TÍCH ĐẶC TRƯNG")
    print("="*40)
    print(f"- Tổng số mẫu (samples): {num_samples:,}")
    print(f"- Số lượng lớp (classes): {num_classes} ({sorted(unique_classes)})")
    print(f"- Số lượng đặc trưng (features): {num_features}")
    print(f"- Kiểu dữ liệu đặc trưng: {feature_dtype}")
    print("-" * 40)
    
    # Thống kê chi tiết mẫu mỗi lớp
    print("\nThống kê mẫu mỗi lớp:")
    print(df['ClassId'].value_counts().sort_index())
    print("="*40)

if __name__ == "__main__":
    csv_path = "/Volumes/Toan/ML2/Features/resnet50_features.csv"
    analyze_csv(csv_path)
