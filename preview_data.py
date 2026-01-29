import numpy as np
import os

def preview_npy(file_path):
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return
    
    data = np.load(file_path, allow_pickle=True)
    print(f"\n--- Preview: {os.path.basename(file_path)} ---")
    print(f"Kiểu dữ liệu: {type(data)}")
    print(f"Kích thước (Shape): {data.shape}")
    
    if len(data.shape) == 1:
        print("5 giá trị đầu tiên:", data[:5])
    else:
        print("Dữ liệu dòng đầu tiên (5 cột đầu):", data[0, :5])

if __name__ == "__main__":
    feature_dir = "/Volumes/Toan/ML2/Features"
    files = [
        "resnet50_features.npy",
        "resnet50_labels.npy",
        "inception_features.npy",
        "extraction_summary.txt"
    ]
    
    for f in files:
        if f.endswith(".npy"):
            preview_npy(os.path.join(feature_dir, f))
        else:
            print(f"\n--- Content of {f} (Text file) ---")
            with open(os.path.join(feature_dir, f), 'r') as txt:
                print(txt.read())
