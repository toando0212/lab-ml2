# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
#     "tabulate",
# ]
# ///

import pandas as pd
import numpy as np
import os
import sys

def main():
    print("=== WINE DATASET: STATISTICAL ANALYSIS ===\n")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../Dataset/WineQT.csv')
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Prepare Features (exclude target column and Id)
    # Target is 'quality', Id is metadata
    cols_to_drop = ['quality', 'Id']
    features_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Analyzing {features_df.shape[1]} numeric features on Wine dataset.\n")

    # 3. Calculate Statistics
    print("[1. Mean (Trung bình)]")
    mean_values = features_df.mean()
    
    print("\n[2. Variance (Phương sai)]")
    variance_values = features_df.var()
    
    print("\n[3. Covariance Matrix (Ma trận hiệp phương sai)]")
    covariance_matrix = features_df.cov()
    
    print("\n[4. Correlation Matrix (Ma trận tương quan)]")
    correlation_matrix = features_df.corr()
    
    # 4. Write to Markdown File
    output_path = os.path.join(script_dir, '1.5_stats_results.md')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 1.5 Kết quả phân tích thống kê (Statistical Analysis Results)\n\n")
        f.write("**Dataset**: Wine Quality Dataset  \n")
        f.write(f"**Số lượng đặc trưng**: {features_df.shape[1]}  \n")
        f.write(f"**Số lượng mẫu**: {features_df.shape[0]}\n\n")
        
        # Mean and Variance Table
        f.write("## 1. Mean (Trung bình) và Variance (Phương sai)\n\n")
        stats_df = pd.DataFrame({
            'Feature': mean_values.index,
            'Mean': mean_values.values,
            'Variance': variance_values.values
        })
        f.write(stats_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Covariance Matrix
        f.write("## 2. Covariance Matrix (Ma trận Hiệp phương sai 13×13)\n\n")
        f.write(covariance_matrix.to_markdown())
        f.write("\n\n")
        
        # Correlation Matrix
        f.write("## 3. Correlation Matrix (Ma trận Tương quan 13×13)\n\n")
        f.write(correlation_matrix.to_markdown())
        f.write("\n\n")
        
        f.write("## Nhận xét\n\n")
        f.write("- Ma trận Covariance cho thấy mức độ biến thiên chung giữa các cặp đặc trưng.\n")
        f.write("- Ma trận Correlation (chuẩn hóa của Covariance) có giá trị trong khoảng [-1, 1].\n")
        f.write("- Giá trị gần 1: Tương quan dương mạnh (cùng tăng/giảm).\n")
        f.write("- Giá trị gần -1: Tương quan âm mạnh (ngược chiều).\n")
        f.write("- Giá trị gần 0: Không có tương quan tuyến tính.\n")
    
    print(f"\nResults saved to: {output_path}")
    print("Done.")

if __name__ == "__main__":
    main()
