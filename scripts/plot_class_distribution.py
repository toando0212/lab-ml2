import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_distribution(csv_path, output_path):
    print(f"Đang đọc dữ liệu từ: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Tính toán số lượng mẫu mỗi lớp
    class_counts = df['ClassId'].value_counts().sort_index()
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Sử dụng tông màu nhã nhặn, chuyên nghiệp
    colors = sns.color_palette("viridis", len(class_counts))
    barplot = sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors)
    
    # Thêm số lượng cụ thể trên đầu mỗi cột
    for i, count in enumerate(class_counts.values):
        barplot.text(i, count + 20, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title("Phân phối số lượng mẫu cho 10 lớp GTSRB", fontsize=15, pad=20)
    plt.xlabel("Class ID", fontsize=12)
    plt.ylabel("Số lượng mẫu (Samples)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Thêm chú thích về tình trạng imbalance
    plt.annotate(f"Min: {class_counts.min()} (Class {class_counts.idxmin()})\nMax: {class_counts.max()} (Class {class_counts.idxmax()})", 
                 xy=(0.05, 0.9), xycoords='axes fraction', fontsize=11, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ phân phối tại: {output_path}")

if __name__ == "__main__":
    CSV_PATH = "/Volumes/Toan/ML2/Features/resnet50_features.csv"
    OUTPUT_PATH = "/Volumes/Toan/ML2/Results_SVM/class_distribution.png"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plot_distribution(CSV_PATH, OUTPUT_PATH)
