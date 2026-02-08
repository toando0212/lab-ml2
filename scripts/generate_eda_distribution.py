import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    # 1. Đường dẫn
    lbl_path = "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_labels.npy"
    report_dir = "/Volumes/Toan/ML2/reports/figures"
    os.makedirs(report_dir, exist_ok=True)
    
    # 2. Đọc nhãn
    labels = np.load(lbl_path)
    df = pd.DataFrame(labels, columns=['ClassId'])
    
    # 3. Thống kê số liệu
    counts = df['ClassId'].value_counts().sort_index()
    print("📊 Thống kê số lượng mẫu mỗi lớp:")
    print(counts)
    
    # 4. Vẽ biểu đồ Phân bố lớp
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(data=df, x='ClassId', palette="viridis")
    
    # Thêm số lượng trên đầu cột
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold')

    plt.title("Sample Distribution for the First 10 Classes (GTSRB)", fontsize=15, pad=20)
    plt.xlabel("Class ID", fontsize=12)
    plt.ylabel("Number of Samples (Images)", fontsize=12)
    
    output_path = os.path.join(report_dir, "class_distribution_10_classes.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Biểu đồ đã được lưu tại: {output_path}")

if __name__ == "__main__":
    main()
