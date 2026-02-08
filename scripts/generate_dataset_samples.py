import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

def generate_sample_grid():
    csv_path = "/Volumes/Toan/ML2/data/Dataset/GTRSB/Train.csv"
    root_dir = "/Volumes/Toan/ML2/data/Dataset/GTRSB/"
    output_path = "/Volumes/Toan/ML2/reports/figures/dataset_samples.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Lấy 10 lớp đầu tiên
    num_classes = 10
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    for cid in range(num_classes):
        # Lấy ảnh đầu tiên của mỗi lớp
        sample = df[df['ClassId'] == cid].iloc[0]
        img_path = os.path.join(root_dir, sample['Path'])
        
        img = Image.open(img_path)
        
        ax = axes[cid // 5, cid % 5]
        ax.imshow(img)
        ax.set_title(f"Class ID: {cid}")
        ax.axis('off')
        
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✅ Đã tạo ảnh lưới mẫu tại: {output_path}")

if __name__ == "__main__":
    generate_sample_grid()
