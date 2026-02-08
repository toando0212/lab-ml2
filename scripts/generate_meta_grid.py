import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

def generate_meta_grid():
    root_dir = "/Volumes/Toan/ML2/data/Dataset/GTRSB/"
    meta_dir = os.path.join(root_dir, "Meta")
    output_path = "/Volumes/Toan/ML2/reports/figures/meta_reference.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_classes = 10
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Giả định ClassId 0-9 tương ứng với 0.png đến 9.png
    for cid in range(num_classes):
        img_path = os.path.join(meta_dir, f"{cid}.png")
        
        if not os.path.exists(img_path):
            print(f"⚠️ Không tìm thấy: {img_path}")
            continue
            
        img = Image.open(img_path)
        
        ax = axes[cid // 5, cid % 5]
        ax.imshow(img)
        ax.set_title(f"Class ID: {cid}")
        ax.axis('off')
        
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✅ Đã tạo bảng tra cứu Meta tại: {output_path}")

if __name__ == "__main__":
    generate_meta_grid()
