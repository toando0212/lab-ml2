import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

def visualize_failures(csv_path, model_name, output_img, base_data_dir):
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy tệp: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Chọn một vài mẫu tiêu biểu (ngẫu nhiên hoặc từ top confusion)
    # Ở đây chúng ta lấy 4 mẫu đầu tiên để minh họa
    samples = df.head(4)
    
    class_names = {
        0: '20kph', 1: '30kph', 2: '50kph', 3: '60kph', 4: '70kph',
        5: '80kph', 6: 'End80', 7: '100kph', 8: '120kph', 9: 'NoPass'
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Misclassified Samples: {model_name}", fontsize=16)

    for i, (idx, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(base_data_dir, row['image_path'])
        actual = row['actual_class']
        pred = row['predicted_class']
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"A: {actual} ({class_names.get(actual, '??')})\nP: {pred} ({class_names.get(pred, '??')})", color='red')
        else:
            axes[i].text(0.5, 0.5, f"Missing:\n{row['image_path']}", ha='center')
        
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    plt.close()
    print(f"✅ Đã lưu ảnh minh họa tại: {output_img}")

def main():
    report_dir = "/Volumes/Toan/ML2/reports/performance"
    base_data_dir = "/Volumes/Toan/ML2/data/Dataset/GTRSB"
    
    resnet_csv = f"{report_dir}/resnet50_rbf_misclassified.csv"
    inception_csv = f"{report_dir}/inceptionv3_rbf_misclassified.csv"
    
    # Chúng ta sẽ tạo 2 ảnh riêng biệt hoặc một ảnh gộp. 
    # Để đơn giản cho LaTeX, ta tạo 1 ảnh gộp 2 hàng.
    
    df_resnet = pd.read_csv(resnet_csv).head(4)
    df_inception = pd.read_csv(inception_csv).head(4)
    
    class_names = {
        0: '20kph', 1: '30kph', 2: '50kph', 3: '60kph', 4: '70kph',
        5: '80kph', 6: 'End80', 7: '100kph', 8: '120kph', 9: 'NoPass'
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Hàng 1: ResNet50
    for i, (idx, row) in enumerate(df_resnet.iterrows()):
        img_path = os.path.join(base_data_dir, row['image_path'])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"ResNet50\nAct: {row['actual_class']} ({class_names.get(row['actual_class'])})\nPre: {row['predicted_class']} ({class_names.get(row['predicted_class'])})", color='darkred', fontsize=10)
        axes[0, i].axis('off')

    # Hàng 2: InceptionV3
    for i, (idx, row) in enumerate(df_inception.iterrows()):
        img_path = os.path.join(base_data_dir, row['image_path'])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"InceptionV3\nAct: {row['actual_class']} ({class_names.get(row['actual_class'])})\nPre: {row['predicted_class']} ({class_names.get(row['predicted_class'])})", color='darkred', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    output_combined = f"{report_dir}/misclassified_samples.png"
    plt.savefig(output_combined)
    plt.close()
    print(f"✅ Đã lưu ảnh minh họa tổng hợp tại: {output_combined}")

if __name__ == "__main__":
    main()
