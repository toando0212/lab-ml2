import pandas as pd
import os

def analyze_errors(csv_path, model_name):
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy tệp: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Thống kê các cặp nhầm lẫn
    confusion_counts = df.groupby(['actual_class', 'predicted_class']).size().reset_index(name='count')
    confusion_counts = confusion_counts.sort_values(by='count', ascending=False)
    
    print(f"\n--- Phân tích lỗi cho mô hình: {model_name} ---")
    print(f"Tổng số mẫu sai: {len(df)}")
    print("\nTop 10 cặp nhầm lẫn phổ biến nhất:")
    
    class_names = {
        0: '20km/h', 1: '30km/h', 2: '50km/h', 3: '60km/h', 4: '70km/h',
        5: '80km/h', 6: 'End 80km/h', 7: '100km/h', 8: '120km/h', 9: 'No passing'
    }
    
    for i, row in confusion_counts.head(10).iterrows():
        actual = row['actual_class']
        pred = row['predicted_class']
        count = row['count']
        percentage = (count / len(df)) * 100
        print(f"Lớp {actual} ({class_names.get(actual, 'Unknown')}) → Lớp {pred} ({class_names.get(pred, 'Unknown')}): {count} mẫu ({percentage:.1f}%)")

def main():
    report_dir = "/Volumes/Toan/ML2/reports/performance"
    resnet_csv = f"{report_dir}/resnet50_rbf_misclassified.csv"
    inception_csv = f"{report_dir}/inceptionv3_rbf_misclassified.csv"
    
    analyze_errors(resnet_csv, "ResNet50 + RBF")
    analyze_errors(inception_csv, "InceptionV3 + RBF")

if __name__ == "__main__":
    main()
