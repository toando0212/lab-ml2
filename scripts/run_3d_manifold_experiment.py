import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Thêm path gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments.models import BaselineModels
from src.experiments.evaluator import ExperimentEvaluator
from src.experiments.manifold import FeatureProjector

def solve_3d_experiment():
    """
    Kịch bản: 
    1. Lấy feature 2048D thô.
    2. Chiếu xuống 3D UMAP và LƯU LẠI file .npy mới.
    3. Huấn luyện RFC, SGDC, SVM-RBF trên bộ 3D này.
    4. Lưu report và confusion matrix.
    """
    print("\n🎯 Bắt đầu thực nghiệm Manifold (3D UMAP) - Modular Version")
    
    # Cấu hình đường dẫn
    variants = [
        ("ResNet50", "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_features.npy", 
                     "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_labels.npy"),
        ("InceptionV3", "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_features.npy", 
                       "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_labels.npy")
    ]
    
    output_data_dir = "/Volumes/Toan/ML2/data/Features/Manifold_3D"
    output_report_dir = "/Volumes/Toan/ML2/reports/experiments/manifold_3d"
    
    projector = FeatureProjector(n_components=3)
    evaluator = ExperimentEvaluator(output_report_dir)
    
    all_summary = []
    
    for name, f_path, l_path in variants:
        if not os.path.exists(f_path):
            print(f"⚠️ Bỏ qua {name}")
            continue
            
        print(f"\n--- Xử lý Backbone: {name} ---")
        X_raw = np.load(f_path)
        y = np.load(l_path)
        
        # 1. Giảm chiều & Lưu file feature mới (Theo yêu cầu của bạn)
        X_3d = projector.fit_transform(X_raw)
        projector.save_features(X_3d, y, output_data_dir, name.lower())
        
        # 2. Chia tập dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X_3d, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Chuẩn hóa (Cần cho SGDC và SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Chạy các mô hình
        test_models = {
            "RFC": (BaselineModels.get_rfc(), X_train),           # RFC ko cần scale
            "SGDC": (BaselineModels.get_sgdc(), X_train_scaled),
            "SVM-RBF": (BaselineModels.get_svm_rbf(), X_train_scaled)
        }
        
        for m_name, (model, train_data) in test_models.items():
            print(f"  > Training {m_name}...")
            model.fit(train_data, y_train)
            
            # Predict
            eval_data = X_test if m_name == "RFC" else X_test_scaled
            y_pred = model.predict(eval_data)
            
            # Đánh giá & Lưu định tính (CM) + định lượng (CSV)
            res = evaluator.evaluate(y_test, y_pred, m_name, name)
            all_summary.append(res)
            
    # Lưu bảng tổng hợp cuối cùng
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(f"{output_report_dir}/3d_experiment_summary.csv", index=False)
    print("\n✅ Hoàn thành toàn bộ thực nghiệm 3D.")
    print(summary_df)

if __name__ == "__main__":
    solve_3d_experiment()
