import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mocking the path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments.models import BaselineModels
from src.experiments.evaluator import ExperimentEvaluator
from src.experiments.manifold import FeatureProjector

def run_low_d_experiment():
    print("\n🚀 Bắt đầu thực nghiệm Manifold-based (3D UMAP)...")
    
    variants = [
        ("ResNet50", "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_features.npy", 
                     "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_labels.npy"),
        ("InceptionV3", "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_features.npy", 
                       "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_labels.npy")
    ]
    
    evaluator = ExperimentEvaluator("/Volumes/Toan/ML2/reports/experiments/low_d")
    projector = FeatureProjector(n_components=3)
    
    all_results = []
    output_data_dir = "/Volumes/Toan/ML2/data/Features/Experiments"
    
    for name, f_path, l_path in variants:
        if not os.path.exists(f_path):
            print(f"⚠️ Bỏ qua {name}, không thấy file.")
            continue
            
        print(f"\n--- Backbone: {name} ---")
        X_raw = np.load(f_path)
        y = np.load(l_path)
        
        # Bước cực kỳ quan trọng: Chiếu xuống 3D
        X_3d = projector.fit_transform(X_raw)
        
        # Lưu lại feature mới theo yêu cầu của user
        projector.save_features(X_3d, y, output_data_dir, name.lower())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_3d, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Ở không gian 3D, ta không cần scale quá phức tạp vì UMAP đã normalize vùng lân cận
        # nhưng vẫn giữ thói quen scale cho SGDC (linear model)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Random Forest
        rfc = BaselineModels.get_rfc()
        rfc.fit(X_train, y_train)
        y_pred_rfc = rfc.predict(X_test)
        res_rfc = evaluator.evaluate(y_test, y_pred_rfc, "RFC-3D", name)
        all_results.append(res_rfc)
        
        # 2. SGDC
        sgdc = BaselineModels.get_sgdc()
        sgdc.fit(X_train_scaled, y_train)
        y_pred_sgdc = sgdc.predict(X_test_scaled)
        res_sgdc = evaluator.evaluate(y_test, y_pred_sgdc, "SGDC-3D", name)
        all_results.append(res_sgdc)
        
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("/Volumes/Toan/ML2/reports/experiments/low_d_summary.csv", index=False)
    print("\n✅ Hoàn thành thực nghiệm Low-D.")
    print(summary_df)

if __name__ == "__main__":
    run_low_d_experiment()
