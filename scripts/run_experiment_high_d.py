import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Mocking the path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments.models import BaselineModels
from src.experiments.evaluator import ExperimentEvaluator

def run_high_d_experiment():
    print("\n🚀 Bắt đầu thực nghiệm High-Dimensional (2048D)...")
    
    variants = [
        ("ResNet50", "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_features.npy", 
                     "/Volumes/Toan/ML2/data/Features/Variants/resnet50/resnet50_labels.npy"),
        ("InceptionV3", "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_features.npy", 
                       "/Volumes/Toan/ML2/data/Features/Variants/inception_v3/inception_v3_labels.npy")
    ]
    
    evaluator = ExperimentEvaluator("/Volumes/Toan/ML2/reports/experiments/high_d")
    all_results = []
    
    for name, f_path, l_path in variants:
        if not os.path.exists(f_path):
            print(f"⚠️ Bỏ qua {name}, không thấy file.")
            continue
            
        print(f"\n--- Backbone: {name} ---")
        X = np.load(f_path)
        y = np.load(l_path)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Scaling cho SGDC
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Random Forest
        rfc = BaselineModels.get_rfc()
        rfc.fit(X_train, y_train) # RFC ko cần scale
        y_pred_rfc = rfc.predict(X_test)
        res_rfc = evaluator.evaluate(y_test, y_pred_rfc, "RFC", name)
        all_results.append(res_rfc)
        
        # 2. SGDC
        sgdc = BaselineModels.get_sgdc()
        sgdc.fit(X_train_scaled, y_train)
        y_pred_sgdc = sgdc.predict(X_test_scaled)
        res_sgdc = evaluator.evaluate(y_test, y_pred_sgdc, "SGDC", name)
        all_results.append(res_sgdc)
        
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("/Volumes/Toan/ML2/reports/experiments/high_d_summary.csv", index=False)
    print("\n✅ Hoàn thành thực nghiệm High-D.")
    print(summary_df)

if __name__ == "__main__":
    run_high_d_experiment()
