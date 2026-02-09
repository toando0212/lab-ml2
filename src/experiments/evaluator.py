import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

class ExperimentEvaluator:
    """Tạo báo cáo metrics chi tiết và ma trận nhầm lẫn."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate(self, y_true, y_pred, model_name, backbone):
        print(f"  > Đang đánh giá {model_name}...")
        
        # 1. Classification Report
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv = os.path.join(self.output_dir, f"{model_name.lower()}_{backbone.lower()}_report.csv")
        report_df.to_csv(report_csv)
        
        # 2. Confusion Matrix Plot
        self._plot_cm(y_true, y_pred, model_name, backbone)
        
        acc = accuracy_score(y_true, y_pred)
        return {
            "Model": model_name,
            "Backbone": backbone,
            "Accuracy": acc
        }
    
    def _plot_cm(self, y_true, y_pred, model_name, backbone):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Confusion Matrix: {model_name} ({backbone})')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        cm_path = os.path.join(self.output_dir, f"{model_name.lower()}_{backbone.lower()}_cm.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"    ✓ Đã lưu Confusion Matrix tại: {cm_path}")
