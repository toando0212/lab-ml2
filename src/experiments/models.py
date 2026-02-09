from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

class BaselineModels:
    """Cung cấp các mô hình baseline với cấu hình chuẩn."""
    
    @staticmethod
    def get_rfc(n_estimators=100, random_state=42):
        return RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1
        )
    
    @staticmethod
    def get_sgdc(loss='hinge', random_state=42):
        return SGDClassifier(
            loss=loss, 
            alpha=0.0001, 
            max_iter=1000, 
            random_state=random_state, 
            n_jobs=-1
        )

    @staticmethod
    def get_svm_rbf(random_state=42):
        return SVC(
            kernel='rbf', 
            probability=True, 
            random_state=random_state
        )
