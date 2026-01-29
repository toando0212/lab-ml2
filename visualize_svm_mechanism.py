import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 1. Tạo dữ liệu giả lập bị mất cân bằng (Imbalanced)
# Lớp 0: 200 điểm (Đông)
# Lớp 1: 20 điểm (Ít)
n_samples_1 = 200
n_samples_2 = 20
centers = [[0.0, 0.0], [1.5, 1.5]]
clusters_std = [0.8, 0.8]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=42, shuffle=False)

# 2. Huấn luyện 2 mô hình SVM: Một cái không trọng số và một cái có 'balanced'
# Dùng kernel='linear' để nhìn đường biên thẳng cho dễ hiểu
clf_no_weight = svm.SVC(kernel='linear', C=1.0)
clf_no_weight.fit(X, y)

clf_balanced = svm.SVC(kernel='linear', class_weight='balanced', C=1.0)
clf_balanced.fit(X, y)

# 3. Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', alpha=0.7)

# Hàm để vẽ đường biên
def plot_svc_decision_boundary(clf, color, label):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # Vẽ đường decision boundary (Z=0)
    ax.contour(XX, YY, Z, colors=color, levels=[0], alpha=0.9, linestyles=['-'])
    # Vẽ nhãn cho legend
    plt.plot([], [], color=color, label=label)

plot_svc_decision_boundary(clf_no_weight, 'red', 'SVM Không trọng số (Bị lấn át)')
plot_svc_decision_boundary(clf_balanced, 'green', 'SVM Balanced (Bảo vệ lớp ít mẫu)')

plt.title("Minh họa cơ chế Class Weight trong SVM")
plt.xlabel("Đặc trưng 1")
plt.ylabel("Đặc trưng 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

output_path = "/Volumes/Toan/ML2/Results_SVM/svm_weight_mechanism.png"
plt.savefig(output_path)
print(f"Đã lưu biểu đồ minh họa tại: {output_path}")
