# Thử Nghiệm K-means trên Dataset Mice Protein

## Tổng Quan

Thử nghiệm k-means clustering trên dataset Mice Protein Expression với 1,080 mẫu, 77 protein features, và 8 lớp (dựa trên Genotype, Treatment, Behavior).

## Files Trong Thư Mục

- `kmeans_experiments.py` - Script chính để chạy thử nghiệm
- `kmeans_results.csv` - Bảng tổng hợp kết quả các metrics
- `elbow_curve.png` - Đồ thị Elbow Method
- `internal_metrics.png` - Đồ thị các internal metrics
- `external_metrics.png` - Đồ thị các external metrics
- `pca_visualization.png` - Trực quan hóa PCA 2D

## Kết Quả Chính

### Bảng Tóm Tắt Metrics

| k  | Silhouette | Davies-Bouldin | NMI    | ARI    |
|----|------------|----------------|--------|--------|
| 2  | **0.143**  | 2.199          | 0.035  | 0.021  |
| 3  | 0.129      | 2.176          | 0.110  | 0.074  |
| 4  | 0.121      | 2.003          | 0.185  | 0.114  |
| 5  | 0.115      | 2.050          | 0.205  | 0.115  |
| 6  | 0.121      | 1.850          | 0.221  | 0.119  |
| 7  | 0.124      | 1.866          | 0.228  | 0.128  |
| 8  | 0.134      | 1.810          | 0.255  | 0.136  |
| 9  | 0.132      | 1.794          | 0.257  | 0.123  |
| 10 | 0.128      | 1.817          | 0.273  | 0.132  |
| 12 | 0.136      | 1.788          | 0.301  | 0.141  |
| 15 | 0.135      | **1.787**      | **0.345** | **0.150** |

### K Tốt Nhất Theo Từng Metric

- **Silhouette Score** (cao nhất): k = 2
- **Davies-Bouldin Index** (thấp nhất): k = 15
- **NMI** (cao nhất): k = 15
- **ARI** (cao nhất): k = 15

## Phân Tích Kết Quả

### 1. Mâu Thuẫn Giữa Internal và External Metrics

**Quan sát quan trọng:**
- Internal metrics (Silhouette) cho rằng k=2 là tốt nhất
- External metrics (NMI, ARI, Davies-Bouldin) cho rằng k=15 là tốt nhất
- Dataset có 8 classes thực tế, nhưng k=8 không phải là tối ưu

**Giải thích:**
- k=2 tạo ra 2 cụm lớn, rất tách biệt (Silhouette cao)
- k=15 phân chia chi tiết hơn, khớp tốt hơn với cấu trúc phức tạp của data
- Dataset có cấu trúc phân cấp (Genotype × Treatment × Behavior)

### 2. Hiệu Quả Clustering

**Với k=8 (số lớp thực tế):**
- ✅ **NMI = 0.255**: Trung bình - có một số thông tin chung với ground truth
- ⚠️ **ARI = 0.136**: Thấp - clustering khác khá nhiều so với ground truth
- ⚠️ **Homogeneity = 0.248**: Thấp - clusters không thuần khiết
- ⚠️ **Completeness = 0.264**: Thấp - classes không tập trung
- ⚠️ **Silhouette = 0.134**: Thấp - có overlap đáng kể giữa clusters

**Với k=15 (tốt nhất theo external metrics):**
- ✅ **NMI = 0.345**: Khá tốt - thông tin chung cao hơn
- ✅ **ARI = 0.150**: Cải thiện - tương đồng tốt hơn với ground truth
- ✅ **Homogeneity = 0.389**: Khá tốt - clusters thuần khiết hơn
- ⚠️ **Completeness = 0.310**: Trung bình - classes vẫn bị phân tán

**Kết luận:**
- K-means gặp khó khăn với dataset này (ARI < 0.2)
- Dataset có cấu trúc phức tạp, không phù hợp với giả định spherical clusters
- Cần nhiều clusters hơn số lớp thực tế để capture được cấu trúc

### 3. Xu Hướng Theo k

**Khi k tăng từ 2 → 15:**
- Inertia (SSE): Giảm liên tục (từ 62K → 39K)
- Silhouette: Giảm nhẹ sau k=2, ổn định ở ~0.13
- NMI: Tăng liên tục (từ 0.035 → 0.345)
- ARI: Tăng liên tục (từ 0.021 → 0.150)
- Davies-Bouldin: Giảm dần (từ 2.2 → 1.8)

**Không có elbow rõ ràng:**
- Inertia giảm đều đặn, không có điểm "khuỷu tay"
- Cho thấy dữ liệu không có cấu trúc cluster tự nhiên rõ ràng

### 4. Thách Thức

1. **Missing Values**: Dataset có ~2.5% missing values (đã xử lý bằng mean imputation)
2. **High Dimensionality**: 77 features → curse of dimensionality
3. **Complex Structure**: 8 classes từ 3 factors (Genotype × Treatment × Behavior)
4. **Overlap**: Các protein expressions có thể overlap giữa các nhóm
5. **Imbalanced Classes**: Phân bố không đều giữa các lớp

## Khuyến Nghị

### Để Cải Thiện Kết Quả:

1. **Dimensionality Reduction**:
   - Áp dụng PCA trước khi clustering (giữ 95% variance)
   - Thử t-SNE hoặc UMAP để giảm chiều phi tuyến
   - Feature selection dựa trên protein importance

2. **Alternative Algorithms**:
   - **GMM** (Gaussian Mixture Models): Cho phép clusters hình ellipse
   - **Hierarchical Clustering**: Phát hiện cấu trúc phân cấp (Genotype → Treatment → Behavior)
   - **DBSCAN**: Không cần chỉ định k trước, tìm clusters tự nhiên
   - **Spectral Clustering**: Tốt cho non-convex clusters

3. **Feature Engineering**:
   - Normalize protein expressions theo từng nhóm
   - Tạo features tương tác giữa các proteins
   - Log transformation cho skewed distributions

4. **Ensemble Methods**:
   - Consensus clustering từ nhiều lần chạy
   - Combine multiple algorithms

5. **Supervised Learning**:
   - Vì có ground truth labels, có thể dùng supervised methods
   - Random Forest, SVM, Neural Networks có thể cho kết quả tốt hơn

## So Sánh với ISOLET

| Aspect | ISOLET | Mice Protein |
|--------|--------|--------------|
| **Samples** | 6,239 | 1,080 |
| **Features** | 617 | 77 |
| **Classes** | 26 | 8 |
| **Best k (Silhouette)** | 2 | 2 |
| **Best k (External)** | 26 | 15 |
| **NMI @ best k** | 0.731 | 0.345 |
| **ARI @ best k** | 0.522 | 0.150 |
| **Clustering Quality** | Khá tốt | Kém |

**Nhận xét:**
- ISOLET có kết quả tốt hơn nhiều (NMI=0.73 vs 0.35)
- Mice Protein khó cluster hơn do cấu trúc phức tạp
- ISOLET có số lớp = số clusters tối ưu, Mice Protein không

## Cách Chạy Lại

```bash
# Di chuyển đến thư mục
cd /Volumes/Toan/ML2/lab2/mice_protein

# Chạy thử nghiệm
uv run python kmeans_experiments.py
```

## Tài Liệu Tham Khảo

- Dataset: [Mice Protein Expression - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)
- Paper: Higuera et al. (2015) "Self-organizing feature maps identify proteins critical to learning in a mouse model of Down syndrome"
