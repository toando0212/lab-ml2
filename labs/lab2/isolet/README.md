# Thử Nghiệm K-means trên Dataset ISOLET

## Tổng Quan

Thử nghiệm k-means clustering trên dataset ISOLET (Isolated Letter Speech Recognition) với 6,239 mẫu, 617 đặc trưng, và 26 lớp (chữ cái A-Z).

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
| 2  | **0.142**  | **2.353**      | 0.269  | 0.060  |
| 5  | 0.098      | 2.630          | 0.500  | 0.183  |
| 10 | 0.091      | 2.439          | 0.612  | 0.290  |
| 15 | 0.083      | 2.510          | 0.668  | 0.407  |
| 20 | 0.085      | 2.547          | 0.702  | 0.410  |
| 26 | 0.074      | 2.745          | **0.731** | **0.522** |
| 30 | 0.073      | 2.739          | 0.709  | 0.460  |
| 35 | 0.072      | 2.821          | 0.716  | 0.463  |
| 40 | 0.064      | 2.879          | 0.696  | 0.440  |

### K Tốt Nhất Theo Từng Metric

- **Silhouette Score** (cao nhất): k = 2
- **Davies-Bouldin Index** (thấp nhất): k = 2
- **NMI** (cao nhất): k = 26
- **ARI** (cao nhất): k = 26

## Phân Tích Kết Quả

### 1. Internal Metrics vs External Metrics

**Mâu thuẫn quan trọng:**
- Internal metrics (Silhouette, Davies-Bouldin) cho rằng k=2 là tốt nhất
- External metrics (NMI, ARI) cho rằng k=26 là tốt nhất

**Giải thích:**
- Internal metrics đo "độ tách biệt" của clusters mà không biết ground truth
- k=2 tạo ra 2 cụm lớn, rất tách biệt nhưng không phản ánh 26 lớp thực tế
- k=26 khớp với số lớp thực tế, có NMI=0.731 và ARI=0.522 (khá tốt)

### 2. Đánh Giá Hiệu Quả K-means

**Với k=26 (số lớp thực tế):**
- ✅ **NMI = 0.731**: Tốt - có thông tin chung cao với ground truth
- ✅ **ARI = 0.522**: Khá tốt - clustering có độ tương đồng vừa phải
- ✅ **Homogeneity = 0.722**: Tốt - clusters khá "thuần khiết"
- ✅ **Completeness = 0.740**: Tốt - các class khá "tập trung"
- ⚠️ **Silhouette = 0.074**: Thấp - clusters có overlap đáng kể

**Kết luận:**
- K-means có thể phát hiện được cấu trúc 26 lớp với độ chính xác vừa phải
- Có sự overlap đáng kể giữa các chữ cái (do phát âm tương tự)
- Một số chữ cái khó phân biệt: B/D/E, M/N, F/S, v.v.

### 3. Xu Hướng Theo k

**Khi k tăng từ 2 → 40:**
- Inertia (SSE): Giảm liên tục (từ 3.2M → 2.0M)
- Silhouette: Giảm dần (từ 0.142 → 0.064)
- NMI: Tăng đến k=26 (0.731), sau đó giảm nhẹ
- ARI: Tăng đến k=26 (0.522), sau đó giảm

**Điểm "Elbow":**
- Không có elbow rõ ràng trong Inertia curve
- Điều này cho thấy dữ liệu không có cấu trúc cluster tự nhiên rõ ràng

### 4. Thách Thức

1. **High Dimensionality**: 617 features → curse of dimensionality
2. **Overlap Classes**: Nhiều chữ cái có đặc trưng âm thanh tương tự
3. **Spherical Assumption**: K-means giả định clusters hình cầu, có thể không phù hợp
4. **Sensitivity**: Kết quả phụ thuộc vào khởi tạo centroids

## Khuyến Nghị

### Để Cải Thiện Kết Quả:

1. **Dimensionality Reduction**:
   - Áp dụng PCA trước khi clustering (giữ 95% variance)
   - Thử t-SNE hoặc UMAP để giảm chiều phi tuyến

2. **Alternative Algorithms**:
   - GMM (Gaussian Mixture Models): Cho phép clusters hình ellipse
   - Hierarchical Clustering: Phát hiện cấu trúc phân cấp
   - DBSCAN: Không cần chỉ định k trước

3. **Feature Engineering**:
   - Chọn subset features quan trọng nhất
   - Thử các feature transformations khác

4. **Ensemble Methods**:
   - Kết hợp nhiều lần chạy k-means
   - Consensus clustering

## Cách Chạy Lại

```bash
# Di chuyển đến thư mục
cd /Volumes/Toan/ML2/lab2/isolet

# Chạy thử nghiệm
uv run python kmeans_experiments.py
```

## Tài Liệu Tham Khảo

- Dataset: [UCI ISOLET](https://archive.ics.uci.edu/ml/datasets/ISOLET)
- Giao thức thực nghiệm chi tiết: Xem `experimental_protocol.md` trong artifacts
- Metrics: Scikit-learn Clustering Metrics Documentation
