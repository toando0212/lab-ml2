# Báo cáo Phân tích Kỹ thuật: Đặc trưng & Phân cụm (Technical Feature Analysis)

Tài liệu này tập trung vào các thông số định lượng và chiều không gian dữ liệu được sử dụng trong quá trình đối chứng ResNet50 và InceptionV3.

---

## 1. Phân tích Chiều dữ liệu (Dimension Analysis)

Điểm cốt lõi của Ablation Study này là đánh giá khả năng nén và biểu diễn thông tin từ không gian cực cao về không gian thấp.

### 1.1. Không gian đặc trưng gốc (Raw Feature Space)
- **Đầu vào:** Tất cả các phân tích định lượng (Clustering Metrics) và giảm chiều (UMAP/PCA) đều bắt đầu từ vector thô:
    - **Số chiều (Dimensions):** **2,048** (Trích xuất từ lớp Global Average Pooling).
    - **Nguồn:** ResNet50 và InceptionV3 (Pre-trained on ImageNet).

### 1.2. Phân tích Thành phần chính (PCA Variance)
- **Phạm vi tính toán:** Thực hiện trên **200 thành phần (components)** đầu tiên.
- **Mục tiêu:** Đo lường lượng thông tin bảo toàn khi nén từ 2048 chiều xuống 200 chiều.
- **Thông số kỹ thuật:**
    - PCA nén dữ liệu dựa trên phương sai (Variance) của 2048 chiều gốc.
    - Kết quả cho thấy ResNet50 cô đặc thông tin hiệu quả hơn (PC1 gánh vác nhiều trọng trách hơn InceptionV3).

### 1.3. Giảm chiều Phi tuyến (UMAP Projection)
- **Cơ chế:** Nén trực tiếp từ **2,048 chiều -> 3 chiều** (để vẽ 3D).
- **Khác biệt với PCA:** PCA là phép chiếu tuyến tính (giữ khoảng cách lớn), còn UMAP là bảo toàn cấu trúc lân cận (Topology). UMAP uốn nắn không gian 2048 chiều để gom các ảnh giống nhau lại gần nhau trong không gian 3D.

---
∏
## 2. Đo lường Định lượng (Quantitative Metrics)

Các chỉ số này được tính toán trực tiếp trên **không gian 2,048 chiều** thô để đảm bảo độ khách quan (không bị ảnh hưởng bởi quá trình giảm chiều của UMAP/PCA).

| Chỉ số (Metric) | ResNet50 (2048-d) | InceptionV3 (2048-d) | Ý nghĩa Kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | **-0.0084** | -0.0116 | Đo độ trùng lặp giữa 43 cụm biển báo trong không gian 2048-chiều. |
| **Davies-Bouldin** | **6.6661** | 6.8892 | Tỷ số giữa độ rộng cụm và khoảng cách giữa các cụm. |

> [!IMPORTANT]
> **Tại sao Silhouette Score âm?** Trong không gian gốc 2048 chiều, các biển báo có độ tương đồng rất cao (ví dụ các biển báo giới hạn tốc độ chỉ khác nhau con số). Điểm số này cho thấy dữ liệu cực kỳ phức tạp và đan xen. Tuy nhiên, ResNet50 có điểm số "ít âm" hơn và chỉ số DB nhỏ hơn, chứng minh nó tạo ra các cụm **tách biệt tốt hơn** InceptionV3 trước khi đi vào SVM.

---

## 3. Tổng kết Thông số cấu hình

Để tái lập (reproducible) các kết quả trong báo cáo, các tham số sau đã được sử dụng:

- **PCA Components:** 200 (Dùng cho Scree Plot).
- **UMAP Neighbors:** 15 (Số lượng láng giềng để xây dựng đồ thị lân cận).
- **UMAP Min Dist:** 0.1 (Khoảng cách tối thiểu giữa các điểm trong không gian 3D).
- **Sample Size (Metrics):** 5,000 mẫu ngẫu nhiên (Để đảm bảo tính hội tụ khi tính Silhouette trên 2048 chiều).

---
## 4. Truy xuất Nguồn dữ liệu (Data Lineage)
- Dữ liệu thô (Features): `data/Features/Variants/` (14,670 x 2048).
- Kết quả PCA: `reports/figures/pca_variance_trace.csv`.
- Kết quả Metrics: `reports/cluster_metrics_comparison.csv`.
