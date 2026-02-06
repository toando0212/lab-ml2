# Hướng dẫn Phân tích & So sánh Đối chứng (ResNet50 vs InceptionV3)

Tài liệu này hướng dẫn bạn thực hiện quy trình EDA và So sánh đối chứng (Ablation Study) sau khi đã trích xuất xong đặc trưng từ hai kiến trúc mạng Neural khác nhau.

---

## 1. Phân tích nền tảng: Phân phối lớp (Class Distribution)

Trước khi so sánh hai mô hình, chúng ta cần hiểu về "nguyên liệu" đầu vào. Vì cả ResNet và Inception đều dùng chung tập dữ liệu, đây là bước phân tích chung.

*   **Mục tiêu:** Kiểm tra sự cân bằng của dữ liệu (Data Balance).
*   **Kỹ thuật:** Vẽ biểu đồ cột (Count Plot) cho 10 lớp ClassId.
*   **Giá trị báo cáo:** Chỉ ra lớp nào có nhiều ảnh nhất, lớp nào ít nhất. Nếu mô hình đoán sai ở lớp ít dữ liệu, bạn có bằng chứng để giải thích là do "thiếu dữ liệu" chứ không hẳn do mô hình dở.

---

## 2. Quy trình So sánh Đối chứng (Ablation Study)

Đây là phần trọng tâm để bạn so sánh bộ đặc trưng của ResNet50 và InceptionV3.

### Bước 2.1: So sánh hiệu quả nén thông tin (PCA Scree Plot)

*   **Cách làm:** Vẽ biểu đồ đường *Cumulative Explained Variance* cho cả ResNet và Inception trên **cùng một đồ thị**.
*   **Điểm cần soi:** Đường nào dốc lên nhanh hơn?
    *   Nếu ResNet đạt 90% thông tin chỉ với 40 chiều, còn Inception cần 60 chiều -> ResNet trích xuất đặc trưng "cô đặc" hơn.

### Bước 2.2: So sánh khả năng phân cụm (UMAP Side-by-Side)

*   **Cách làm:** Vẽ hai biểu đồ Scatter Plot 2D cạnh nhau.
    *   **Trái:** Đặc trưng ResNet sau khi qua UMAP.
    *   **Phải:** Đặc trưng Inception sau khi qua UMAP.
*   **Điểm cần soi:**
    *   Ở bên nào các cụm màu tách rời nhau rõ rệt hơn?
    *   Có lớp nào bị "dính chùm" ở ResNet nhưng lại tách ra được ở Inception không? (Ví dụ: Biển báo 60km/h và 80km/h).

### Bước 2.3: So sánh mẫu khó (Edge Case Analysis)

*   **Cách làm:** Tìm những ảnh bị mô hình đoán sai (Outliers) trên biểu đồ UMAP.
*   **Điểm cần soi:**
    *   Với những ảnh biển báo ở xa (nhỏ), Inception (multi-scale) có gom chúng vào đúng cụm tốt hơn ResNet không?
    *   Với những ảnh bị tối/mờ, ResNet (deep residual) có xử lý ổn định hơn không?

### Bước 2.4: So sánh Tài nguyên & Thời gian

*   **Số liệu:** Ghi lại thời gian trích xuất (Inception: 150.67s, ResNet: *Sắp chạy*).
*   **Nhận xét:** Đánh giá xem sự chênh lệch thời gian có xứng đáng với sự chênh lệch về độ tách biệt lớp hay không.

---

## 3. Bảng tổng hợp so sánh (Dùng cho Báo cáo)

| Tiêu chuẩn so sánh | ResNet50 (Baseline) | InceptionV3 (Multi-scale) | Kết luận |
| :--- | :--- | :--- | :--- |
| **Số lượng ảnh (10 lớp)** | 52,000 (giả định) | 52,000 (giả định) | Chung dữ liệu |
| **PCA (để giữ 90% var)** | *Điền số chiều* | *Điền số chiều* | ... |
| **Độ tách cụm UMAP** | *Tốt/Trung bình* | *Tốt/Trung bình* | ... |
| **Thời gian trích xuất** | *Đang đợi* | 150.67s | ... |

> [!TIP]
> Việc trình bày được sự đối chiếu này trong đồ án sẽ giúp bạn chứng minh được tư duy phân tích hệ thống (Systematic Analysis), là điểm khác biệt lớn nhất giữa một lập trình viên và một nhà khoa học dữ liệu.
