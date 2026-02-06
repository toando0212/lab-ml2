# Hướng dẫn Giảm chiều Dữ liệu: PCA vs UMAP

Tài liệu này giải thích lý do và cách thức sử dụng PCA và UMAP để phân tích các đặc trưng (features) trích xuất từ mô hình CNN cho bộ dữ liệu GTSRB.

---

## 1. PCA (Principal Component Analysis)
**Tên gọi:** Phân tích Thành phần Chính.
**Mục đích trong dự án:** Nén dữ liệu "thô" để giảm nhiễu và tăng tốc tính toán.

### Đặc điểm:
*   **Tuyến tính:** Tìm ra các trục (components) mang lại biến thiên lớn nhất.
*   **Tốc độ:** Cực nhanh (thích hợp xử lý hàng triệu tham số).
*   **Tầm nhìn:** Giữ tốt cấu trúc tổng thể (Global structure) nhưng yếu trong việc tách biệt các cụm phức tạp.

---

## 2. UMAP (Uniform Manifold Approximation and Projection)
**Mục đích trong dự án:** Trực quan hóa dữ liệu để chứng minh chất lượng đặc trưng.

### Đặc điểm:
*   **Phi tuyến:** Mô phỏng dữ liệu trên các mặt cong phức tạp.
*   **Gom cụm:** Cực kỳ mạnh mẽ. Giúp tách biệt rõ rệt 10 loại biển báo thành 10 "đảo" màu sắc khác nhau.
*   **Tầm nhìn:** Giữ tốt mối quan hệ cục bộ (Local structure) - các ảnh giống nhau sẽ ở gần nhau.

---

## 3. Chiến lược áp dụng (Tư duy Senior)

Để đạt hiệu quả tốt nhất và tăng độ chuyên nghiệp cho báo cáo, chúng ta sẽ thực hiện quy trình sau:

1.  **Bước 1 (Preprocessing):** Dùng PCA nén từ **2048 chiều** xuống còn **50 chiều**.
    *   *Tại sao?* Để loại bỏ nhiễu và làm cho các bước sau chạy nhanh hơn 10 lần.
2.  **Bước 2 (Visualization):** Dùng UMAP nén từ **50 chiều** xuống **2 chiều**.
    *   *Tại sao?* Để vẽ lên biểu đồ (Scatter Plot).
3.  **Bước 3 (Validation):** Vẽ biểu đồ UMAP cho ResNet và Inception đặt cạnh nhau.
    *   *Kết quả:* Mô hình nào có các cụm màu tách rời nhau hơn, mô hình đó trích xuất đặc trưng "tách lớp" tốt hơn.

---

## 4. Tóm tắt so sánh

| Đặc điểm | PCA | UMAP |
| :--- | :--- | :--- |
| **Loại** | Tuyến tính | Phi tuyến |
| **Tốc độ** | Rất nhanh | Bình thường |
| **Gom cụm** | Yếu | Rất mạnh |
| **Công dụng** | Nén dữ liệu, giảm nhiễu | Trực quan hóa, báo cáo |

> [!TIP]
> Luôn chạy PCA trước khi chạy UMAP để tiết kiệm bộ nhớ RAM và thời gian tính toán của máy Mac.
