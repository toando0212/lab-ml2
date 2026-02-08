# Kế hoạch Nâng cấp Báo cáo Dự án (Report Improvement Plan)

Để báo cáo đạt chất lượng dự án cuối kỳ (Final Project) với độ dài khoảng 25-30+ trang, chúng ta sẽ triển khai các hạng mục sau:

## 1. Mở rộng Phân tích Dữ liệu (Deep EDA)
- **Hình ảnh minh họa:** Chèn mẫu ảnh thực tế của 10 lớp biển báo (để người đọc hình dung độ khó).
- **Phân tích phân phối:** Biểu đồ số lượng mẫu trên mỗi lớp (đánh giá tính cân bằng/imbalance).
- **Đặc điểm vật lý:** Phân tích độ phân giải, ánh sáng, và các biến thể hình học trong tập GTSRB.

## 2. Chi tiết hóa Kiến trúc Mô hình (Technical Deep-Dive)
- **CNN Architectures:** Giải thích chuyên sâu về ResNet50 (Residual Learning) và InceptionV3 (Multi-scale convolutions).
- **Feature Extraction:** Giải thích tại sao chọn lớp Global Average Pooling và tại sao ra vector 2048 chiều.
- **Sơ đồ khối:** Thêm các sơ đồ biểu diễn luồng dữ liệu (Data Pipeline).

## 3. Nâng tầm Phân tích Đối chứng (Ablation Study)
- **PCA vs UMAP:** Viết tiểu mục so sánh ưu/nhược điểm của phương pháp Tuyến tính và Phi tuyến.
- **Metric Phân cụm:** Đưa các chỉ số Silhouette Score và Davies-Bouldin vào báo cáo chính thức để tăng tính hàn lâm.

## 4. Phân tích lỗi và Ca thất bại (Failure Case Analysis)
- **Visualizing Errors:** Trích xuất các ảnh bị đoán sai (Miss-classified images).
- **Root Cause Analysis:** Giải thích lý do sai (mờ, nhầm lẫn giữa 30km/h và 50km/h, biển báo bị khuất...).

## 5. Kỹ thuật Hệ thống (System Engineering)
- **Môi trường:** Chi tiết hardware (GPU, CPU, RAM) và danh sách thư viện (pip freeze/requirements).
- **Cấu trúc Source Code:** Giải thích cách tổ chức code theo mô hình Modular để dễ bảo trì.

---
**Mục tiêu:** Chuyển đổi báo cáo từ hình thức "Bài Lab" sang "Dự án Nghiên cứu" toàn diện.
