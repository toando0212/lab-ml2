# Đề xuất Nâng cao Chiều sâu Thực nghiệm - Dự án GTSRB
## Môn: Machine Learning & Data Mining

Tài liệu này tổng hợp các hướng phát triển nhằm tối ưu hóa kết quả và tăng giá trị học thuật cho dự án cuối kỳ.

---

### 1. Đánh giá hiện trạng (Current Status)
Hiện tại, dự án đã có một quy trình (pipeline) hoàn chỉnh và hiện đại:
- **Feature Extraction:** Sử dụng ResNet50 (Deep Learning) để lấy đặc trưng.
- **Data Analysis:** Có phân tích phân phối lớp (Class Distribution).
- **Visualization:** Sử dụng UMAP để trực quan hóa cấu trúc dữ liệu.
- **Classification:** Sử dụng SVM để phân loại.

> **Đánh giá:** Quy trình đạt chuẩn, mã nguồn sạch và có ứng dụng kỹ thuật mới nhất.

---

### 2. Các hướng nâng cấp đề xuất (Proposed Enhancements)

#### A. So sánh các kiến trúc đặc trưng (Ablation Study)
Việc chỉ sử dụng một loại đặc trưng từ ResNet50 đôi khi chưa đủ để khẳng định tính tối ưu.
- **Thực hiện:** Trích xuất đặc trưng bổ sung từ **InceptionV3** hoặc **MobileNetV2**.
- **Mục tiêu:** So sánh xem kiến trúc mạng neural nào mô tả biển báo giao thông tốt hơn. 
- **Giá trị:** Chứng minh được khả năng lựa chọn mô hình dựa trên thực nghiệm (Evidence-based selection).

#### B. Tối ưu hóa tham số mô hình (Hyperparameter Tuning)
Mô hình SVM rất nhạy cảm với các tham số `C` và `gamma`.
- **Thực hiện:** Sử dụng `GridSearchCV` để quét qua các giá trị như:
    - `C`: [0.1, 1, 10, 100]
    - `gamma`: ['scale', 'auto', 0.1, 0.01]
- **Mục tiêu:** Tìm ra "điểm ngọt" (Sweet spot) để đạt Accuracy cao nhất mà không bị Overfitting.
- **Giá trị:** Thể hiện kỹ năng tinh chỉnh mô hình chuyên nghiệp.

#### C. Đối chứng với phương pháp truyền thống (Traditional Baseline)
Trong Data Mining, việc so sánh với các phương pháp cổ điển rất quan trọng để thấy được giá trị của kỹ thuật mới.
- **Thực hiện:** Trích xuất đặc trưng **HOG (Histogram of Oriented Gradients)** - phương pháp kinh điển trong nhận diện vật thể.
- **So sánh:** Chạy SVM trên HOG Features vs ResNet Features.
- **Giá trị:** Làm nổi bật ưu điểm của Transfer Learning trong xử lý ảnh.

#### D. Phân tích lỗi chuyên sâu (Error Analysis)
Đây là phần thường bị bỏ qua nhưng lại mang tính "Khai phá dữ liệu" cao nhất.
- **Thực hiện:** 
    1. Trích xuất danh sách các ảnh bị dự đoán sai nhiều nhất.
    2. Phân loại lỗi: Lỗi do ảnh tối (Low light), lỗi do biển báo bị che khuất (Occlusion), lỗi do các lớp quá giống nhau (Similarity).
- **Giá trị:** Cho thấy sự am hiểu sâu sắc về dữ liệu, không chỉ dừng lại ở các con số vô hồn.

#### E. Xử lý mất cân bằng dữ liệu (Imbalanced Data Handling)
Bộ GTSRB có những lớp rất ít mẫu (Class 0, 19, 37...) và có lớp rất nhiều mẫu.
- **Thực hiện:** So sánh giữa việc dùng `class_weight='balanced'` hiện tại với kỹ thuật **SMOTE** (Synthetic Minority Over-sampling Technique).
- **Giá trị:** Giải quyết bài toán thực tế phổ biến trong Data Mining.

---

### 3. Lộ trình thực hiện (Roadmap)

1.  **Tuần 1:** Chạy `GridSearchCV` cho SVM hiện tại (Task dễ, hiệu quả ngay).
2.  **Tuần 2:** Thực hiện Error Analysis (Phần quan trọng nhất cho báo cáo).
3.  **Tuần 3:** Trích xuất HOG Features hoặc thêm 1 Backbone (Inception) để so sánh.
4.  **Tuần 4:** Hoàn thiện báo cáo với các biểu đồ so sánh (Bar chart, Confusion Matrix so sánh).

---
*Hy vọng các đề xuất này sẽ giúp nhóm đạt kết quả cao nhất!*
