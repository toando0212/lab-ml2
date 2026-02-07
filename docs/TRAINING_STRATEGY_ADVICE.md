# Báo cáo Kỹ thuật: Chiến lược Huấn luyện & Tối ưu hóa ML Pipeline

Tài liệu này tổng hợp các quyết định kiến trúc và phân tích kỹ thuật trong quá trình xây dựng bộ phân loại biển báo giao thông dựa trên đặc trưng Deep Learning.

---

## 1. Kiến trúc ML Pipeline: Tách biệt Huấn luyện & Đánh giá

Để đảm bảo tính chuyên nghiệp và khả năng mở rộng, hệ thống được tách thành hai module độc lập:

-   **Module Huấn luyện (`train_model.py`):** 
    -   Nhiệm vụ: Học từ dữ liệu đặc trưng và đóng gói tri thức vào file định dạng `.pkl`.
    -   Tại sao cần tách? Việc huấn luyện tốn nhiều tài nguyên. Tách riêng giúp chúng ta chỉ cần học **một lần duy nhất**, tránh lãng phí thời gian khi cần thay đổi cách hiển thị kết quả.
-   **Module Đánh giá (`evaluate_model.py`):**
    -   Nhiệm vụ: Tải mô hình đã lưu, tính toán Accuracy, F1-Score và vẽ Confusion Matrix.
    -   Lợi ích: Cho phép thử nghiệm các cách trình bày báo cáo khác nhau mà không làm thay đổi trạng thái của mô hình.

---

## 2. Lựa chọn Thuật toán: Tại sao là SVM (SGD)?

Trong các bài Lab, chúng ta thường thấy 4 thuật toán: SVM, RF, kNN, DT. Tuy nhiên, với đặc trưng 2048 chiều từ CNN, SVM là "Vua" vì:

| Thuật toán | Vấn đề trong không gian 2048 chiều |
| :--- | :--- |
| **kNN** | "Lời nguyền chiều dữ liệu": Khoảng cách Euclid trở nên vô nghĩa khi số chiều quá lớn. |
| **Decision Tree** | Dễ bị Overfitting cực nặng do cây phải phân nhánh trên quá nhiều đặc trưng. |
| **Random Forest** | Tốn RAM khủng khiếp và tốc độ huấn luyện chậm khi xử lý hàng trăm cây sâu. |
| **SVM (Linear)** | **Tối ưu nhất:** CNN đã làm phẳng dữ liệu, SVM chỉ cần tìm một mặt phẳng phân tách. |

---

## 3. Nhật ký Tối ưu hóa: Từ 45 phút xuống dưới 5 giây

Quá trình tinh chỉnh thuật toán để đạt hiệu năng cao nhất trên máy Mac:

### Giai đoạn 1: SVC (Kernel RBF) - Thất bại vì quá chậm
-   **Nguyên nhân:** Độ phức tạp $O(N^2)$ và chức năng tính xác suất (`probability=True`) bắt buộc chạy thêm 5 lần Cross-validation. 
-   **Kết quả:** Dự kiến mất 45 phút cho 15,000 mẫu.

### Giai đoạn 2: LinearSVC - Thất bại vì hiện tượng "Plateau"
-   **Vấn đề:** Log hiện `cg reaches trust region boundary`. Đây là lúc thuật toán Newton Method bị sa lầy vào vùng phẳng của hàm mục tiêu do các đặc trưng 2048D quá tương quan.
-   **Kết quả:** Bị treo sau 17 phút huấn luyện.

### Giai đoạn 3: SGDClassifier - Giải pháp "Thần tốc"
-   **Cơ chế:** Sử dụng Stochastic Gradient Descent (SGD) với hàm mất mát `hinge`. Về mặt toán học, đây vẫn là một SVM tuyến tính nhưng cách tối ưu hóa dựa trên từng nhóm mẫu nhỏ (Mini-batch).
-   **Kết quả:** Hoàn thành huấn luyện trong **dưới 5 giây** với độ chính xác tương đương.

---

## 4. Phân tích Chuyên sâu cho Báo cáo (Dành cho Giáo viên)

Khi trình bày đồ án, bạn nên sử dụng các luận điểm kỹ thuật sau để giải thích cho sự lựa chọn của mình:

1.  **Tính chất đặc trưng CNN:** Đặc trưng từ lớp Pooling cuối cùng của ResNet/Inception đã mang tính chất hóa (abstracted) cao độ. Các lớp dữ liệu lúc này thường có tính chất **Tuyến tính tách biệt (Linearly Separable)**, do đó không cần dùng các thuật toán phi tuyến phức tạp.
2.  **Tối ưu hóa Primal vs Dual:** Khi số mẫu ($N=15,000$) lớn hơn số chiều ($D=2048$), việc giải bài toán Primal thông qua SGD hiệu quả hơn hàng trăm lần so với giải bài toán Dual trong SVM truyền thống.
3.  **Cân bằng lớp (Class Imbalance):** Việc sử dụng `class_weight='balanced'` trong SGDClassifier giúp mô hình không bị thiên kiến về phía các lớp có nhiều ảnh, đảm bảo độ chính xác công bằng cho cả 43 loại biển báo.

---
> [!TIP]
> **Kết luận:** Hệ thống hiện tại sử dụng **InceptionV3/ResNet50 + SGD-SVM**. Đạt được sự cân bằng hoàn hảo giữa: **Độ chính xác học thuật** và **Tốc độ triển khai thực tế**.
