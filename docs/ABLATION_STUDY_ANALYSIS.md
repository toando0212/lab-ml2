# Phân tích Đối chứng (Ablation Study Analysis)

Tài liệu này ghi lại các kết quả phân tích định lượng khi so sánh hai kiến trúc trích xuất đặc trưng: **ResNet50** và **InceptionV3**.

---

## 1. So sánh Hiệu quả Nén thông tin (PCA Variance)

Mục tiêu của phân tích này là xác định mô hình nào trích xuất được những đặc trưng "giàu thông tin" nhất trong những chiều không gian đầu tiên.

### 1.1. Biểu đồ đối chứng PCA

![Biểu đồ so sánh phương sai tích lũy PCA](/Users/duytoando/.gemini/antigravity/brain/91102d19-5494-46eb-9ec3-bf293fc7d9c8/assets/pca_variance_comparison.png)

### 1.2. Các con số chủ chốt (Key Metrics)

Dựa trên dữ liệu thực tế tại [pca_variance_trace.csv](file:///Volumes/Toan/ML2/reports/figures/pca_variance_trace.csv):

| Chỉ số | ResNet50 | InceptionV3 | Giải thích "bình dân" |
| :--- | :--- | :--- | :--- |
| **PC1 (Hạt nhân)** | **20.50%** | 16.34% | "Trục" thông tin quan trọng nhất. ResNet tóm gọn được nhiều ý chính hơn ngay từ đầu. |
| **PC1-PC10 (Bộ khung)** | **51.77%** | 47.07% | 10 chiều đầu tiên gánh vác khoảng một nửa lượng thông tin. ResNet xây bộ khung vững hơn. |
| **Cumulative @ 200** | **89.94%** | 88.37% | Tốc độ "quét" hết thông tin ảnh. ResNet quét nhanh và sạch hơn. |

### 1.3. Giải thích chuyên sâu cho "người trần mắt thịt"

Để bạn dễ hình dung, hãy tưởng tượng việc trích xuất đặc trưng giống như bạn đang **tóm tắt một cuốn truyện bằng hình ảnh**:

1.  **PC1 (Thành phần chính số 1):** 
    *   Đây là câu tóm tắt quan trọng nhất. ResNet nói: "Đây là một biển báo hình tròn, viền đỏ" (**20.5%** thông tin). Inception nói: "Có cái gì đó màu đỏ và hình tròn" (**16.3%**). 
    *   => Con số của ResNet cao hơn nghĩa là nó tìm được cái "gốc" của vấn đề tốt hơn.

2.  **PC1-PC10 (10 thành phần đầu tiên):**
    *   Giống như việc bạn dùng 10 câu để kể lại cốt truyện. Với ResNet, sau 10 câu bạn đã hiểu được **51.7%** nội dung. Với Inception, bạn mới hiểu được **47.1%**.
    *   => ResNet kể chuyện "có tâm" và súc tích hơn.

3.  **Tại sao lại có sự khác biệt này?**
    *   **ResNet** dùng lối đi tắt (Residual), nó tập trung "nhìn thẳng" vào vật thể chính.
    *   **Inception** dùng nhiều kính lúp soi cùng lúc (đa quy mô). Nó nhìn thấy cả những chi tiết nhỏ nhặt xung quanh, dẫn đến việc thông tin bị "phân tán" ra nhiều trục, không tập trung mạnh vào một vài trục chính như ResNet.

### 1.4. Tầm quan trọng đối với SVM
Vì ResNet nén thông tin tốt hơn (đường biểu diễn dốc hơn), khi bạn đưa dữ liệu này vào **SVM**, thuật toán sẽ dễ dàng tìm ra "đường biên" để phân loại các biển báo hơn. Dữ liệu của ResNet sẽ giúp SVM đạt độ chính xác cao hơn với ít tài nguyên tính toán hơn.

---

## 2. Truy xuất dữ liệu (Traceability)

Để phục vụ việc kiểm chứng và đưa vào phụ lục báo cáo, dữ liệu thô được lưu trữ tại:
*   📊 **Biểu đồ so sánh:** `reports/figures/pca_variance_comparison.png`
*   📄 **Dữ liệu số liệu (CSV):** `reports/figures/pca_variance_trace.csv`

---

## 3. Nhận định cho bước tiếp theo
Dựa trên PCA, ResNet50 đang thắng thế về độ tinh lọc thông tin. Tuy nhiên, PCA chỉ xét tính tuyến tính. Chúng ta cần quan sát **UMAP** (phi tuyến tính) ở bước sau để xem cấu trúc các cụm (clusters) biển báo thực sự tách biệt như thế nào trên không gian 2D.
