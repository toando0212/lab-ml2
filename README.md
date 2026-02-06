# Dự án Phân loại Biển báo Giao thông (GTSRB)

Dự án này tập trung vào việc nhận diện và phân loại biển báo giao thông sử dụng bộ dữ liệu German Traffic Sign Recognition Benchmark (GTSRB). Hệ thống áp dụng mô hình **Modular Monolith** với các kỹ thuật trích xuất đặc trưng sâu (Deep Feature Extraction) kết hợp với **SVM (Support Vector Machine)**.

> [!IMPORTANT]
> Dự án đã được tái cấu trúc theo tiêu chuẩn Software Engineering (SE) để đảm bảo tính module hóa và dễ bảo trì.

---

## 🏗️ Cấu trúc Dự án (Modular Architecture)

Dự án được tổ chức thành các khối chuyên biệt để tách biệt logic xử lý và script thực thi:

- `src/`: Chứa mã nguồn logic cốt lõi (Data Loading, Feature Extractors, Analysis Tools).
- `scripts/`: Chứa các script thực thi quy trình (Extraction Runners, Comparision, Training).
- `data/`: Lưu trữ dữ liệu thô (`Dataset/`) và đặc trưng đã trích xuất (`Features/`).
- `docs/`: Tài liệu chi tiết về dự án và các nghiên cứu đối chứng.
- `reports/`: Chứa các báo cáo kết quả, biểu đồ và file trace dữ liệu.
- `labs/`: Lưu trữ các bài tập thực hành cũ (Lab 1 - Lab 4).

---

## 🧠 Ablation Study: ResNet50 vs InceptionV3

Chúng tôi thực hiện nghiên cứu đối chứng để tìm ra bộ trích xuất đặc trưng tối ưu nhất cho biển báo giao thông.

### Kết quả Phân tích PCA (Tính đến hiện tại):
- **Hiệu quả nén**: **ResNet50** vượt trội hơn khi PC1 capture được **20.50%** thông tin (so với 16.34% của InceptionV3).
- **Độ cô đặc**: Đặc trưng của ResNet50 mang tính tập trung cao hơn, giúp mô hình SVM dễ dàng phân loại hơn.
- Chi tiết xem tại: [ABLATION_STUDY_ANALYSIS.md](file:///Volumes/Toan/ML2/docs/ABLATION_STUDY_ANALYSIS.md).

---

## 🛠️ Cài đặt & Sử dụng

Dự án sử dụng [uv](https://github.com/astral-sh/uv) để quản lý môi trường và dependencies một cách tối ưu.

### 1. Khởi tạo môi trường
```bash
uv venv
source .venv/bin/activate  # MacOS/Linux
```

### 2. Cài đặt thư viện
```bash
uv pip install -r pyproject.toml  # Hoặc cài lẻ torch torchvision pandas numpy...
```

### 3. Quy trình thực thi (Pipeline)

Toàn bộ các lệnh được chạy thông qua `uv run` từ thư mục gốc:

```bash
# 1. Trích xuất đặc trưng ResNet50 (Hỗ trợ Apple Silicon MPS)
uv run scripts/run_resnet.py

# 2. Trích xuất đặc trưng InceptionV3
uv run scripts/run_inception.py

# 3. Chạy phân tích đối chứng PCA (Đồ họa & Trace CSV)
uv run scripts/compare_pca_variance.py

# 4. Huấn luyện SVM và báo cáo kết quả
uv run scripts/svm_classification.py
```

---

## 📁 Tài liệu tham khảo quan trọng
- [Hướng dẫn Giảm chiều dữ liệu](file:///Volumes/Toan/ML2/docs/DIMENSION_REDUCTION_GUIDE.md)
- [Cấu trúc Project Chi tiết](file:///Volumes/Toan/ML2/docs/PROJECT_STRUCTURE.md)
- [Quy trình EDA Đặc trưng](file:///Volumes/Toan/ML2/docs/FEATURE_EDA_GUIDE.md)

---

*Người thực hiện: Đỗ Duy Toàn & Tạ Hiếu Nam*
