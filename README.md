# Dự án Phân loại Biển báo Giao thông (GTSRB)

Dự án này tập trung vào việc nhận diện và phân loại biển báo giao thông sử dụng bộ dữ liệu German Traffic Sign Recognition Benchmark (GTSRB). Hệ thống sử dụng các kỹ thuật trích xuất đặc trưng sâu (Deep Feature Extraction) kết hợp với mô hình SVM (Support Vector Machine).

> [!NOTE]
> Repository này chứa cả nội dung từ Lab 1 đến Lab 4. Các file liên quan đến dự án GTSRB được liệt kê chi tiết dưới đây để tránh nhầm lẫn.

## 📂 Thu thập và Tiền xử lý Dữ liệu

- `data_loader.py`: Chịu trách nhiệm load ảnh từ thư mục `Dataset/GTRSB`, hỗ trợ resize và chuẩn hóa dữ liệu.
- `preview_data.py`: Script dùng để xem nhanh các mẫu ảnh trong bộ dữ liệu.
- `plot_class_distribution.py`: Trực quan hóa số lượng mẫu của từng loại biển báo (phân bố lớp).

## 🧠 Trích xuất Đặc trưng (Feature Extraction)

- `feature_extractor.py`: Sử dụng các mô hình Pre-trained (ResNet50, InceptionV3) để chuyển đổi ảnh thành vector đặc trưng.
- `main_extraction.py`: Script thực thi luồng trích xuất đặc trưng và lưu kết quả vào thư mục `Features/`.
- `analyze_features.py`: Phân tích và kiểm tra tính hợp lệ của các đặc trưng đã trích xuất.

## 📉 Giảm chiều và Trực quan hóa

- `dimension_reduction.py`: Áp dụng các thuật toán PCA (Principal Component Analysis) và UMAP để giảm chiều dữ liệu, giúp dễ dàng quan sát cấu trúc dữ liệu.

## 🤖 Huấn luyện và Đánh giá Mô hình

- `svm_classification.py`: Thành phần chính thực hiện huấn luyện mô hình SVM, tối ưu hóa tham số và báo cáo độ chính xác (Accuracy, F1-Score, Confusion Matrix).
- `visualize_svm_mechanism.py`: Trực quan hóa cách mô hình SVM tạo ra các ranh giới phân biệt (decision boundaries) giữa các lớp biển báo.

## 📁 Cấu trúc Thư mục Quan trọng

- `Dataset/GTRSB/`: Nơi chứa ảnh gốc (Đã được cấu hình `.gitignore` để tránh nặng repo).
- `Features/`: Chứa các file `.npy`, `.csv` lưu trữ đặc trưng sau khi trích xuất.
- `Results_SVM/`: Lưu trữ mô hình đã huấn luyện (`.pkl`) và các biểu đồ kết quả đánh giá.
- `Visuals/`: Chứa các hình ảnh trực quan hóa dữ liệu và đặc trưng.

## 🛠️ Cài đặt Môi trường (Setup)

Dự án này khuyến khích sử dụng [uv](https://github.com/astral-sh/uv) để quản lý môi trường nhanh chóng.

### 1. Khởi tạo môi trường ảo

```bash
# Tạo môi trường ảo .venv
uv venv

# Kích hoạt môi trường (MacOS/Linux)
source .venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
uv pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn joblib tqdm pillow
```

## ⚙️ Hướng dẫn Chạy

Bạn có 2 cách để chạy các script trong dự án này:

### Cách 1: Sử dụng `uv run` (Khuyên dùng - Nhanh & Tự động)

Lệnh `uv run` sẽ tự động sử dụng môi trường ảo:

```bash
# Trích xuất đặc trưng
uv run python main_extraction.py

# Huấn luyện mô hình SVM
uv run python svm_classification.py

# Trực quan hóa kết quả
uv run python visualize_svm_mechanism.py
```

### Cách 2: Sử dụng Python truyền thống

Nếu bạn đã kích hoạt môi trường ảo (`source .venv/bin/activate`):

1. **Trích xuất đặc trưng**: `python main_extraction.py`
2. **Huấn luyện mô hình**: `python svm_classification.py`
3. **Trực quan hóa**: `python visualize_svm_mechanism.py`

---

*Người thực hiện: Đỗ Duy Toàn & Tạ Hiếu Nam*
