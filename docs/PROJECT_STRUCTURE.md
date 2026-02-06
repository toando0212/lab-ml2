# Cấu trúc Dự án (Project Structure)

Dự án được tổ chức theo tiêu chuẩn kỹ thuật phần mềm (Software Engineering) để đảm bảo tính module hóa, dễ bảo trì và mở rộng, tuân theo tinh thần Modular Monolith.

---

## 1. Sơ đồ cấu trúc

```text
.
├── data/               # Quản lý dữ liệu
│   ├── Dataset/        # Dữ liệu ảnh thô (Raw Data)
│   └── Features/       # Đặc trưng đã trích xuất (Processed Data)
├── src/                # Logic cốt lõi (Core Logic)
│   ├── data_loader.py  # Xử lý nạp và tiền xử lý ảnh
│   └── extractors/     # Các module trích xuất (CNN Base)
│       ├── base.py     # Base class cho Extractor (MPS support)
│       ├── resnet.py   
│       └── inception.py
├── scripts/            # Kịch bản thực thi (Runners & EDA)
│   ├── run_resnet.py   # Chạy trích xuất đặc trưng ResNet
│   ├── run_inception.py
│   ├── analyze_eda.py  # Các script vẽ biểu đồ PCA/UMAP
│   └── train_svm.py    # Huấn luyện và đánh giá SVM
├── docs/               # Tài liệu hướng dẫn và chiến lược
├── labs/               # Lưu trữ các bài Lab cũ
└── README.md           # Hướng dẫn chung
```

---

## 2. Nguyên tắc tổ chức

1.  **Tách biệt logic (Separation of Concerns):**
    *   Thư mục `src/` chỉ chứa code logic không thay đổi (Class, Function).
    *   Thư mục `scripts/` chứa code có tính thực thi (Scripts) thường xuyên thay đổi tham số.
2.  **Quản lý Import:**
    *   Tất cả các script trong `scripts/` sẽ gọi logic từ `src/`.
3.  **Dữ liệu bất biến:**
    *   Dữ liệu trong `data/Dataset/` là read-only, không được script nào chỉnh sửa trực tiếp vào gốc.

---

## 3. Lợi ích cho Báo cáo
Việc đưa sơ đồ cấu trúc này vào báo cáo sẽ chứng minh bạn không chỉ làm AI/ML theo kiểu "viết đâu chạy đấy" mà có tư duy xây dựng một hệ thống phần mềm (Systematic Software Building). Thầy cô sẽ đánh giá rất cao khả năng đóng gói (Encapsulation) của bạn.
