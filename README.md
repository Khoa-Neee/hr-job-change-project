# HR Job Change Prediction (NumPy from scratch)

Dự án dự đoán khả năng ứng viên “đổi việc” dựa trên dữ liệu nhân sự, triển khai Logistic Regression từ đầu bằng NumPy, kèm pipeline tiền xử lý, trực quan hóa và đánh giá mô hình.

---

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Method](#method)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

---

## Giới thiệu
- Bài toán: Dự đoán xác suất/khả năng ứng viên sẽ đổi công việc trong tương lai gần (nhị phân: 0/1).
- Động lực/ứng dụng: Hỗ trợ bộ phận nhân sự (HR) nhận diện sớm nhóm có rủi ro rời đi để có chính sách giữ chân (retention), tối ưu tuyển dụng và đào tạo.
- Mục tiêu cụ thể: Xây dựng pipeline end-to-end bằng NumPy thuần, gồm: xử lý dữ liệu, tạo đặc trưng, huấn luyện Logistic Regression, đánh giá bằng các chỉ số phân loại và trực quan hóa kết quả.

---

## Dataset
- Nguồn: “HR Analytics: Job Change of Data Scientists” (Kaggle). Tệp chính trong thư mục `data/raw/`:
  - `aug_train.csv` (có nhãn `target`)
  - `aug_test.csv` (không có nhãn — dùng tham khảo/so sánh)
  - Link: Dataset: https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists

- Một số cột tiêu biểu:
  - `city`, `city_development_index` (CDI), `gender`, `relevent_experience`, `enrolled_university`, `education_level`, `major_discipline`, `experience`, `company_size`, `company_type`, `training_hours`, `last_new_job`, `target`.

- Đặc điểm dữ liệu nổi bật (xem chi tiết trong notebook 01/02):
  - Nhiều biến phân loại (categorical), giá trị thiếu và giá trị đặc biệt (vd: `experience` chứa `>20`, `<1`, `never`).
  - `city` có số lượng giá trị rất lớn → dùng Label Encoding thay vì One-hot.
  - `CDI` có tương quan mạnh với `target` → hữu ích cho mô hình.

Lưu ý: Số lượng mẫu/đặc trưng hiển thị trực tiếp khi chạy notebook preprocessing; con số có thể khác nhau tùy cấu hình features.

---

## Method
### Quy trình xử lý dữ liệu
1. Khám phá dữ liệu cơ bản (EDA) trong `notebooks/01_data_exploration.ipynb`.
2. Tiền xử lý trong `notebooks/02_preprocessing.ipynb`:
   - Làm sạch và chuẩn hóa các cột numeric (`experience` impute theo median; chuẩn hóa z-score).
   - Mã hóa categorical: one-hot cho nhóm cột chọn lọc; label encode cho `city` (nhiều giá trị).
   - Feature engineering: tương tác (`CDI * experience`), bậc hai (`CDI^2`, `experience^2`), tỉ lệ và log-transform.
   - Thêm cột bias vào ma trận đặc trưng.
   - Chia dữ liệu theo stratified split 80/20 → lưu `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`.

### Thuật toán (Logistic Regression – NumPy)
- Mô hình: $\hat{y} = \sigma(Xw)$, với $\sigma(z) = 1/(1+e^{-z})$.
- Hàm mất mát (Binary Cross-Entropy):
  $$J(w) = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\big]$$
- Gradient: $$\nabla_w J = \frac{1}{n} X^\top (\hat{y} - y)$$
- Tối ưu bằng Gradient Descent; dừng sớm khi $|J_{t}-J_{t-1}| < \text{tol}$.
- Biện pháp ổn định số: clip đầu vào sigmoid và xác suất trong log để tránh overflow/underflow.

Chi tiết cài đặt nằm trong cell class `LogisticRegression` của `notebooks/03_modeling.ipynb`.

---

## Installation & Setup
Yêu cầu: Python 3.10+.

Trên Windows (PowerShell):

```pwsh
# (Tùy chọn) Tạo môi trường Conda
conda create -n hr-numpy python=3.10 -y
conda activate hr-numpy

# Cài thư viện
pip install -r hr-job-change-project/requirements.txt

# Mở Jupyter Lab hoặc VS Code
jupyter lab
# hoặc
code .
```

Nếu không dùng Conda, có thể dùng `python -m venv .venv` và kích hoạt venv tương ứng trên Windows.

---

## Usage
Thứ tự đề xuất:
1. `notebooks/01_data_exploration.ipynb` – khám phá và trực quan hóa dữ liệu thô.
2. `notebooks/02_preprocessing.ipynb` – tiền xử lý + lưu dữ liệu đã chuẩn hóa và tách tập:
   - Kết quả trong `data/processed/`: `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, `preprocessing_info.pkl`, ...
3. `notebooks/03_modeling.ipynb` – huấn luyện Logistic Regression (NumPy) và đánh giá trên test:
   - In các chỉ số: Accuracy, Precision, Recall, F1.
   - Vẽ learning curves (loss/accuracy theo iteration).
   - Lưu trọng số `lr_weights.npy` và dự đoán nếu cần.

---

## Results
Ví dụ một lần chạy (các con số có thể thay đổi tùy cấu hình features và seed):

- Train (tham khảo): Accuracy ~ 0.77
- Test (đánh giá): báo cáo đầy đủ Accuracy/Precision/Recall/F1 được in ở cuối notebook `03_modeling.ipynb`.

Trực quan:
- Learning curves thể hiện hội tụ ổn định.
- EDA cho thấy `city_development_index` tương quan mạnh với `target`.
- Phân tích theo `company_size` cho thấy quy mô nhỏ thường có tỷ lệ đổi việc cao hơn so với quy mô lớn.

---

## Project Structure
```
hr-job-change-project/
├─ data/
│  ├─ raw/                # Dữ liệu gốc (aug_train.csv, ...)
│  └─ processed/          # Dữ liệu sau preprocess (X_*.npy, y_*.npy, ...)
├─ notebooks/
│  ├─ 01_data_exploration.ipynb   # EDA
│  ├─ 02_preprocessing.ipynb      # Tiền xử lý + split
│  └─ 03_modeling.ipynb           # Logistic Regression (NumPy) + đánh giá
├─ src/
│  ├─ data_processing.py  # Hàm tiện ích xử lý dữ liệu (đọc csv, trích cột, thống kê,...)
│  ├─ models.py           # Hàm metrics (accuracy, precision, recall, f1, classification_report)
│  └─ visualization.py    # (Nếu dùng) tiện ích vẽ biểu đồ
├─ requirements.txt
└─ README.md
```

---

## Challenges & Solutions
- Missing values và giá trị đặc biệt (vd: `experience` chứa `>20`, `<1`, `never`) → chuẩn hóa và impute hợp lý.
- High-cardinality ở `city` → dùng Label Encoding thay vì One-hot để tránh nổ chiều.
- Ổn định số cho sigmoid/log-loss → clip đầu vào/đầu ra.
- Mất cân bằng lớp mức vừa → đánh giá bằng F1/Recall, cân nhắc điều chỉnh threshold.
- Notebook output bị in lặp trong VS Code/Jupyter → đảm bảo không gọi `fit` hai lần và restart kernel khi cần (đã khắc phục trong quá trình phát triển).

---

## Future Improvements
- Thêm regularization (L2/ElasticNet) và chọn siêu tham số bằng validation/cross-validation.
- So sánh với baseline thư viện (scikit-learn) và các mô hình khác (Tree/Ensemble).
- Tinh chỉnh threshold theo ROC/PR để tối ưu F1/Recall tùy mục tiêu kinh doanh.
- Feature selection/importance, SHAP cho khả năng giải thích.
- Chuẩn hóa pipeline dưới dạng script CLI hoặc module Python.

---

## Contributors
- Họ và Tên: Châu Văn Minh Khoa
- MSSV: 23122035

---

## License
Apache License - See LICENSE file for details