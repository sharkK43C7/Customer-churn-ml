## Dự án: Dự đoán khách hàng rời bỏ dịch vụ (Telco Customer Churn)

### 1. Ý tưởng (kế hoạch sơ bộ các bước)

Ý tưởng của dự án là xây dựng một pipeline “production-style” để:

1. **Nạp dữ liệu** từ CSV (đặt trong `data/`)
2. **Làm sạch cơ bản**
   - Bỏ cột định danh không dự báo (`customerID`)
   - Chuẩn hóa `TotalCharges` về dạng số, xử lý giá trị thiếu
3. **Tạo đặc trưng kinh doanh** (feature engineering) từ các cột sẵn có
4. **Chia train/test** và tiền xử lý theo kiểu production
   - Encode biến phân loại, scale biến số
   - Fit transformer trên train, test chỉ transform để tránh leakage
5. **Huấn luyện 2 mô hình** (baseline + mô hình phi tuyến)
   - Logistic Regression
   - Random Forest
6. **Đánh giá** và **chọn mô hình tốt nhất theo ROC-AUC**
7. **Lưu sản phẩm** để dùng lại
   - Model + preprocessing artifacts
   - Báo cáo metrics
8. **Dự đoán cho dữ liệu mới**
   - Tạo file dự đoán theo xác suất churn và phân loại rủi ro (Low/Medium/High)

### 2. Dữ liệu sử dụng

- Dataset: **Telco Customer Churn** (dạng CSV, ~7000 khách hàng)
- Một số cột chính:
  - `customerID`: mã khách hàng (chỉ dùng để nhận diện, không đưa vào model)
  - `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - `tenure`: số tháng gắn bó
  - `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, ...
  - `Contract`, `PaperlessBilling`, `PaymentMethod`
  - `MonthlyCharges`, `TotalCharges`
  - `Churn`: nhãn mục tiêu (`Yes` / `No`)

### 3. Kiến trúc project (production-style, đã tinh gọn)

Thư mục chính:

- `data/`
  - `Telco-Customer-Churn.csv`: dữ liệu gốc
  - `new_customers.csv`: dữ liệu khách hàng mới (dùng cho predict)
- `src/`
  - `preprocessing.py`: load & làm sạch dữ liệu, split train/test, encode/scale
  - `features.py`: tạo **đặc trưng kinh doanh** (feature engineering)
  - `models.py`: định nghĩa & huấn luyện model (Logistic Regression, Random Forest), lưu/tải model
  - `evaluation.py`: hàm đánh giá model (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
  - `predict.py`: **script predict end-to-end** cho dữ liệu mới
- `models/`
  - `churn_model.pkl`: bundle model tốt nhất + thông tin preprocessing
- `outputs/`
  - `metrics.json`: metric đánh giá model trên tập test
  - `predictions.csv`: dự đoán churn cho khách hàng mới
- `logs/`
  - `pipeline.log`: log chạy pipeline train/predict
- `PIPELINE.md`: mô tả pipeline chi tiết (theo flow kỹ thuật)

### 4. Cách pipeline hoạt động

#### 4.1. Tiền xử lý & tạo đặc trưng

1. **Load & clean** (`src/preprocessing.py`):
   - Đọc CSV (`load_data`)
   - Bỏ `customerID` khỏi feature (`clean_data`)
   - Chuyển `TotalCharges` sang số, giá trị lỗi/thiếu → `0` (khách mới, chưa phát sinh phí)
2. **Feature engineering** (`src/features.py` → `create_features`):
   - `tenure_group`: nhóm thời gian gắn bó: `0-6`, `7-12`, `13-24`, `24+`
   - `avg_monthly_spend`: chi tiêu trung bình / tháng = `TotalCharges / tenure`
   - `service_count`: số dịch vụ đang dùng (đếm các cột = `"Yes"`)
   - `is_auto_payment`: cờ thanh toán tự động (rủi ro churn thường thấp hơn)
3. **Split & scale** (`split_data`):
   - Tách `X`, `y` (với nhãn `Churn`)
   - Encode nhãn: `Yes/No` → `1/0` (LabelEncoder)
   - Tách train/test (stratify theo `y` để giữ tỷ lệ lớp)
   - Encode các biến phân loại (LabelEncoder theo từng cột, fit trên train)
   - Scale biến số (StandardScaler, fit trên train)
   - Lưu **artifacts**:
     - encoder từng cột
     - scaler
     - danh sách các cột feature

#### 4.2. Dự đoán cho khách hàng mới (`src/predict.py`)

Script `python src/predict.py`:

1. Load bundle model từ `models/churn_model.pkl`
   - Hỗ trợ 2 kiểu bundle:
     - Bundle của project (`{'model', 'artifacts'}`)
     - Bundle từ Colab (kiểu `{'model','encoders','feature_columns',...}`)
2. Đọc `data/new_customers.csv`
3. Xử lý `customer_id`:
   - Nếu có `customerID` → dùng làm `customer_id`
   - Nếu có `customer_id` → chuẩn hoá
   - Nếu không có ID → tự tạo `row_0`, `row_1`, ...
   - **ID chỉ dùng cho output, không đưa vào feature**
4. `clean_data` → `create_features` → transform theo artifacts/bundle
5. Dự đoán xác suất churn (`predict_churn_proba`)
6. Gán **risk level**:
   - `0.0 - 0.4` → `Low`
   - `0.4 - 0.7` → `Medium`
   - `0.7 - 1.0` → `High`
7. Lưu `outputs/predictions.csv` với các cột:
   - `customer_id`
   - `churn_probability`
   - `risk_level`

### 5. Kết quả (sản phẩm đầu ra là gì?)

Sau khi thiết lập và chạy pipeline dự đoán, sản phẩm chính của dự án là:

- **Model đã train** (thường được train trên Colab, sau đó copy về đây):
  - `models/churn_model.pkl`
  - Bên trong là “bundle” gồm mô hình và thông tin preprocessing cần thiết để transform dữ liệu mới đúng cách.
- **Kết quả dự đoán cho dữ liệu mới**:
  - `outputs/predictions.csv`
  - Gồm các cột:
    - `customer_id`: lấy từ `customerID` / `customer_id`, nếu không có sẽ tự tạo `row_0`, `row_1`, ...
    - `churn_probability`: xác suất churn (0 → 1)
    - `risk_level`: `Low` / `Medium` / `High`
- **Log chạy pipeline**:
  - `logs/pipeline.log`
  - Dùng để theo dõi tiến trình, debug lỗi, và audit.

### 6. Cách chạy nhanh

#### Cài thư viện

```bash
pip install -r requirements.txt
```

#### Predict cho khách hàng mới

Chuẩn bị `data/new_customers.csv` (cùng schema với file gốc, không cần cột `Churn`), sau đó:

```bash
python src/predict.py
```

Kết quả:
- `outputs/predictions.csv` với:
  - `customer_id`
  - `churn_probability`
  - `risk_level`

### 7. Hướng phát triển tiếp

- Thêm model nâng cao:
  - XGBoost / LightGBM / CatBoost
- Tối ưu hyperparameter (GridSearchCV / Optuna)
- Thêm explainability:
  - SHAP, permutation importance
- Đóng gói thành API (FastAPI/Flask) để tích hợp với hệ thống khác.

