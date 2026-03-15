# Pipeline dự đoán khách hàng rời bỏ (Customer Churn Prediction)

## Mục tiêu
Xây dựng mô hình học máy dự đoán khả năng khách hàng rời bỏ dịch vụ dựa trên thông tin cá nhân, hành vi sử dụng dịch vụ và dữ liệu thanh toán. Kết quả nhằm hỗ trợ doanh nghiệp và khách hàng.

## Cấu trúc output (production-style)
- `models/`: lưu model đã train (best) dưới dạng bundle (`.pkl`)
- `outputs/`: lưu metrics và dự đoán
- `logs/`: log pipeline (`pipeline.log`)

---

## Dữ liệu vào 
- File CSV: Telco Customer Churn
- ~7000 khách hàng
- Bao gồm:
 + Thông tin nhân khẩu học
 + Thông tin dịch vụ
 + Thông tin thanh toán
 + Trạng thái rời bỏ (churn)

---

## Bước 1: Tải và kiểm tra dữ liệu
- Đọc dữ liệu từ file CSV (`src/preprocessing.load_data`)
- Làm sạch cơ bản (`src/preprocessing.clean_data`)
  - Bỏ `customerID`
  - Chuẩn hóa `TotalCharges` sang số, giá trị thiếu -> 0

---

## Bước 2: Làm sạch dữ liệu 
- Chuyển đổi cột `TotalCharges` về kiểu số
- Xử lý giá trị thiếu 
- Kiểm tra và xử lý giá trị bất thường

---

## Bước 3: Tiền xử lý dữ liệu (pressprocessing)
- Tạo đặc trưng (`src/features.create_features`)
- Tách train/test + encode/scale (`src/preprocessing.split_data`)
  - Fit encoder/scaler chỉ trên train để tránh rò rỉ dữ liệu
  - Artifacts (encoder/scaler + danh sách cột) được lưu kèm model để dùng khi predict

---

## Bước 4: Tạo đặc trưng (Feature Engineering)
- Phân nhóm khách hàng theo thời gian(tenure)
- Tạo đặc trưng liên quan đến chi tiêu (vd: chi phí theo trung bình tháng)
- Tạo đặc trưng phản ánh mức độ sử dụng dịch vụ (số lượng dịch vụ đang dùng)
- Tránh sử dụng thông tin gây rò rỉ nhãn (data leakage)

---

## Bước 5: Xây dựng & đánh giá mô hình (concept)
- Logistic Regression (mô hình cơ sở) (`src/models.train_logistic_regression`)
- Random Forest (`src/models.train_random_forest`)
- (Mở rộng sau) XGBoost
- Lưu model/bundle (`src/models.save_model`)
  - Khuyến nghị lưu dạng bundle: `{'model': model, 'artifacts': artifacts}`

Việc train model có thể thực hiện trên Colab/notebook, sau đó:
- Lưu bundle (model + preprocessing info)
- Copy về thư mục `models/churn_model.pkl`

---

## Bước 6: Đánh giá mô hình (concept) 
- Sử dụng các chỉ số:
 + Accuracy
 + Precision
 + Recall
 + F1-score
 + ROc-AUC
- Phân tích mức độ quan trọng đặc trưng (feature importance)
- Đánh giá mô hình dưới góc nhìn bài toán kinh doanh

Đánh giá bằng code: `src/evaluation.evaluate_model(model, X_test, y_test)`

## Dự đoán cho dữ liệu mới (inference)
- Chuẩn bị file CSV dữ liệu khách hàng mới (không cần cột `Churn`)
- Chạy script:
  - `python src/predict.py`
  - Hoặc chỉ định: `python src/predict.py --data-path data/new_customers.csv --output-path outputs/predictions.csv`
- Output: `outputs/predictions.csv` gồm `customer_id`, `churn_probability`, `risk_level`
  - Low: 0.0 - 0.4
  - Medium: 0.4 - 0.7
  - High: 0.7 - 1.0

## Đầu ra:
- Xác suất khách hàng rời bỏ dịch vụ cho từng khách hàng trong `outputs/predictions.csv`
- Danh sách khách hàng theo `risk_level` (Low/Medium/High) để phục vụ chiến dịch giữ chân