"""
Preprocessing cho dự đoán churn (Telco Customer Churn).

Module này chỉ làm các bước cơ bản:
- Load CSV
- Làm sạch (bỏ customerID, chuẩn hóa TotalCharges)
- Tách train/test
- Mã hóa biến phân loại + chuẩn hóa biến số (fit trên train để tránh leakage)

Các hàm chính theo yêu cầu:
- load_data(path)
- clean_data(df)
- split_data(df, target_column)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV."""
    return pd.read_csv(path)


def clean_data(
    df: pd.DataFrame,
    *,
    id_columns: list[str] | None = None,
    total_charges_col: str = "TotalCharges",
) -> pd.DataFrame:
    """
    Làm sạch cơ bản.

    - Bỏ cột định danh (mặc định: customerID)
    - Đổi TotalCharges sang số; giá trị lỗi/thiếu -> 0 (khách mới)
    """
    out = df.copy()
    id_cols = id_columns or ["customerID"]
    out = out.drop(columns=id_cols, errors="ignore")

    if total_charges_col in out.columns:
        out[total_charges_col] = pd.to_numeric(out[total_charges_col], errors="coerce")
        out[total_charges_col] = out[total_charges_col].fillna(0)

    return out


def _split_X_y(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Tách X và y từ DataFrame đã clean."""
    if target_column not in df.columns:
        raise ValueError(f"Không thấy cột nhãn: {target_column}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def _infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Xác định cột phân loại và cột số (dựa trên dtype)."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return cat_cols, num_cols


def split_data(
    df: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Tách train/test và tiền xử lý X (encode + scale) theo kiểu production.

    Chống leakage:
    - Split trước
    - Fit encoder/scaler chỉ trên X_train
    - X_test chỉ transform

    Returns
    -------
    X_train, X_test, y_train, y_test, artifacts
        artifacts chứa label_encoders, scaler, danh sách cột và feature_names.
    """
    X, y_raw = _split_X_y(df, target_column)

    # Nhãn churn: Yes/No -> 1/0 (fit trên toàn y là OK vì không ảnh hưởng X)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y_raw.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    cat_cols, num_cols = _infer_column_types(X_train)

    X_train_p = X_train.copy()
    X_test_p = X_test.copy()

    # Encode categorical (LabelEncoder theo từng cột; fit trên train)
    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train_p[col] = le.fit_transform(X_train[col].astype(str))
        X_test_p[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # Scale numerical (fit trên train)
    scaler = StandardScaler()
    if num_cols:
        X_train_p[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_p[num_cols] = scaler.transform(X_test[num_cols])

    artifacts: dict[str, Any] = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
        "feature_names": X_train.columns.tolist(),
        "y_encoder": y_encoder,
    }

    return X_train_p.values, X_test_p.values, y_train, y_test, artifacts


def transform_new_data(
    df_new: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    clean: bool = True,
) -> np.ndarray:
    """
    Áp dụng đúng encoder/scaler đã fit để biến đổi dữ liệu mới (inference).

    Parameters
    ----------
    df_new : pd.DataFrame
        Dữ liệu khách hàng mới (không có nhãn).
    artifacts : dict
        Kết quả trả về từ split_data().
    clean : bool, default=True
        Nếu True: gọi clean_data() trước khi transform.

    Returns
    -------
    np.ndarray
        Ma trận đặc trưng đã encode/scale theo đúng artifacts.
    """
    df_in = clean_data(df_new) if clean else df_new.copy()

    feature_names: list[str] = artifacts["feature_names"]
    X = df_in.copy()

    # Bảo đảm đủ cột và đúng thứ tự
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise ValueError(f"Thiếu các cột cần thiết: {missing}")
    X = X[feature_names]

    cat_cols: list[str] = artifacts["categorical_cols"]
    num_cols: list[str] = artifacts["numerical_cols"]
    label_encoders: dict[str, LabelEncoder] = artifacts["label_encoders"]
    scaler: StandardScaler = artifacts["scaler"]

    X_p = X.copy()
    for col in cat_cols:
        le = label_encoders[col]
        X_p[col] = le.transform(X[col].astype(str))

    if num_cols:
        X_p[num_cols] = scaler.transform(X[num_cols])

    return X_p.values
