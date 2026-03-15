"""
Mô-đun mô hình cho dự đoán churn khách hàng Telco.

Chỉ chứa logic liên quan mô hình:
- Định nghĩa mô hình
- Huấn luyện mô hình
- Dự đoán xác suất churn (lớp 1)

Không đọc CSV, in kết quả, tính metric hay vẽ đồ thị.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib


def predict_churn_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Dự đoán xác suất churn (lớp 1) cho mỗi mẫu.

    Parameters
    ----------
    model : classifier có .predict_proba(X)
        Mô hình đã fit (LogisticRegression, RandomForest, ...).
    X : np.ndarray, shape (n_samples, n_features)
        Ma trận đặc trưng đã tiền xử lý.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Xác suất P(churn = 1) cho từng mẫu.
    """
    proba = model.predict_proba(X)
    # Cột 1 tương ứng lớp 1 (churn)
    return proba[:, 1]


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """
    Huấn luyện Logistic Regression (baseline, dễ giải thích).

    Mặc định:
    - class_weight='balanced': điều chỉnh trọng số theo tần số lớp để xử lý mất cân bằng.
    - C=1.0: nghịch đảo độ mạnh regularization; 1.0 là lựa chọn ổn định.
    - max_iter=1000: đủ cho hội tụ trên dữ liệu Telco kích thước trung bình.
    - solver='lbfgs': phù hợp cho bài toán nhị phân, ít hyperparameter.

    Parameters
    ----------
    X_train : np.ndarray
        Đặc trưng tập huấn luyện.
    y_train : np.ndarray
        Nhãn tập huấn luyện (0/1).
    C : float
        Nghịch đảo regularization strength.
    max_iter : int
        Số epoch tối đa.
    random_state : int
        Seed tái lặp.
    **kwargs
        Tham số bổ sung cho LogisticRegression.

    Returns
    -------
    LogisticRegression
        Mô hình đã fit.
    """
    defaults = {
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": random_state,
        "C": C,
        "max_iter": max_iter,
    }
    opts = {**defaults, **kwargs}
    clf = LogisticRegression(**opts)
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 100,
    max_depth: int | None = 12,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Huấn luyện Random Forest (mô hình phi tuyến, ensemble).

    Mặc định:
    - class_weight='balanced_subsample': cân bằng theo từng bootstrap, phù hợp RF.
    - n_estimators=100: cân bằng giữa chất lượng và thời gian.
    - max_depth=12: hạn chế overfit, vẫn nắm được tương tác.
    - min_samples_leaf=5: lá có ít nhất 5 mẫu, giảm nhiễu.

    Parameters
    ----------
    X_train : np.ndarray
        Đặc trưng tập huấn luyện.
    y_train : np.ndarray
        Nhãn tập huấn luyện (0/1).
    n_estimators : int
        Số cây trong rừng.
    max_depth : int | None
        Độ sâu tối đa mỗi cây; None = không giới hạn.
    min_samples_leaf : int
        Số mẫu tối thiểu ở lá.
    random_state : int
        Seed tái lặp.
    **kwargs
        Tham số bổ sung cho RandomForestClassifier.

    Returns
    -------
    RandomForestClassifier
        Mô hình đã fit.
    """
    defaults = {
        "class_weight": "balanced_subsample",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
    }
    opts = {**defaults, **kwargs}
    clf = RandomForestClassifier(**opts)
    clf.fit(X_train, y_train)
    return clf


def save_model(model: Any, path: str) -> None:
    """
    Lưu model (hoặc bundle: {'model': ..., 'artifacts': ...}) ra file.

    Dùng joblib vì nhanh và phù hợp object sklearn.
    """
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Tải model (hoặc bundle) đã lưu bằng joblib."""
    return joblib.load(path)
