"""
Mô-đun đánh giá cho dự đoán churn khách hàng Telco.

Chỉ chứa logic đánh giá: confusion matrix, metric phân loại,
ROC-AUC, Precision-Recall AUC, dự đoán theo ngưỡng.
Không đọc CSV, huấn luyện mô hình, tạo đặc trưng hay in kết quả.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _check_same_length(*arrays: np.ndarray) -> None:
    """Kiểm tra các mảng cùng độ dài."""
    n = len(arrays[0])
    for i, a in enumerate(arrays[1:], start=1):
        if len(a) != n:
            raise ValueError(f"Mảng {i} có độ dài {len(a)}, khác mảng 0 ({n}).")


def probabilities_to_labels(
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Chuyển xác suất thành nhãn 0/1 theo ngưỡng.

    Parameters
    ----------
    y_prob : np.ndarray, shape (n_samples,)
        Xác suất lớp dương (churn = 1).
    threshold : float, default=0.5
        Ngưỡng: y_pred = 1 khi y_prob >= threshold.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Nhãn 0 hoặc 1.
    """
    return (np.asarray(y_prob) >= threshold).astype(np.int64)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pos_label: int = 1,
) -> np.ndarray:
    """
    Tính ma trận nhầm lẫn.

    Thứ tự lớp: [không churn, churn] tương ứng [0, 1].
    Hàng = thật, cột = dự đoán.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_pred : np.ndarray
        Nhãn dự đoán (0/1).
    pos_label : int, default=1
        Lớp dương (churn).

    Returns
    -------
    np.ndarray, shape (2, 2)
        [[TN, FP], [FN, TP]].
    """
    _check_same_length(y_true, y_pred)
    labels = [0, 1] if pos_label == 1 else [1, 0]
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pos_label: int = 1,
) -> dict[str, float]:
    """
    Tính accuracy, precision, recall, F1 cho lớp dương (churn = 1).

    Recall cho lớp churn được ưu tiên khi thiết kế metric.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_pred : np.ndarray
        Nhãn dự đoán (0/1).
    pos_label : int, default=1
        Lớp dương (churn).

    Returns
    -------
    dict[str, float]
        accuracy, precision, recall, f1 (zero_division=0).
    """
    _check_same_length(y_true, y_pred)
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "precision": float(
            precision_score(y_t, y_p, pos_label=pos_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_t, y_p, pos_label=pos_label, zero_division=0)
        ),
        "f1": float(f1_score(y_t, y_p, pos_label=pos_label, zero_division=0)),
    }


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Tính ROC-AUC từ xác suất dự đoán.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray, shape (n_samples,)
        Xác suất lớp dương (churn = 1).

    Returns
    -------
    float
        ROC-AUC. Trường hợp chỉ một lớp xuất hiện trả về 0.0.
    """
    _check_same_length(y_true, y_prob)
    y_t = np.asarray(y_true)
    y_pr = np.asarray(y_prob)
    if np.unique(y_t).size < 2:
        return 0.0
    return float(roc_auc_score(y_t, y_pr))


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Tính Precision-Recall AUC (Average Precision) từ xác suất.

    Phù hợp khi lớp dương (churn) thiểu số.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray, shape (n_samples,)
        Xác suất lớp dương (churn = 1).

    Returns
    -------
    float
        PR-AUC. Trường hợp không có lớp dương trả về 0.0.
    """
    _check_same_length(y_true, y_prob)
    y_t = np.asarray(y_true)
    y_pr = np.asarray(y_prob)
    if y_t.sum() == 0:
        return 0.0
    return float(average_precision_score(y_t, y_pr))


def evaluate(
    y_true: np.ndarray,
    *,
    y_pred: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
    threshold: float = 0.5,
    pos_label: int = 1,
) -> dict[str, Any]:
    """
    Tổng hợp metric: confusion matrix, accuracy, precision, recall, F1,
    ROC-AUC, PR-AUC (khi có y_prob).

    Nếu chỉ có y_prob: dùng threshold để sinh y_pred cho metric phân loại.
    Nếu có cả y_pred và y_prob: dùng y_pred cho metric phân loại và
    confusion matrix; y_prob chỉ dùng cho ROC-AUC và PR-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_pred : np.ndarray | None
        Nhãn dự đoán (0/1). Có thể bỏ nếu có y_prob.
    y_prob : np.ndarray | None
        Xác suất lớp dương. Cần có để tính ROC-AUC, PR-AUC.
    threshold : float, default=0.5
        Ngưỡng chuyển xác suất -> nhãn khi chỉ có y_prob.
    pos_label : int, default=1
        Lớp dương (churn).

    Returns
    -------
    dict[str, Any]
        confusion_matrix (ndarray), accuracy, precision, recall, f1,
        roc_auc (nếu có y_prob), pr_auc (nếu có y_prob).
    """
    if y_pred is None and y_prob is None:
        raise ValueError("Cần ít nhất một trong y_pred hoặc y_prob.")

    y_t = np.asarray(y_true)

    if y_pred is None:
        y_pred = probabilities_to_labels(y_prob, threshold=threshold)
    else:
        _check_same_length(y_true, y_pred)

    out: dict[str, Any] = {
        "confusion_matrix": compute_confusion_matrix(
            y_t, y_pred, pos_label=pos_label
        ),
        **compute_classification_metrics(y_t, y_pred, pos_label=pos_label),
    }

    if y_prob is not None:
        _check_same_length(y_true, y_prob)
        out["roc_auc"] = compute_roc_auc(y_t, np.asarray(y_prob))
        out["pr_auc"] = compute_pr_auc(y_t, np.asarray(y_prob))

    return out


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """
    API theo pipeline: đánh giá model trên tập test.

    Trả về:
    - accuracy
    - roc_auc (nếu model có predict_proba)
    - confusion_matrix
    - classification_report (dạng dict)
    """
    y_true = np.asarray(y_test)

    # Nhãn dự đoán
    y_pred = model.predict(X_test)

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, pos_label=1),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

    # ROC-AUC cần xác suất
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        out["roc_auc"] = compute_roc_auc(y_true, y_prob)
        out["pr_auc"] = compute_pr_auc(y_true, y_prob)

    return out
