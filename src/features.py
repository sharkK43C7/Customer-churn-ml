"""
Tạo đặc trưng kinh doanh cho bài toán churn (không dùng cột nhãn).

Các đặc trưng thêm:
- Nhóm theo thời gian gắn bó (tenure_group)
- Chi tiêu trung bình tháng (avg_monthly_spend)
- Số dịch vụ đang dùng (service_count)
- Cờ thanh toán tự động (is_auto_payment)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Danh sách cột dịch vụ phổ biến trong bộ Telco; có thể được override qua tham số
DEFAULT_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_group(df: pd.DataFrame, tenure_col: str = "tenure") -> pd.DataFrame:
    """
    Thêm cột nhóm thời gian gắn bó: 0-6, 7-12, 13-24, 24+ tháng.
    - Ý nghĩa: khách mới (0-6) có rủi ro churn cao hơn; trung thành tăng theo thời gian.
    """
    out = df.copy()
    bins = [-np.inf, 6, 12, 24, np.inf]
    labels = ["0-6", "7-12", "13-24", "24+"]
    out["tenure_group"] = pd.cut(out[tenure_col].fillna(0), bins=bins, labels=labels)
    return out


def add_avg_monthly_spend(
    df: pd.DataFrame, total_col: str = "TotalCharges", tenure_col: str = "tenure"
) -> pd.DataFrame:
    """
    Thêm cột chi tiêu trung bình tháng = TotalCharges / tenure.
    - Ý nghĩa: mức chi trung bình phản ánh giá trị và cam kết của khách.
    - Tránh chia cho 0: nếu tenure <= 0, dùng mẫu số = 1 (coi khách rất mới).
    """
    out = df.copy()
    tenure_safe = np.where(out[tenure_col] > 0, out[tenure_col], 1)
    out["avg_monthly_spend"] = out[total_col] / tenure_safe
    return out


def add_service_count(
    df: pd.DataFrame, service_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Thêm cột service_count: đếm số dịch vụ khách đang dùng (giá trị 'Yes').
    - Ý nghĩa: càng nhiều dịch vụ, chi phí chuyển đổi càng cao, churn có thể thấp hơn.
    """
    out = df.copy()
    cols = service_cols or [c for c in DEFAULT_SERVICE_COLS if c in out.columns]
    if not cols:
        out["service_count"] = 0
        return out

    # Đếm số cột có giá trị "Yes" (không phân biệt hoa thường)
    service_df = out[cols].fillna("No").apply(lambda col: col.str.lower() == "yes")
    out["service_count"] = service_df.sum(axis=1)
    return out


def add_payment_auto_flag(
    df: pd.DataFrame, payment_col: str = "PaymentMethod"
) -> pd.DataFrame:
    """
    Thêm cột is_auto_payment: 1 nếu phương thức thanh toán tự động, 0 nếu thủ công.
    - Ý nghĩa: thanh toán tự động giảm ma sát, thường gắn với churn thấp hơn.
    """
    out = df.copy()
    if payment_col not in out.columns:
        out["is_auto_payment"] = 0
        return out

    out["is_auto_payment"] = (
        out[payment_col]
        .fillna("")
        .str.lower()
        .str.contains("automatic", na=False)
        .astype(int)
    )
    return out


def tao_dac_trung(
    df: pd.DataFrame,
    tenure_col: str = "tenure",
    total_col: str = "TotalCharges",
    service_cols: list[str] | None = None,
    payment_col: str = "PaymentMethod",
) -> pd.DataFrame:
    """
    Hàm tổng hợp: áp dụng toàn bộ bước tạo đặc trưng, trả về DataFrame mới.
    Không đụng tới cột nhãn (Churn).
    """
    out = df.copy()
    out = add_tenure_group(out, tenure_col=tenure_col)
    out = add_avg_monthly_spend(out, total_col=total_col, tenure_col=tenure_col)
    out = add_service_count(out, service_cols=service_cols)
    out = add_payment_auto_flag(out, payment_col=payment_col)
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    API theo pipeline: tạo đặc trưng từ DataFrame đã clean (không dùng nhãn).

    Trả về DataFrame mới sẵn sàng cho bước split/encode/scale trong preprocessing.
    """
    return tao_dac_trung(df)


# Giải thích trực quan (không dùng trong code):
# - tenure_group: phân khúc tuổi đời dịch vụ, nhóm mới dễ rời hơn.
# - avg_monthly_spend: chi tiêu bình quân, cao/thấp có thể gợi ý giá trị cảm nhận.
# - service_count: số dịch vụ đang dùng; nhiều dịch vụ -> chi phí chuyển đổi cao.
# - is_auto_payment: thanh toán tự động giảm ma sát, gắn với churn thấp hơn.
