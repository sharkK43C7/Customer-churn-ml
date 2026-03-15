"""
Chạy dự đoán churn cho dữ liệu khách hàng mới (production-style).

Luồng:
- Load bundle đã lưu (model + artifacts preprocessing)
- Load CSV dữ liệu mới
- Clean + feature engineering + transform theo artifacts
- Predict xác suất churn (class=1)

Không huấn luyện model và không tính metric.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Cho phép chạy: python src/predict.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import create_features  # noqa: E402
from src.models import load_model, predict_churn_proba  # noqa: E402
from src.preprocessing import clean_data, transform_new_data  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict churn probability for new data")
    p.add_argument(
        "--model-path",
        default=str(Path("models") / "churn_model.pkl"),
        help="Path tới file model/bundle (mặc định models/churn_model.pkl)",
    )
    p.add_argument(
        "--data-path",
        default=str(Path("data") / "new_customers.csv"),
        help="Path tới CSV dữ liệu khách hàng mới (mặc định data/new_customers.csv)",
    )
    p.add_argument(
        "--output-path",
        default=str(Path("outputs") / "predictions.csv"),
        help="File CSV output (mặc định outputs/predictions.csv)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Mức log (mặc định INFO)",
    )
    return p.parse_args()


def _setup_logging(*, log_level: str = "INFO") -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(Path("logs") / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _risk_level(p: float) -> str:
    if p < 0.4:
        return "Low"
    if p < 0.7:
        return "Medium"
    return "High"


def _get_customer_ids(df: pd.DataFrame) -> pd.Series:
    """
    Lấy hoặc tạo customer_id theo cách an toàn:
    - Nếu có 'customerID'  -> dùng làm ID
    - Nếu có 'customer_id' -> dùng làm ID
    - Nếu không có         -> tạo ID dạng 'row_0', 'row_1', ...

    ID này chỉ dùng cho output/report, KHÔNG đưa vào feature cho model.
    """
    if "customerID" in df.columns:
        return df["customerID"].astype(str)
    if "customer_id" in df.columns:
        return df["customer_id"].astype(str)
    # Fallback: tạo ID ổn định theo index trong file hiện tại
    return df.index.to_series().astype(str).radd("row_")


def _is_project_bundle(obj: Any) -> bool:
    """Bundle do project tạo ra: {'model': ..., 'artifacts': ...}."""
    return isinstance(obj, dict) and "model" in obj and "artifacts" in obj


def _is_colab_bundle(obj: Any) -> bool:
    """Bundle từ Colab (theo file bạn đưa): {'model','encoders','feature_columns',...}."""
    return (
        isinstance(obj, dict)
        and "model" in obj
        and "encoders" in obj
        and "feature_columns" in obj
    )


def _transform_with_colab_bundle(df_feat: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    """
    Biến đổi dữ liệu mới theo format bundle Colab:
    - bundle['feature_columns']: thứ tự cột đầu vào model
    - bundle['encoders']: dict các encoder theo từng cột (nếu có)

    Ghi chú:
    - Không giả định có scaler; nếu bundle có scaler trong encoders thì vẫn áp dụng được.
    """
    feature_cols: list[str] = list(bundle["feature_columns"])
    encoders: dict[str, Any] = dict(bundle["encoders"])

    # Bảo đảm đủ cột và đúng thứ tự
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Thiếu các cột cần thiết theo model Colab: {missing}")

    X = df_feat[feature_cols].copy()

    # Áp dụng encoder theo cột nếu có
    for col, enc in encoders.items():
        if col not in X.columns:
            continue
        # encoder kiểu sklearn thường có .transform()
        if hasattr(enc, "transform"):
            # Một số encoder yêu cầu string
            try:
                X[col] = enc.transform(X[col].astype(str))
            except Exception:
                X[col] = enc.transform(X[col])

    # Ép kiểu số cho an toàn (các cột đã encode)
    X = X.apply(pd.to_numeric, errors="ignore")
    return X.values


def predict_file(model_path: str, data_path: str, output_path: str) -> pd.DataFrame:
    """
    Dự đoán churn probability cho một file CSV và lưu ra CSV.

    Bundle khuyến nghị khi save:
    {'model': fitted_model, 'artifacts': preprocessing_artifacts}
    """
    logger = logging.getLogger(__name__)
    bundle: Any = load_model(model_path)

    if _is_project_bundle(bundle):
        model = bundle["model"]
        artifacts = bundle["artifacts"]
        bundle_type = "project"
    elif _is_colab_bundle(bundle):
        model = bundle["model"]
        artifacts = None
        bundle_type = "colab"
    else:
        raise ValueError(
            "Model file không đúng format. Cần 1 trong 2 dạng:\n"
            "- Project: {'model': ..., 'artifacts': ...}\n"
            "- Colab: {'model': ..., 'encoders': ..., 'feature_columns': ...}"
        )

    df_new = pd.read_csv(data_path)
    # Giữ / tạo customer_id để output (không phụ thuộc preprocessing)
    customer_ids = _get_customer_ids(df_new)

    df_new_clean = clean_data(df_new)
    df_new_feat = create_features(df_new_clean)

    if bundle_type == "project":
        X_new = transform_new_data(df_new_feat, artifacts, clean=False)
    else:
        X_new = _transform_with_colab_bundle(df_new_feat, bundle)

    y_prob = predict_churn_proba(model, np.asarray(X_new))
    out = pd.DataFrame(
        {
            "customer_id": customer_ids.values,
            "churn_probability": y_prob,
            "risk_level": [_risk_level(float(p)) for p in y_prob],
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s", output_path)
    return out


def main() -> None:
    args = _parse_args()
    _setup_logging(log_level=args.log_level)
    predict_file(args.model_path, args.data_path, args.output_path)


if __name__ == "__main__":
    main()

