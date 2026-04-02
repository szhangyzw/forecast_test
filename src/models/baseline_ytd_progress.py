"""YTD 节奏外推模型。"""

from __future__ import annotations

import pandas as pd


def predict_by_ytd_progress(ytd_sales: float, historical_share: float) -> float:
    if historical_share is None or pd.isna(historical_share) or historical_share <= 0:
        raise ValueError("historical_share 必须大于 0")
    return float(ytd_sales / historical_share)
