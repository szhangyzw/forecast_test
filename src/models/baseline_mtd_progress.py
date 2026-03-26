"""MTD节奏外推模型。"""


def predict_by_progress(mtd_sales: float, historical_share: float) -> float:
    if historical_share <= 0:
        raise ValueError("historical_share 必须大于 0")
    return float(mtd_sales / historical_share)
