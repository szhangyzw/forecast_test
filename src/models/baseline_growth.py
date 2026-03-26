"""同比/增长率外推基线模型。"""

import pandas as pd


def predict_growth(last_year_value: float, growth_rate: float) -> float:
    return float(last_year_value * (1 + growth_rate))


def estimate_recent_yoy_growth(monthly_df: pd.DataFrame, target_month: str, window: int = 3):
    """基于目标月之前最近 window 个月估计同比增长率。monthly_df 需含 year_month, month_total_sales"""
    df = monthly_df.copy().sort_values("year_month").reset_index(drop=True)
    df["period"] = pd.PeriodIndex(df["year_month"], freq="M")
    target_period = pd.Period(target_month, freq="M")

    hist = df[df["period"] < target_period].copy()
    if hist.empty:
        return None

    yoy_values = []
    for _, row in hist.tail(window).iterrows():
        prev_year_period = row["period"] - 12
        prev = hist[hist["period"] == prev_year_period]
        if not prev.empty and float(prev.iloc[0]["month_total_sales"]) != 0:
            yoy = float(row["month_total_sales"] / prev.iloc[0]["month_total_sales"] - 1)
            yoy_values.append(yoy)

    if not yoy_values:
        return None
    return float(sum(yoy_values) / len(yoy_values))
