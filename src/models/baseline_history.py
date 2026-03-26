"""历史同期/历史均值基线模型。"""

import pandas as pd


def predict_history_average(df: pd.DataFrame, horizon_days: int) -> float:
    daily_avg = df["sales"].tail(90).mean()
    return float(daily_avg * horizon_days)
