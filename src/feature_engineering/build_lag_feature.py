"""Lag与历史均值特征。"""

import pandas as pd


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["platform", "date"])
    for lag in [1, 7, 14, 30]:
        out[f"lag_{lag}"] = out.groupby("platform")["sales"].shift(lag)
    return out
