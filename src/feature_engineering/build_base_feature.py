"""基础特征构建。"""

import pandas as pd


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["platform", "date"])
    out["last_7d_avg"] = out.groupby("platform")["sales"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["last_30d_avg"] = out.groupby("platform")["sales"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    return out
