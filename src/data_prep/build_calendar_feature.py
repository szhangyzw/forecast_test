"""日历特征模块。"""

import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["year_month"] = out[date_col].dt.strftime("%Y-%m")
    out["dow"] = out[date_col].dt.dayofweek
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["days_in_month"] = out[date_col].dt.days_in_month
    out["day_seq_in_month"] = out["day"]
    return out
