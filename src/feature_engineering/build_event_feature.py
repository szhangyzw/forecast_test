"""活动特征占位模块。"""

import pandas as pd


def build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_promo" not in out.columns:
        out["is_promo"] = 0
    return out
