"""日粒度聚合模块。"""

import pandas as pd


REQUIRED_COLUMNS = ["date", "platform", "sales"]


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    base = (
        df.groupby(["date", "platform"], as_index=False)["sales"]
        .sum()
        .sort_values(["platform", "date"])
    )

    total = (
        df.groupby(["date"], as_index=False)["sales"]
        .sum()
        .assign(platform="total")
        [["date", "platform", "sales"]]
    )

    out = pd.concat([base, total], ignore_index=True)
    out = out.sort_values(["platform", "date"]).reset_index(drop=True)
    return out
