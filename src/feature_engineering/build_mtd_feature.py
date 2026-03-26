"""MTD进度特征。"""

import pandas as pd


def build_mtd_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["platform", "date"])
    out["mtd_sales"] = out.groupby(["platform", "year_month"])["sales"].cumsum()
    out["mtd_avg_sales"] = out["mtd_sales"] / out["day_seq_in_month"].clip(lower=1)
    out["mtd_progress"] = out["day_seq_in_month"] / out["days_in_month"]
    out["days_elapsed"] = out["day_seq_in_month"]
    out["days_remaining"] = out["days_in_month"] - out["day_seq_in_month"]

    # 历史同月同日累计（去年）
    ly = out[["platform", "month", "day_seq_in_month", "year", "mtd_sales"]].copy()
    ly["year"] = ly["year"] + 1
    ly = ly.rename(columns={"mtd_sales": "ly_same_month_same_day_mtd"})
    out = out.merge(ly, on=["platform", "month", "day_seq_in_month", "year"], how="left")

    # 去年同月总销量
    month_total = (
        out.groupby(["platform", "year", "month"], as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "month_total_sales"})
    )
    ly_total = month_total.copy()
    ly_total["year"] = ly_total["year"] + 1
    ly_total = ly_total.rename(columns={"month_total_sales": "ly_same_month_total"})
    out = out.merge(ly_total, on=["platform", "year", "month"], how="left")

    return out
