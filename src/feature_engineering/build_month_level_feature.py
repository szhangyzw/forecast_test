"""构建月级特征，用于 cutoff=0 的整月预测。"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.build_event_calendar_feature import add_event_features_to_month_df


def build_month_level_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    month_df = (
        daily_df.groupby(["platform", "year", "month", "year_month", "days_in_month"], as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "month_total_sales"})
        .sort_values(["platform", "year_month"])
        .reset_index(drop=True)
    )

    month_df["last_month_total"] = month_df.groupby("platform")["month_total_sales"].shift(1)
    month_df["last_3month_avg"] = month_df.groupby("platform")["month_total_sales"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    month_df["last_6month_avg"] = month_df.groupby("platform")["month_total_sales"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).mean()
    )

    ly = month_df[["platform", "year", "month", "month_total_sales"]].copy()
    ly["year"] = ly["year"] + 1
    ly = ly.rename(columns={"month_total_sales": "ly_same_month_total"})
    month_df = month_df.merge(ly, on=["platform", "year", "month"], how="left")

    month_df["recent_yoy_growth"] = month_df["last_month_total"] / month_df.groupby("platform")["last_month_total"].shift(12) - 1

    # 趋势与波动率
    month_df["last_3month_yoy_avg"] = month_df.groupby("platform")["recent_yoy_growth"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    month_df["last_6month_std"] = month_df.groupby("platform")["month_total_sales"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=2).std()
    )

    # 年内相对位置
    month_df["month_num_in_year"] = month_df["month"]

    month_df = add_event_features_to_month_df(month_df)
    return month_df
