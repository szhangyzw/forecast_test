"""电商关键事件特征：春节 / 618 / 双11。"""

from __future__ import annotations

import pandas as pd


# 农历春节（正月初一）对应公历日期，可按需要继续补充
CNY_DATES = {
    2024: pd.Timestamp("2024-02-10"),
    2025: pd.Timestamp("2025-01-29"),
    2026: pd.Timestamp("2026-02-17"),
}


def add_event_features_to_month_df(month_df: pd.DataFrame) -> pd.DataFrame:
    df = month_df.copy()

    # 构造每个月月初日期
    df["month_start"] = pd.to_datetime(df["year_month"] + "-01")

    # 618 / 双11：根据业务约定，预热从 5/10、10/10 开始
    df["is_618_impact_month"] = df["month"].isin([5, 6]).astype(int)
    df["pre_618_flag"] = (df["month"] == 5).astype(int)
    df["in_618_flag"] = (df["month"] == 6).astype(int)

    df["is_double11_impact_month"] = df["month"].isin([10, 11]).astype(int)
    df["pre_double11_flag"] = (df["month"] == 10).astype(int)
    df["in_double11_flag"] = (df["month"] == 11).astype(int)

    # 春节影响：按春节所在月份及前一个月、后一个月做窗口标签
    df["is_cny_impact_month"] = 0
    df["pre_cny_flag"] = 0
    df["in_cny_flag"] = 0
    df["post_cny_flag"] = 0
    df["days_to_cny"] = -1
    df["days_from_cny"] = -1

    for idx, row in df.iterrows():
        y = int(row["year"])
        if y not in CNY_DATES:
            continue
        cny_date = CNY_DATES[y]
        month_start = row["month_start"]
        month_period = pd.Period(month_start, freq="M")
        cny_period = pd.Period(cny_date, freq="M")

        # 目标月是春节前一个月 / 春节当月 / 春节后一个月
        if month_period == cny_period - 1:
            df.at[idx, "is_cny_impact_month"] = 1
            df.at[idx, "pre_cny_flag"] = 1
        elif month_period == cny_period:
            df.at[idx, "is_cny_impact_month"] = 1
            df.at[idx, "in_cny_flag"] = 1
        elif month_period == cny_period + 1:
            df.at[idx, "is_cny_impact_month"] = 1
            df.at[idx, "post_cny_flag"] = 1

        # 月初视角下距离春节还有/已过几天
        delta = (cny_date - month_start).days
        if delta >= 0:
            df.at[idx, "days_to_cny"] = delta
            df.at[idx, "days_from_cny"] = 0
        else:
            df.at[idx, "days_to_cny"] = 0
            df.at[idx, "days_from_cny"] = abs(delta)

    return df
