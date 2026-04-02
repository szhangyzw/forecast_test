"""构建截至某天的月度 snapshot 特征。"""

import pandas as pd


def build_snapshot_features(daily_df: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    df = daily_df.copy()
    if cutoff_day > 0:
        df = df[df["day_seq_in_month"] <= cutoff_day].copy()

    snapshot = (
        df.sort_values(["platform", "date"])
        .groupby(["platform", "year_month"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    snapshot["target_month"] = snapshot["year_month"]
    snapshot["cutoff_day"] = cutoff_day
    snapshot["target_remaining_sales"] = snapshot["month_total_sales"] - snapshot["mtd_sales"]

    keep_cols = [
        "target_month", "cutoff_day", "platform",
        "year", "month", "days_in_month",
        "days_elapsed", "days_remaining",
        "mtd_sales", "mtd_avg_sales", "mtd_progress",
        "ytd_sales", "year_total_sales", "ytd_progress",
        "ly_same_month_same_day_mtd", "ly_same_month_total",
        "month_total_sales", "target_remaining_sales",
        "last_7d_avg", "last_30d_avg",
    ]
    exist_cols = [c for c in keep_cols if c in snapshot.columns]
    return snapshot[exist_cols].sort_values(["platform", "target_month"]).reset_index(drop=True)


def add_historical_share_features(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """按 platform + month + cutoff_day(=days_elapsed) 计算历史累计占比统计。"""
    df = snapshot_df.copy()
    df["historical_share"] = df["mtd_sales"] / df["month_total_sales"]

    stats = (
        df.groupby(["platform", "month", "days_elapsed"], as_index=False)["historical_share"]
        .agg(hist_share_p50="median", hist_share_mean="mean")
    )
    out = df.merge(stats, on=["platform", "month", "days_elapsed"], how="left")
    return out


def add_historical_ytd_share_features(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """按 platform + month 计算历史 YTD 累计占全年比重统计，并补目标月 month share。"""
    df = snapshot_df.copy()
    if 'ytd_sales' not in df.columns or 'year_total_sales' not in df.columns:
        return df

    df['historical_ytd_share'] = df['ytd_sales'] / df['year_total_sales']
    df['historical_month_share'] = df['month_total_sales'] / df['year_total_sales']

    ytd_stats = (
        df.groupby(['platform', 'month'], as_index=False)['historical_ytd_share']
        .agg(hist_ytd_share_p50='median', hist_ytd_share_mean='mean')
    )
    month_stats = (
        df.groupby(['platform', 'month'], as_index=False)['historical_month_share']
        .agg(hist_month_share_p50='median', hist_month_share_mean='mean')
    )
    out = df.merge(ytd_stats, on=['platform', 'month'], how='left')
    out = out.merge(month_stats, on=['platform', 'month'], how='left')
    return out
