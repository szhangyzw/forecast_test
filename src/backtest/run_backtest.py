"""回测主程序第一版（按业务 cutoff 口径）。"""

from pathlib import Path
import pandas as pd

from src.inference.run_forecast import prepare_daily_feature_table
from src.feature_engineering.build_snapshot_feature import build_snapshot_features, add_historical_share_features
from src.models.baseline_history import predict_history_average
from src.models.baseline_mtd_progress import predict_by_progress
from src.models.baseline_growth import predict_growth, estimate_recent_yoy_growth
from src.models.xgb_model import train_xgb, predict_xgb, FEATURES_CUTOFF0, FEATURES_CUTOFFN
from src.feature_engineering.build_month_level_feature import build_month_level_features
from src.models.prophet_model import predict_prophet_full_month
from src.models.prophet_oldlogic_model import predict_prophet_oldlogic
from src.models.daily_direct_models import predict_tree_daily_direct, predict_prophet_daily_direct


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "test_forecast_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "backtest_result"


MONTHLY_EVAL_COLS = [
    "target_month", "cutoff_day", "platform", "model_name",
    "pred_remaining_sales", "actual_remaining_sales",
    "pred_total_sales", "actual_total_sales",
    "abs_error_remaining", "ape_remaining", "bias_remaining",
    "abs_error_total", "ape_total", "bias_total",
]


def _get_complete_months(daily: pd.DataFrame) -> list[str]:
    """返回当前切片数据里可用于回测的完整月份。

    旧逻辑写死依赖 platform=='total'，在 brand/platform/entity 切片后会失效。
    这里改成：只要该切片下任一实体在该月份已完整覆盖，就认为该月份可进入候选；
    后续各实体在自己的循环里仍会基于历史样本是否充分决定是否产出结果。
    """
    month_end = (
        daily.groupby(["platform", "year_month"], as_index=False)
        .agg(max_day=("day_seq_in_month", "max"), days_in_month=("days_in_month", "max"))
    )
    complete = month_end[month_end["max_day"] >= month_end["days_in_month"]].copy()
    return sorted(complete["year_month"].dropna().unique().tolist())


def _safe_pct_err(pred, actual):
    if actual is None or pd.isna(actual) or actual == 0:
        return None
    return abs(pred - actual) / actual


def _append_result(results, **kwargs):
    results.append({k: kwargs.get(k) for k in MONTHLY_EVAL_COLS})


def _normalize_cutoff_days(cutoff_days):
    if cutoff_days is None:
        return (0, 5, 10, 15, 20, 25)
    vals = []
    for x in cutoff_days:
        x = int(x)
        if x < 0:
            raise ValueError(f'cutoff_day 不能小于0: {x}')
        vals.append(x)
    return tuple(dict.fromkeys(vals))


def run_backtest(
    cutoff_days=(0, 15, 25),
    exclude_months=None,
    output_dir: str | Path = DEFAULT_OUTPUT,
    brand: str = 'total',
    platform: str = 'total',
):
    if exclude_months is None:
        exclude_months = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff_days = _normalize_cutoff_days(cutoff_days)
    daily = prepare_daily_feature_table(DEFAULT_INPUT, brand=brand, platform=platform)
    complete_months = [m for m in _get_complete_months(daily) if m not in set(exclude_months)]

    results = []

    for cutoff_day in cutoff_days:
        if cutoff_day == 0:
            # 用 month t 的历史，预测 month t+1 整月
            for platform in sorted(daily["platform"].unique()):
                month_total = build_month_level_features(daily[daily["platform"] == platform])
                month_total = month_total.rename(columns={"month_total_sales": "actual_total_sales"}).sort_values("year_month").reset_index(drop=True)
                month_total["period"] = pd.PeriodIndex(month_total["year_month"], freq="M")

                for i in range(1, len(month_total)):
                    row = month_total.iloc[i]
                    target_month = row["year_month"]
                    if target_month not in complete_months:
                        continue

                    hist_months = month_total.iloc[:i].copy()
                    if hist_months.empty:
                        continue

                    actual_total = float(row["actual_total_sales"])
                    horizon_days = int(pd.Period(target_month, freq="M").days_in_month)
                    daily_hist = daily[(daily["platform"] == platform) & (daily["year_month"].isin(hist_months["year_month"]))]

                    # 1) history average
                    pred_total = predict_history_average(daily_hist, horizon_days=horizon_days)
                    _append_result(
                        results,
                        target_month=target_month,
                        cutoff_day=0,
                        platform=platform,
                        model_name="history_average",
                        pred_remaining_sales=pred_total,
                        actual_remaining_sales=actual_total,
                        pred_total_sales=pred_total,
                        actual_total_sales=actual_total,
                        abs_error_remaining=abs(pred_total - actual_total),
                        ape_remaining=_safe_pct_err(pred_total, actual_total),
                        bias_remaining=pred_total - actual_total,
                        abs_error_total=abs(pred_total - actual_total),
                        ape_total=_safe_pct_err(pred_total, actual_total),
                        bias_total=pred_total - actual_total,
                    )

                    # 1.5) prophet full month
                    pred_total_prophet = predict_prophet_full_month(
                        daily_hist[['date', 'sales']].copy(),
                        target_month=target_month,
                    )
                    if pd.notna(pred_total_prophet):
                        _append_result(
                            results,
                            target_month=target_month,
                            cutoff_day=0,
                            platform=platform,
                            model_name="prophet_fullmonth",
                            pred_remaining_sales=float(pred_total_prophet),
                            actual_remaining_sales=actual_total,
                            pred_total_sales=float(pred_total_prophet),
                            actual_total_sales=actual_total,
                            abs_error_remaining=abs(float(pred_total_prophet) - actual_total),
                            ape_remaining=_safe_pct_err(float(pred_total_prophet), actual_total),
                            bias_remaining=float(pred_total_prophet) - actual_total,
                            abs_error_total=abs(float(pred_total_prophet) - actual_total),
                            ape_total=_safe_pct_err(float(pred_total_prophet), actual_total),
                            bias_total=float(pred_total_prophet) - actual_total,
                        )

                    # 2) last year same month
                    target_period = pd.Period(target_month, freq="M")
                    ly_period = target_period - 12
                    ly_row = hist_months[hist_months["period"] == ly_period]
                    if not ly_row.empty:
                        pred_total_ly = float(ly_row.iloc[0]["actual_total_sales"])
                        _append_result(
                            results,
                            target_month=target_month,
                            cutoff_day=0,
                            platform=platform,
                            model_name="last_year_same_month",
                            pred_remaining_sales=pred_total_ly,
                            actual_remaining_sales=actual_total,
                            pred_total_sales=pred_total_ly,
                            actual_total_sales=actual_total,
                            abs_error_remaining=abs(pred_total_ly - actual_total),
                            ape_remaining=_safe_pct_err(pred_total_ly, actual_total),
                            bias_remaining=pred_total_ly - actual_total,
                            abs_error_total=abs(pred_total_ly - actual_total),
                            ape_total=_safe_pct_err(pred_total_ly, actual_total),
                            bias_total=pred_total_ly - actual_total,
                        )

                        # 3) yoy growth extrapolation
                        yoy_growth = estimate_recent_yoy_growth(
                            hist_months.rename(columns={"actual_total_sales": "month_total_sales"})[["year_month", "month_total_sales"]],
                            target_month=target_month,
                            window=3,
                        )
                        if yoy_growth is not None:
                            pred_total_yoy = predict_growth(pred_total_ly, yoy_growth)
                            _append_result(
                                results,
                                target_month=target_month,
                                cutoff_day=0,
                                platform=platform,
                                model_name="yoy_growth_extrapolation",
                                pred_remaining_sales=pred_total_yoy,
                                actual_remaining_sales=actual_total,
                                pred_total_sales=pred_total_yoy,
                                actual_total_sales=actual_total,
                                abs_error_remaining=abs(pred_total_yoy - actual_total),
                                ape_remaining=_safe_pct_err(pred_total_yoy, actual_total),
                                bias_remaining=pred_total_yoy - actual_total,
                                abs_error_total=abs(pred_total_yoy - actual_total),
                                ape_total=_safe_pct_err(pred_total_yoy, actual_total),
                                bias_total=pred_total_yoy - actual_total,
                            )

                    xgb_train = hist_months.rename(columns={"actual_total_sales": "month_total_sales"}).copy()
                    xgb_model = train_xgb(xgb_train, FEATURES_CUTOFF0, "month_total_sales")
                    pred_row = pd.DataFrame([{
                        "platform": platform,
                        "year": target_period.year,
                        "month": target_period.month,
                        "year_month": target_month,
                        "days_in_month": target_period.days_in_month,
                        "month_total_sales": pd.NA,
                        "last_month_total": xgb_train["month_total_sales"].iloc[-1] if len(xgb_train) >= 1 else pd.NA,
                        "last_3month_avg": xgb_train["month_total_sales"].tail(3).mean() if len(xgb_train) >= 1 else pd.NA,
                        "last_6month_avg": xgb_train["month_total_sales"].tail(6).mean() if len(xgb_train) >= 1 else pd.NA,
                        "ly_same_month_total": float(ly_row.iloc[0]["actual_total_sales"]) if not ly_row.empty else pd.NA,
                        "recent_yoy_growth": estimate_recent_yoy_growth(
                            xgb_train[["year_month", "month_total_sales"]], target_month=target_month, window=3
                        ),
                        "last_3month_yoy_avg": xgb_train["last_3month_yoy_avg"].iloc[-1] if 'last_3month_yoy_avg' in xgb_train.columns and len(xgb_train) >= 1 else pd.NA,
                        "last_6month_std": xgb_train["last_6month_std"].iloc[-1] if 'last_6month_std' in xgb_train.columns and len(xgb_train) >= 1 else pd.NA,
                        "month_num_in_year": target_period.month,
                    }])
                    from src.feature_engineering.build_event_calendar_feature import add_event_features_to_month_df
                    pred_row = add_event_features_to_month_df(pred_row)
                    pred_total_xgb = predict_xgb(xgb_model, pred_row).iloc[0]
                    if pd.notna(pred_total_xgb):
                        _append_result(
                            results,
                            target_month=target_month,
                            cutoff_day=0,
                            platform=platform,
                            model_name="xgboost_v2_event",
                            pred_remaining_sales=float(pred_total_xgb),
                            actual_remaining_sales=actual_total,
                            pred_total_sales=float(pred_total_xgb),
                            actual_total_sales=actual_total,
                            abs_error_remaining=abs(float(pred_total_xgb) - actual_total),
                            ape_remaining=_safe_pct_err(float(pred_total_xgb), actual_total),
                            bias_remaining=float(pred_total_xgb) - actual_total,
                            abs_error_total=abs(float(pred_total_xgb) - actual_total),
                            ape_total=_safe_pct_err(float(pred_total_xgb), actual_total),
                            bias_total=float(pred_total_xgb) - actual_total,
                        )

                    pred_total_gbdt_dd = predict_tree_daily_direct(
                        daily[daily['platform'] == platform].copy().sort_values('date'),
                        target_month=target_month,
                        cutoff_day=0,
                        model_type='gbdt',
                    )
                    if pred_total_gbdt_dd is not None and pd.notna(pred_total_gbdt_dd):
                        _append_result(
                            results,
                            target_month=target_month,
                            cutoff_day=0,
                            platform=platform,
                            model_name="gbdt_daily_direct_c0",
                            pred_remaining_sales=float(pred_total_gbdt_dd),
                            actual_remaining_sales=actual_total,
                            pred_total_sales=float(pred_total_gbdt_dd),
                            actual_total_sales=actual_total,
                            abs_error_remaining=abs(float(pred_total_gbdt_dd) - actual_total),
                            ape_remaining=_safe_pct_err(float(pred_total_gbdt_dd), actual_total),
                            bias_remaining=float(pred_total_gbdt_dd) - actual_total,
                            abs_error_total=abs(float(pred_total_gbdt_dd) - actual_total),
                            ape_total=_safe_pct_err(float(pred_total_gbdt_dd), actual_total),
                            bias_total=float(pred_total_gbdt_dd) - actual_total,
                        )

                    pred_total_xgb_dd = predict_tree_daily_direct(
                        daily[daily['platform'] == platform].copy().sort_values('date'),
                        target_month=target_month,
                        cutoff_day=0,
                        model_type='xgb',
                    )
                    if pred_total_xgb_dd is not None and pd.notna(pred_total_xgb_dd):
                        _append_result(
                            results,
                            target_month=target_month,
                            cutoff_day=0,
                            platform=platform,
                            model_name="xgboost_daily_direct_c0",
                            pred_remaining_sales=float(pred_total_xgb_dd),
                            actual_remaining_sales=actual_total,
                            pred_total_sales=float(pred_total_xgb_dd),
                            actual_total_sales=actual_total,
                            abs_error_remaining=abs(float(pred_total_xgb_dd) - actual_total),
                            ape_remaining=_safe_pct_err(float(pred_total_xgb_dd), actual_total),
                            bias_remaining=float(pred_total_xgb_dd) - actual_total,
                            abs_error_total=abs(float(pred_total_xgb_dd) - actual_total),
                            ape_total=_safe_pct_err(float(pred_total_xgb_dd), actual_total),
                            bias_total=float(pred_total_xgb_dd) - actual_total,
                        )

                    pred_total_prophet_dd = predict_prophet_daily_direct(
                        daily[daily['platform'] == platform].copy().sort_values('date'),
                        target_month=target_month,
                        cutoff_day=0,
                    )
                    if pred_total_prophet_dd is not None and pd.notna(pred_total_prophet_dd):
                        _append_result(
                            results,
                            target_month=target_month,
                            cutoff_day=0,
                            platform=platform,
                            model_name="prophet_daily_direct_c0",
                            pred_remaining_sales=float(pred_total_prophet_dd),
                            actual_remaining_sales=actual_total,
                            pred_total_sales=float(pred_total_prophet_dd),
                            actual_total_sales=actual_total,
                            abs_error_remaining=abs(float(pred_total_prophet_dd) - actual_total),
                            ape_remaining=_safe_pct_err(float(pred_total_prophet_dd), actual_total),
                            bias_remaining=float(pred_total_prophet_dd) - actual_total,
                            abs_error_total=abs(float(pred_total_prophet_dd) - actual_total),
                            ape_total=_safe_pct_err(float(pred_total_prophet_dd), actual_total),
                            bias_total=float(pred_total_prophet_dd) - actual_total,
                        )

        else:
            effective_cutoff = max(int(cutoff_day), 1)
            snapshot = build_snapshot_features(daily, cutoff_day=effective_cutoff)
            snapshot = snapshot[snapshot["target_month"].isin(complete_months)].copy()
            if not snapshot.empty:
                snapshot['cutoff_day'] = snapshot[['cutoff_day', 'days_in_month']].min(axis=1)
                snapshot['days_elapsed'] = snapshot[['days_elapsed', 'days_in_month']].min(axis=1)
                snapshot['days_remaining'] = snapshot['days_in_month'] - snapshot['days_elapsed']
                snapshot['mtd_progress'] = snapshot['days_elapsed'] / snapshot['days_in_month']
            snapshot = add_historical_share_features(snapshot)

            monthly_pivot = (
                daily.groupby(['year_month', 'platform'], as_index=False)['sales']
                .sum()
                .pivot(index='year_month', columns='platform', values='sales')
                .sort_index()
            )

            for _, row in snapshot.iterrows():
                platform = row["platform"]
                target_month = row["target_month"]
                actual_total = float(row["month_total_sales"])
                actual_remaining = float(row["target_remaining_sales"])

                history_df = daily[(daily["platform"] == platform) & (daily["year_month"] < target_month)]
                horizon_days = int(row["days_remaining"])
                pred_remaining_hist = predict_history_average(history_df, horizon_days=horizon_days)
                pred_total_hist = float(row["mtd_sales"] + pred_remaining_hist)

                _append_result(
                    results,
                    target_month=target_month,
                    cutoff_day=cutoff_day,
                    platform=platform,
                    model_name="history_average",
                    pred_remaining_sales=pred_remaining_hist,
                    actual_remaining_sales=actual_remaining,
                    pred_total_sales=pred_total_hist,
                    actual_total_sales=actual_total,
                    abs_error_remaining=abs(pred_remaining_hist - actual_remaining),
                    ape_remaining=_safe_pct_err(pred_remaining_hist, actual_remaining),
                    bias_remaining=pred_remaining_hist - actual_remaining,
                    abs_error_total=abs(pred_total_hist - actual_total),
                    ape_total=_safe_pct_err(pred_total_hist, actual_total),
                    bias_total=pred_total_hist - actual_total,
                )

                if pd.notna(row.get("hist_share_p50")) and row["hist_share_p50"] > 0:
                    pred_total = predict_by_progress(row["mtd_sales"], row["hist_share_p50"])
                    pred_remaining = float(pred_total - row["mtd_sales"])
                    _append_result(
                        results,
                        target_month=target_month,
                        cutoff_day=cutoff_day,
                        platform=platform,
                        model_name="mtd_progress_p50",
                        pred_remaining_sales=pred_remaining,
                        actual_remaining_sales=actual_remaining,
                        pred_total_sales=pred_total,
                        actual_total_sales=actual_total,
                        abs_error_remaining=abs(pred_remaining - actual_remaining),
                        ape_remaining=_safe_pct_err(pred_remaining, actual_remaining),
                        bias_remaining=pred_remaining - actual_remaining,
                        abs_error_total=abs(pred_total - actual_total),
                        ape_total=_safe_pct_err(pred_total, actual_total),
                        bias_total=pred_total - actual_total,
                    )

                # prophet old logic for cutoff > 0
                month_start = pd.Timestamp(f'{target_month}-01')
                observed_end = month_start + pd.Timedelta(days=int(cutoff_day) - 1)
                pred_start = observed_end + pd.Timedelta(days=1)
                pred_end = month_start + pd.offsets.MonthEnd(0)
                pred_remaining_prophet = predict_prophet_oldlogic(
                    history_df[['date', 'sales']].copy().assign(
                        date=lambda d: pd.to_datetime(d['date'])
                    ).pipe(lambda d: pd.concat([
                        d,
                        daily[(daily['platform'] == platform) & (daily['year_month'] == target_month) & (daily['day_seq_in_month'] <= cutoff_day)][['date', 'sales']]
                    ], ignore_index=True).sort_values('date')),
                    monthly_pivot=monthly_pivot,
                    col=platform,
                    observed_actual=float(row['mtd_sales']),
                    target_month=target_month,
                    observed_end=observed_end,
                    pred_start=pred_start,
                    pred_end=pred_end,
                )
                if pd.notna(pred_remaining_prophet):
                    pred_total_prophet = float(row['mtd_sales'] + pred_remaining_prophet)
                    _append_result(
                        results,
                        target_month=target_month,
                        cutoff_day=cutoff_day,
                        platform=platform,
                        model_name="prophet_oldlogic",
                        pred_remaining_sales=float(pred_remaining_prophet),
                        actual_remaining_sales=actual_remaining,
                        pred_total_sales=pred_total_prophet,
                        actual_total_sales=actual_total,
                        abs_error_remaining=abs(float(pred_remaining_prophet) - actual_remaining),
                        ape_remaining=_safe_pct_err(float(pred_remaining_prophet), actual_remaining),
                        bias_remaining=float(pred_remaining_prophet) - actual_remaining,
                        abs_error_total=abs(pred_total_prophet - actual_total),
                        ape_total=_safe_pct_err(pred_total_prophet, actual_total),
                        bias_total=pred_total_prophet - actual_total,
                    )

                # daily_direct nodes (parameterized by cutoff_day)
                pred_total_gbdt_dd = predict_tree_daily_direct(
                    daily[daily['platform'] == platform].copy().sort_values('date'),
                    target_month=target_month,
                    cutoff_day=cutoff_day,
                    model_type='gbdt',
                    hist_share_p50=row.get('hist_share_p50'),
                )
                if pred_total_gbdt_dd is not None and pd.notna(pred_total_gbdt_dd):
                    pred_remaining_gbdt_dd = float(pred_total_gbdt_dd - row['mtd_sales'])
                    _append_result(
                        results,
                        target_month=target_month,
                        cutoff_day=cutoff_day,
                        platform=platform,
                        model_name=f"gbdt_daily_direct_c{cutoff_day}",
                        pred_remaining_sales=pred_remaining_gbdt_dd,
                        actual_remaining_sales=actual_remaining,
                        pred_total_sales=float(pred_total_gbdt_dd),
                        actual_total_sales=actual_total,
                        abs_error_remaining=abs(pred_remaining_gbdt_dd - actual_remaining),
                        ape_remaining=_safe_pct_err(pred_remaining_gbdt_dd, actual_remaining),
                        bias_remaining=pred_remaining_gbdt_dd - actual_remaining,
                        abs_error_total=abs(float(pred_total_gbdt_dd) - actual_total),
                        ape_total=_safe_pct_err(float(pred_total_gbdt_dd), actual_total),
                        bias_total=float(pred_total_gbdt_dd) - actual_total,
                    )

                pred_total_xgb_dd = predict_tree_daily_direct(
                    daily[daily['platform'] == platform].copy().sort_values('date'),
                    target_month=target_month,
                    cutoff_day=cutoff_day,
                    model_type='xgb',
                    hist_share_p50=row.get('hist_share_p50'),
                )
                if pred_total_xgb_dd is not None and pd.notna(pred_total_xgb_dd):
                    pred_remaining_xgb_dd = float(pred_total_xgb_dd - row['mtd_sales'])
                    _append_result(
                        results,
                        target_month=target_month,
                        cutoff_day=cutoff_day,
                        platform=platform,
                        model_name=f"xgboost_daily_direct_c{cutoff_day}",
                        pred_remaining_sales=pred_remaining_xgb_dd,
                        actual_remaining_sales=actual_remaining,
                        pred_total_sales=float(pred_total_xgb_dd),
                        actual_total_sales=actual_total,
                        abs_error_remaining=abs(pred_remaining_xgb_dd - actual_remaining),
                        ape_remaining=_safe_pct_err(pred_remaining_xgb_dd, actual_remaining),
                        bias_remaining=pred_remaining_xgb_dd - actual_remaining,
                        abs_error_total=abs(float(pred_total_xgb_dd) - actual_total),
                        ape_total=_safe_pct_err(float(pred_total_xgb_dd), actual_total),
                        bias_total=float(pred_total_xgb_dd) - actual_total,
                    )

                pred_total_prophet_dd = predict_prophet_daily_direct(
                    daily[daily['platform'] == platform].copy().sort_values('date'),
                    target_month=target_month,
                    cutoff_day=cutoff_day,
                )
                if pred_total_prophet_dd is not None and pd.notna(pred_total_prophet_dd):
                    pred_remaining_prophet_dd = float(pred_total_prophet_dd - row['mtd_sales'])
                    _append_result(
                        results,
                        target_month=target_month,
                        cutoff_day=cutoff_day,
                        platform=platform,
                        model_name=f"prophet_daily_direct_c{cutoff_day}",
                        pred_remaining_sales=pred_remaining_prophet_dd,
                        actual_remaining_sales=actual_remaining,
                        pred_total_sales=float(pred_total_prophet_dd),
                        actual_total_sales=actual_total,
                        abs_error_remaining=abs(pred_remaining_prophet_dd - actual_remaining),
                        ape_remaining=_safe_pct_err(pred_remaining_prophet_dd, actual_remaining),
                        bias_remaining=pred_remaining_prophet_dd - actual_remaining,
                        abs_error_total=abs(float(pred_total_prophet_dd) - actual_total),
                        ape_total=_safe_pct_err(float(pred_total_prophet_dd), actual_total),
                        bias_total=float(pred_total_prophet_dd) - actual_total,
                    )

                train_snapshot = snapshot[(snapshot["platform"] == platform) & (snapshot["target_month"] < target_month)].copy()
                train_snapshot = train_snapshot.copy()
                train_snapshot['baseline_total_p50'] = train_snapshot.apply(
                    lambda r: predict_by_progress(r['mtd_sales'], r['hist_share_p50'])
                    if pd.notna(r.get('hist_share_p50')) and r['hist_share_p50'] > 0 else pd.NA,
                    axis=1,
                )
                train_snapshot['baseline_remaining_p50'] = train_snapshot['baseline_total_p50'] - train_snapshot['mtd_sales']
                train_snapshot['residual_target'] = train_snapshot['target_remaining_sales'] - train_snapshot['baseline_remaining_p50']

                pred_row_resid = pd.DataFrame([row]).copy()
                pred_row_resid['baseline_total_p50'] = predict_by_progress(row['mtd_sales'], row['hist_share_p50']) if pd.notna(row.get('hist_share_p50')) and row['hist_share_p50'] > 0 else pd.NA
                pred_row_resid['baseline_remaining_p50'] = pred_row_resid['baseline_total_p50'] - pred_row_resid['mtd_sales']

                residual_features = FEATURES_CUTOFFN + ['baseline_remaining_p50']
                xgb_model = train_xgb(train_snapshot, residual_features, 'residual_target')
                pred_residual_xgb = predict_xgb(xgb_model, pred_row_resid).iloc[0]
                if pd.notna(pred_residual_xgb) and pd.notna(pred_row_resid.iloc[0]['baseline_remaining_p50']):
                    pred_remaining_xgb = max(float(pred_row_resid.iloc[0]['baseline_remaining_p50']) + float(pred_residual_xgb), 0.0)
                    pred_total_xgb = float(row["mtd_sales"] + pred_remaining_xgb)
                    _append_result(
                        results,
                        target_month=target_month,
                        cutoff_day=cutoff_day,
                        platform=platform,
                        model_name="xgboost_residual_p50_v3",
                        pred_remaining_sales=pred_remaining_xgb,
                        actual_remaining_sales=actual_remaining,
                        pred_total_sales=pred_total_xgb,
                        actual_total_sales=actual_total,
                        abs_error_remaining=abs(pred_remaining_xgb - actual_remaining),
                        ape_remaining=_safe_pct_err(pred_remaining_xgb, actual_remaining),
                        bias_remaining=pred_remaining_xgb - actual_remaining,
                        abs_error_total=abs(pred_total_xgb - actual_total),
                        ape_total=_safe_pct_err(pred_total_xgb, actual_total),
                        bias_total=pred_total_xgb - actual_total,
                    )

    if not results:
        result_df = pd.DataFrame(columns=MONTHLY_EVAL_COLS)
        summary = pd.DataFrame(columns=[
            'cutoff_day', 'platform', 'model_name', 'months',
            'mae_remaining', 'mape_remaining', 'bias_remaining',
            'mae_total', 'mape_total', 'bias_total'
        ])
    else:
        result_df = pd.DataFrame(results)[MONTHLY_EVAL_COLS].sort_values(
            ["cutoff_day", "platform", "target_month", "model_name"]
        )
        summary = (
            result_df.groupby(["cutoff_day", "platform", "model_name"], as_index=False)
            .agg(
                months=("target_month", "nunique"),
                mae_remaining=("abs_error_remaining", "mean"),
                mape_remaining=("ape_remaining", "mean"),
                bias_remaining=("bias_remaining", "mean"),
                mae_total=("abs_error_total", "mean"),
                mape_total=("ape_total", "mean"),
                bias_total=("bias_total", "mean"),
            )
            .sort_values(["cutoff_day", "platform", "mape_total", "mae_total"])
        )

    result_path = output_dir / "backtest_v1.csv"
    result_df.to_csv(result_path, index=False)
    summary_path = output_dir / "backtest_v1_summary.csv"
    summary.to_csv(summary_path, index=False)

    return result_df, summary
