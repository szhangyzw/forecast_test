from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.inference.run_forecast import prepare_daily_feature_table
from src.feature_engineering.build_snapshot_feature import build_snapshot_features, add_historical_share_features
from src.models.xgb_model import train_xgb, predict_xgb
from src.feature_engineering.build_event_calendar_feature import add_event_features_to_month_df
from src.models.baseline_mtd_progress import predict_by_progress

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / 'data' / 'raw' / 'test_forecast_data.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'backtest_result'
CUTOFFS = (5, 10, 15, 20, 25)

FEATURES_RESIDUAL = [
    'cutoff_day', 'month', 'days_in_month', 'days_elapsed', 'days_remaining',
    'mtd_sales', 'mtd_avg_sales', 'mtd_progress',
    'last_7d_avg', 'last_30d_avg',
    'ly_same_month_same_day_mtd', 'ly_same_month_total',
    'hist_share_p50', 'hist_share_mean',
    'baseline_remaining_p50', 'baseline_total_p50',
    'is_cny_impact_month', 'pre_cny_flag', 'in_cny_flag', 'post_cny_flag',
    'is_618_impact_month', 'pre_618_flag', 'in_618_flag',
    'is_double11_impact_month', 'pre_double11_flag', 'in_double11_flag',
]


def _get_complete_months(daily: pd.DataFrame) -> list[str]:
    month_end = (
        daily.groupby(['platform', 'year_month'], as_index=False)
        .agg(max_day=('day_seq_in_month', 'max'), days_in_month=('days_in_month', 'max'))
    )
    complete = month_end[month_end['max_day'] >= month_end['days_in_month']].copy()
    complete_total = complete[complete['platform'] == 'total']['year_month'].tolist()
    return sorted(complete_total)


def _build_all_snapshots(daily: pd.DataFrame, cutoffs=CUTOFFS) -> pd.DataFrame:
    parts = []
    for cutoff in cutoffs:
        snap = build_snapshot_features(daily, cutoff_day=cutoff)
        snap = add_historical_share_features(snap)

        snap['baseline_total_p50'] = snap.apply(
            lambda r: predict_by_progress(r['mtd_sales'], r['hist_share_p50'])
            if pd.notna(r.get('hist_share_p50')) and r['hist_share_p50'] > 0 else pd.NA,
            axis=1,
        )
        snap['baseline_remaining_p50'] = snap['baseline_total_p50'] - snap['mtd_sales']
        snap['residual_target'] = snap['target_remaining_sales'] - snap['baseline_remaining_p50']

        snap_for_event = snap.rename(columns={'target_month': 'year_month'})
        snap_for_event = add_event_features_to_month_df(snap_for_event)
        snap = snap_for_event.rename(columns={'year_month': 'target_month'})
        parts.append(snap)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(['platform', 'target_month', 'cutoff_day']).reset_index(drop=True)


def run_residual_xgb_backtest(output_dir: str | Path = DEFAULT_OUTPUT, exclude_months=None):
    if exclude_months is None:
        exclude_months = ['2026-03']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = prepare_daily_feature_table(DEFAULT_INPUT)
    complete_months = [m for m in _get_complete_months(daily) if m not in set(exclude_months)]
    all_snap = _build_all_snapshots(daily, CUTOFFS)
    all_snap = all_snap[all_snap['target_month'].isin(complete_months)].copy()

    rows = []
    for platform in sorted(all_snap['platform'].unique()):
        p_all = all_snap[all_snap['platform'] == platform].copy()
        for target_month in complete_months:
            for cutoff in CUTOFFS:
                row = p_all[(p_all['target_month'] == target_month) & (p_all['cutoff_day'] == cutoff)].copy()
                if row.empty:
                    continue
                train = p_all[p_all['target_month'] < target_month].copy()
                model = train_xgb(train, FEATURES_RESIDUAL, 'residual_target')
                pred_residual = predict_xgb(model, row).iloc[0]
                if pd.isna(pred_residual):
                    continue
                pred_residual = float(pred_residual)
                baseline_remaining = float(row.iloc[0]['baseline_remaining_p50'])
                pred_remaining = max(baseline_remaining + pred_residual, 0.0)
                actual_remaining = float(row.iloc[0]['target_remaining_sales'])
                pred_total = float(row.iloc[0]['mtd_sales']) + pred_remaining
                actual_total = float(row.iloc[0]['month_total_sales'])

                rows.append({
                    'target_month': target_month,
                    'cutoff_day': cutoff,
                    'platform': platform,
                    'model_name': 'xgboost_residual',
                    'baseline_remaining_p50': baseline_remaining,
                    'pred_residual': pred_residual,
                    'pred_remaining_sales': pred_remaining,
                    'actual_remaining_sales': actual_remaining,
                    'pred_total_sales': pred_total,
                    'actual_total_sales': actual_total,
                    'abs_error_remaining': abs(pred_remaining - actual_remaining),
                    'ape_remaining': None if actual_remaining == 0 else abs(pred_remaining - actual_remaining) / actual_remaining,
                    'bias_remaining': pred_remaining - actual_remaining,
                    'abs_error_total': abs(pred_total - actual_total),
                    'ape_total': None if actual_total == 0 else abs(pred_total - actual_total) / actual_total,
                    'bias_total': pred_total - actual_total,
                })

    result_df = pd.DataFrame(rows).sort_values(['cutoff_day', 'platform', 'target_month']) if rows else pd.DataFrame()
    summary_df = (
        result_df.groupby(['cutoff_day', 'platform', 'model_name'], as_index=False)
        .agg(
            months=('target_month', 'nunique'),
            mae_remaining=('abs_error_remaining', 'mean'),
            mape_remaining=('ape_remaining', 'mean'),
            bias_remaining=('bias_remaining', 'mean'),
            mae_total=('abs_error_total', 'mean'),
            mape_total=('ape_total', 'mean'),
            bias_total=('bias_total', 'mean'),
        )
        .sort_values(['cutoff_day', 'platform', 'mape_total', 'mae_total'])
    ) if not result_df.empty else pd.DataFrame()

    if not result_df.empty:
        result_df.to_csv(output_dir / 'residual_xgb_backtest.csv', index=False)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / 'residual_xgb_summary.csv', index=False)

    return result_df, summary_df


if __name__ == '__main__':
    result_df, summary_df = run_residual_xgb_backtest()
    print('=== RESIDUAL XGB SUMMARY ===')
    print(summary_df.to_string(index=False))
