"""预测入口：支持 brand + platform + cutoff_day 三维参数化。"""

from pathlib import Path
import pandas as pd

from src.data_prep.clean_data import clean_raw_data
from src.data_prep.aggregate_daily import aggregate_daily
from src.data_prep.build_calendar_feature import add_calendar_features
from src.feature_engineering.build_base_feature import build_base_features
from src.feature_engineering.build_mtd_feature import build_mtd_features
from src.feature_engineering.build_snapshot_feature import build_snapshot_features, add_historical_share_features
from src.feature_engineering.build_month_level_feature import build_month_level_features
from src.models.baseline_mtd_progress import predict_by_progress
from src.models.baseline_history import predict_history_average
from src.models.baseline_growth import predict_growth, estimate_recent_yoy_growth
from src.models.xgb_model import train_xgb, predict_xgb, FEATURES_CUTOFF0


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "test_forecast_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"


def _normalize_choice(value: str | None, default: str = 'total') -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def _parse_scope_values(value: str | None, available: list[str], default: str = 'total') -> list[str]:
    value = _normalize_choice(value, default)
    if value == 'all':
        return list(available)
    if ',' in value:
        return [x.strip() for x in value.split(',') if x.strip()]
    return [value]


def _build_entity_views(raw: pd.DataFrame, brand: str = 'total', platform: str = 'total') -> pd.DataFrame:
    brand = _normalize_choice(brand, 'total')
    platform = _normalize_choice(platform, 'total')
    parts = []

    brands = sorted(raw['brand'].dropna().astype(str).unique().tolist()) if 'brand' in raw.columns else []
    platforms = sorted(raw['platform'].dropna().astype(str).unique().tolist())

    brand_values = _parse_scope_values(brand, brands, 'total')
    platform_values = _parse_scope_values(platform, platforms, 'total')

    for b in brand_values:
        for p in platform_values:
            df = raw.copy()
            brand_label = b
            platform_label = p

            if b != 'total':
                df = df[df['brand'] == b].copy()
            if p != 'total':
                df = df[df['platform'] == p].copy()
            if df.empty:
                continue

            if b == 'total' and p == 'total':
                entity = 'total'
            elif b == 'total':
                entity = p
            elif p == 'total':
                entity = f'brand:{b}'
            else:
                entity = f'{b}@{p}'

            out = df.copy()
            out['entity'] = entity
            out['brand_scope'] = brand_label
            out['platform_scope'] = platform_label
            parts.append(out)

    if not parts:
        return pd.DataFrame(columns=list(raw.columns) + ['entity', 'brand_scope', 'platform_scope'])
    return pd.concat(parts, ignore_index=True)


def prepare_daily_feature_table(
    input_path: str | Path = DEFAULT_INPUT,
    brand: str = 'total',
    platform: str = 'total',
) -> pd.DataFrame:
    raw = clean_raw_data(input_path)
    raw = _build_entity_views(raw, brand=brand, platform=platform)
    if raw.empty:
        return raw

    agg = (
        raw.groupby(['date', 'entity'], as_index=False)['sales']
        .sum()
        .rename(columns={'entity': 'platform'})
        .sort_values(['platform', 'date'])
    )

    scope_map = raw[['entity', 'brand_scope', 'platform_scope']].drop_duplicates().rename(columns={'entity': 'platform'})
    daily = add_calendar_features(agg)
    daily = build_base_features(daily)
    daily = build_mtd_features(daily)

    month_total = (
        daily.groupby(['platform', 'year_month'], as_index=False)['sales']
        .sum()
        .rename(columns={'sales': 'month_total_sales'})
    )
    daily = daily.merge(month_total, on=['platform', 'year_month'], how='left')
    daily = daily.merge(scope_map, on='platform', how='left')
    return daily


def run_forecast(
    target_month: str,
    cutoff_day: int = 15,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    brand: str = 'total',
    platform: str = 'total',
):
    output_dir = Path(output_dir)
    (output_dir / 'processed').mkdir(parents=True, exist_ok=True)
    (output_dir / 'feature_store').mkdir(parents=True, exist_ok=True)

    daily = prepare_daily_feature_table(DEFAULT_INPUT, brand=brand, platform=platform)
    if daily.empty:
        raise ValueError(f'未找到可用数据：brand={brand}, platform={platform}。如果传多个值，请用逗号分隔，例如 brand=A,B,C')
    daily.to_csv(output_dir / 'processed' / 'sales_daily_agg_features.csv', index=False)

    if cutoff_day == 0:
        target_period = pd.Period(target_month, freq='M')
        history_until = str(target_period - 1)
        history_daily = daily[daily['year_month'] <= history_until].copy()
        history_monthly = build_month_level_features(history_daily)
        history_monthly['period'] = pd.PeriodIndex(history_monthly['year_month'], freq='M')

        results = []
        horizon_days = int(target_period.days_in_month)
        ly_period = target_period - 12

        for entity in sorted(daily['platform'].unique()):
            hist_p_daily = history_daily[history_daily['platform'] == entity]
            hist_p_monthly = history_monthly[history_monthly['platform'] == entity].copy()
            if hist_p_daily.empty:
                continue

            row = {
                'target_month': target_month,
                'cutoff_day': 0,
                'platform': entity,
                'brand_scope': hist_p_daily['brand_scope'].dropna().iloc[0] if 'brand_scope' in hist_p_daily.columns and len(hist_p_daily) else brand,
                'platform_scope': hist_p_daily['platform_scope'].dropna().iloc[0] if 'platform_scope' in hist_p_daily.columns and len(hist_p_daily) else platform,
                'pred_total_history_average': predict_history_average(hist_p_daily, horizon_days=horizon_days),
            }

            ly_row = hist_p_monthly[hist_p_monthly['period'] == ly_period]
            if not ly_row.empty:
                ly_value = float(ly_row.iloc[0]['month_total_sales'])
                row['pred_total_last_year_same_month'] = ly_value
                yoy_growth = estimate_recent_yoy_growth(
                    hist_p_monthly[['year_month', 'month_total_sales']],
                    target_month=target_month,
                    window=3,
                )
                if yoy_growth is not None:
                    row['pred_total_yoy_growth_extrapolation'] = predict_growth(ly_value, yoy_growth)

            pred_row = pd.DataFrame([{
                'platform': entity,
                'year': target_period.year,
                'month': target_period.month,
                'year_month': target_month,
                'days_in_month': target_period.days_in_month,
                'month_total_sales': pd.NA,
                'last_month_total': hist_p_monthly['month_total_sales'].iloc[-1] if len(hist_p_monthly) >= 1 else pd.NA,
                'last_3month_avg': hist_p_monthly['month_total_sales'].tail(3).mean() if len(hist_p_monthly) >= 1 else pd.NA,
                'last_6month_avg': hist_p_monthly['month_total_sales'].tail(6).mean() if len(hist_p_monthly) >= 1 else pd.NA,
                'ly_same_month_total': float(ly_row.iloc[0]['month_total_sales']) if not ly_row.empty else pd.NA,
                'recent_yoy_growth': estimate_recent_yoy_growth(
                    hist_p_monthly[['year_month', 'month_total_sales']], target_month=target_month, window=3
                ),
                'last_3month_yoy_avg': hist_p_monthly['last_3month_yoy_avg'].iloc[-1] if 'last_3month_yoy_avg' in hist_p_monthly.columns and len(hist_p_monthly) >= 1 else pd.NA,
                'last_6month_std': hist_p_monthly['last_6month_std'].iloc[-1] if 'last_6month_std' in hist_p_monthly.columns and len(hist_p_monthly) >= 1 else pd.NA,
                'month_num_in_year': target_period.month,
            }])
            from src.feature_engineering.build_event_calendar_feature import add_event_features_to_month_df
            pred_row = add_event_features_to_month_df(pred_row)
            xgb_model = train_xgb(hist_p_monthly, FEATURES_CUTOFF0, 'month_total_sales')
            xgb_pred = predict_xgb(xgb_model, pred_row).iloc[0]
            row['pred_total_xgboost_v2_event'] = xgb_pred

            results.append(row)

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_dir / 'feature_store' / f'forecast_fullmonth_prediction_{target_month}_{brand}_{platform}.csv', index=False)
        return result_df

    snapshot = build_snapshot_features(daily, cutoff_day=cutoff_day)
    snapshot = add_historical_share_features(snapshot)
    snapshot = snapshot[snapshot['target_month'] == target_month].copy()

    if snapshot.empty:
        raise ValueError(f'未找到 target_month={target_month} 的 snapshot (brand={brand}, platform={platform})')

    snapshot['pred_total_mtd_progress'] = snapshot.apply(
        lambda r: predict_by_progress(r['mtd_sales'], r['hist_share_p50'])
        if pd.notna(r.get('hist_share_p50')) and r['hist_share_p50'] > 0 else pd.NA,
        axis=1,
    )
    snapshot['pred_remaining_mtd_progress'] = snapshot['pred_total_mtd_progress'] - snapshot['mtd_sales']

    hist_preds = []
    for _, row in snapshot.iterrows():
        hist = daily[(daily['platform'] == row['platform']) & (daily['year_month'] < target_month)]
        pred_remaining = predict_history_average(hist, horizon_days=int(row['days_remaining']))
        hist_preds.append(pred_remaining)
    snapshot['pred_remaining_history_average'] = hist_preds
    snapshot['pred_total_history_average'] = snapshot['mtd_sales'] + snapshot['pred_remaining_history_average']

    snapshot.to_csv(output_dir / 'feature_store' / f'forecast_feature_snapshot_{target_month}_cutoff{cutoff_day}_{brand}_{platform}.csv', index=False)
    return snapshot
