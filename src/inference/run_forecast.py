"""预测入口：支持 brand + platform + cutoff_day 三维参数化。"""

from pathlib import Path
import pandas as pd

from src.data_prep.clean_data import clean_raw_data
from src.data_prep.build_calendar_feature import add_calendar_features
from src.feature_engineering.build_base_feature import build_base_features
from src.feature_engineering.build_mtd_feature import build_mtd_features
from src.feature_engineering.build_snapshot_feature import build_snapshot_features, add_historical_share_features, add_historical_ytd_share_features
from src.feature_engineering.build_month_level_feature import build_month_level_features
from src.models.baseline_mtd_progress import predict_by_progress
from src.models.baseline_ytd_progress import predict_by_ytd_progress
from src.models.baseline_history import predict_history_average
from src.models.baseline_growth import predict_growth, estimate_recent_yoy_growth
from src.models.cutoff0_enhanced import (
    predict_seasonal_event_adjusted_yoy,
    predict_similar_month_retrieval,
    predict_cutoff0_triplet_ensemble,
    apply_preheat_bias_correction,
    apply_fixed_preheat_uplift,
    DEFAULT_PREHEAT_UPLIFT,
)
from src.config_runtime import get_preheat_uplift
from src.models.xgb_model import train_xgb, predict_xgb, FEATURES_CUTOFF0, FEATURES_CUTOFFN
from src.models.prophet_model import predict_prophet_full_month
from src.models.prophet_oldlogic_model import predict_prophet_mixed
from src.models.daily_direct_models import predict_tree_daily_direct, predict_prophet_daily_direct
from src.inference.recommend_model import recommend_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = WORKSPACE_ROOT / "data" / "test_forecast_data.csv"
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


def _append_prediction(records: list[dict], entity_row: dict, model_name: str, pred_total, pred_remaining=None):
    if pd.isna(pred_total):
        return
    rec = dict(entity_row)
    rec['model_name'] = model_name
    rec['pred_total_sales'] = float(pred_total)
    rec['pred_remaining_sales'] = float(pred_remaining) if pred_remaining is not None and not pd.isna(pred_remaining) else pd.NA
    records.append(rec)


def _finalize_prediction_output(result_df: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    if result_df.empty:
        return result_df

    result_df = result_df.copy()
    result_df['recommended_model'] = result_df.apply(
        lambda r: recommend_model(
            r['platform'],
            cutoff_day,
            brand=r.get('brand_scope'),
            month=int(str(r.get('target_month'))[-2:]) if pd.notna(r.get('target_month')) else None,
            platform=r.get('platform_scope'),
        ),
        axis=1,
    )
    result_df['is_recommended'] = result_df['model_name'] == result_df['recommended_model']

    front_cols = [
        'target_month', 'cutoff_day', 'platform', 'brand_scope', 'platform_scope',
        'model_name', 'pred_total_sales', 'pred_remaining_sales',
        'recommended_model', 'is_recommended'
    ]
    rest_cols = [c for c in result_df.columns if c not in front_cols]
    return result_df[front_cols + rest_cols].sort_values(['platform', 'model_name']).reset_index(drop=True)


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

        history_snapshot_c0 = build_snapshot_features(history_daily, cutoff_day=0)
        history_snapshot_c0 = add_historical_ytd_share_features(history_snapshot_c0)

        results = []
        horizon_days = int(target_period.days_in_month)
        ly_period = target_period - 12

        for entity in sorted(daily['platform'].unique()):
            hist_p_daily = history_daily[history_daily['platform'] == entity].copy()
            hist_p_monthly = history_monthly[history_monthly['platform'] == entity].copy()
            if hist_p_daily.empty:
                continue

            base_row = {
                'target_month': target_month,
                'cutoff_day': 0,
                'platform': entity,
                'brand_scope': hist_p_daily['brand_scope'].dropna().iloc[0] if 'brand_scope' in hist_p_daily.columns and len(hist_p_daily) else brand,
                'platform_scope': hist_p_daily['platform_scope'].dropna().iloc[0] if 'platform_scope' in hist_p_daily.columns and len(hist_p_daily) else platform,
            }

            pred_total_hist = predict_history_average(hist_p_daily, horizon_days=horizon_days)
            _append_prediction(results, base_row, 'history_average', pred_total_hist, pred_total_hist)

            hist_p_snapshot_c0 = history_snapshot_c0[history_snapshot_c0['platform'] == entity].copy()
            target_ytd_share_row = hist_p_snapshot_c0[hist_p_snapshot_c0['target_month'] == history_until]
            if not target_ytd_share_row.empty and pd.notna(target_ytd_share_row.iloc[-1].get('hist_ytd_share_p50')) and pd.notna(target_ytd_share_row.iloc[-1].get('hist_month_share_p50')):
                pred_year_total_ytd = predict_by_ytd_progress(
                    float(target_ytd_share_row.iloc[-1]['ytd_sales']),
                    float(target_ytd_share_row.iloc[-1]['hist_ytd_share_p50'])
                )
                pred_total_ytd_to_month = pred_year_total_ytd * float(target_ytd_share_row.iloc[-1]['hist_month_share_p50'])
                _append_prediction(results, base_row, 'ytd_to_month_share_p50', pred_total_ytd_to_month, pred_total_ytd_to_month)

            ly_row = hist_p_monthly[hist_p_monthly['period'] == ly_period]
            pred_total_ly = pd.NA
            pred_total_yoy = pd.NA
            if not ly_row.empty:
                ly_value = float(ly_row.iloc[0]['month_total_sales'])
                pred_total_ly = ly_value
                _append_prediction(results, base_row, 'last_year_same_month', ly_value, ly_value)

                yoy_growth = estimate_recent_yoy_growth(
                    hist_p_monthly[['year_month', 'month_total_sales']],
                    target_month=target_month,
                    window=3,
                )
                if yoy_growth is not None:
                    pred_total_yoy = predict_growth(ly_value, yoy_growth)
                    _append_prediction(results, base_row, 'yoy_growth_extrapolation', pred_total_yoy, pred_total_yoy)

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

            pred_total_seasonal_yoy = predict_seasonal_event_adjusted_yoy(hist_p_monthly, pred_row)
            _append_prediction(results, base_row, 'seasonal_event_adjusted_yoy', pred_total_seasonal_yoy, pred_total_seasonal_yoy)

            hist_with_base_pred = hist_p_monthly.copy()
            if not hist_with_base_pred.empty:
                pred_hist_vals = []
                for _, hist_row in hist_with_base_pred.iterrows():
                    hist_target_row = pd.DataFrame([hist_row.to_dict()])
                    pred_hist_vals.append(predict_seasonal_event_adjusted_yoy(hist_with_base_pred, hist_target_row))
                hist_with_base_pred['seasonal_event_adjusted_yoy_pred'] = pred_hist_vals

            pred_total_seasonal_yoy_preheat = apply_preheat_bias_correction(
                pred_total_seasonal_yoy,
                hist_with_base_pred if 'hist_with_base_pred' in locals() else hist_p_monthly,
                pred_row,
            )
            _append_prediction(results, base_row, 'seasonal_event_adjusted_yoy_preheat_corrected', pred_total_seasonal_yoy_preheat, pred_total_seasonal_yoy_preheat)

            runtime_uplift = get_preheat_uplift(
                brand=base_row.get('brand_scope'),
                month=target_period.month,
                platform=base_row.get('platform_scope'),
            )
            pred_total_fixed_default = apply_fixed_preheat_uplift(pred_total_seasonal_yoy, pred_row, runtime_uplift)
            _append_prediction(results, base_row, 'seasonal_preheat_yoy', pred_total_fixed_default, pred_total_fixed_default)

            for uplift in [1.10, 1.15, 1.20, 1.25]:
                pred_total_fixed_uplift = apply_fixed_preheat_uplift(pred_total_seasonal_yoy, pred_row, uplift)
                uplift_name = f"seasonal_event_adjusted_yoy_preheat_uplift_{int(round(uplift * 100))}"
                _append_prediction(results, base_row, uplift_name, pred_total_fixed_uplift, pred_total_fixed_uplift)

            pred_total_ensemble = predict_cutoff0_triplet_ensemble(
                pred_last_year_same_month=pred_total_ly,
                pred_yoy_growth_extrapolation=pred_total_yoy,
                pred_seasonal_event_adjusted_yoy=pred_total_seasonal_yoy,
                target_row=pred_row,
                entity_name=entity,
            )
            _append_prediction(results, base_row, 'cutoff0_triplet_ensemble', pred_total_ensemble, pred_total_ensemble)

            pred_total_similar = predict_similar_month_retrieval(hist_p_monthly, pred_row)
            _append_prediction(results, base_row, 'similar_month_retrieval', pred_total_similar, pred_total_similar)

            xgb_model = train_xgb(hist_p_monthly, FEATURES_CUTOFF0, 'month_total_sales')
            pred_total_gbdt_dd = predict_tree_daily_direct(
                daily[daily['platform'] == entity].copy().sort_values('date'),
                target_month=target_month,
                cutoff_day=0,
                model_type='gbdt',
            )
            _append_prediction(results, base_row, 'gbdt_daily_direct_c0', pred_total_gbdt_dd, pred_total_gbdt_dd)

            pred_total_xgb_dd = predict_tree_daily_direct(
                daily[daily['platform'] == entity].copy().sort_values('date'),
                target_month=target_month,
                cutoff_day=0,
                model_type='xgb',
            )
            _append_prediction(results, base_row, 'xgboost_daily_direct_c0', pred_total_xgb_dd, pred_total_xgb_dd)

            pred_total_prophet_dd = predict_prophet_daily_direct(
                daily[daily['platform'] == entity].copy().sort_values('date'),
                target_month=target_month,
                cutoff_day=0,
            )
            _append_prediction(results, base_row, 'prophet_daily', pred_total_prophet_dd, pred_total_prophet_dd)

        result_df = _finalize_prediction_output(pd.DataFrame(results), cutoff_day=0)
        result_df.to_csv(output_dir / 'feature_store' / f'forecast_fullmonth_prediction_{target_month}_{brand}_{platform}.csv', index=False)
        return result_df

    snapshot = build_snapshot_features(daily, cutoff_day=cutoff_day)
    scope_map = daily[['platform', 'brand_scope', 'platform_scope']].drop_duplicates()
    snapshot = snapshot.merge(scope_map, on='platform', how='left')
    snapshot = add_historical_share_features(snapshot)
    snapshot = snapshot[snapshot['target_month'] == target_month].copy()

    if snapshot.empty:
        raise ValueError(f'未找到 target_month={target_month} 的 snapshot (brand={brand}, platform={platform})')

    results = []
    monthly_pivot = (
        daily.groupby(['year_month', 'platform'], as_index=False)['sales']
        .sum()
        .pivot(index='year_month', columns='platform', values='sales')
        .sort_index()
    )

    for _, row in snapshot.iterrows():
        entity = row['platform']
        base_row = {
            'target_month': target_month,
            'cutoff_day': cutoff_day,
            'platform': entity,
            'brand_scope': row.get('brand_scope', brand),
            'platform_scope': row.get('platform_scope', platform),
            'mtd_sales': row['mtd_sales'],
            'days_elapsed': row['days_elapsed'],
            'days_remaining': row['days_remaining'],
        }

        pred_total_mtd = predict_by_progress(row['mtd_sales'], row['hist_share_p50']) if pd.notna(row.get('hist_share_p50')) and row['hist_share_p50'] > 0 else pd.NA
        pred_remaining_mtd = pred_total_mtd - row['mtd_sales'] if pd.notna(pred_total_mtd) else pd.NA
        _append_prediction(results, base_row, 'mtd_progress_p50', pred_total_mtd, pred_remaining_mtd)

        hist = daily[(daily['platform'] == entity) & (daily['year_month'] < target_month)]
        pred_remaining_hist = predict_history_average(hist, horizon_days=int(row['days_remaining']))
        pred_total_hist = row['mtd_sales'] + pred_remaining_hist if not pd.isna(pred_remaining_hist) else pd.NA
        _append_prediction(results, base_row, 'history_average', pred_total_hist, pred_remaining_hist)

        month_start = pd.Timestamp(f'{target_month}-01')
        observed_end = month_start + pd.Timedelta(days=int(cutoff_day) - 1)
        pred_start = observed_end + pd.Timedelta(days=1)
        pred_end = month_start + pd.offsets.MonthEnd(0)
        train_daily = hist[['date', 'sales']].copy().assign(date=lambda d: pd.to_datetime(d['date']))
        current_obs = daily[(daily['platform'] == entity) & (daily['year_month'] == target_month) & (daily['day_seq_in_month'] <= cutoff_day)][['date', 'sales']]
        pred_remaining_prophet = predict_prophet_mixed(
            pd.concat([train_daily, current_obs], ignore_index=True).sort_values('date'),
            monthly_pivot=monthly_pivot,
            col=entity,
            observed_actual=float(row['mtd_sales']),
            target_month=target_month,
            observed_end=observed_end,
            pred_start=pred_start,
            pred_end=pred_end,
        )
        pred_total_prophet = row['mtd_sales'] + pred_remaining_prophet if not pd.isna(pred_remaining_prophet) else pd.NA
        _append_prediction(results, base_row, 'prophet_mixed', pred_total_prophet, pred_remaining_prophet)

        pred_total_gbdt_dd = predict_tree_daily_direct(
            daily[daily['platform'] == entity].copy().sort_values('date'),
            target_month=target_month,
            cutoff_day=cutoff_day,
            model_type='gbdt',
            hist_share_p50=row.get('hist_share_p50'),
        )
        pred_remaining_gbdt_dd = pred_total_gbdt_dd - row['mtd_sales'] if pred_total_gbdt_dd is not None and not pd.isna(pred_total_gbdt_dd) else pd.NA
        _append_prediction(results, base_row, f'gbdt_daily_direct_c{cutoff_day}', pred_total_gbdt_dd, pred_remaining_gbdt_dd)

        pred_total_xgb_dd = predict_tree_daily_direct(
            daily[daily['platform'] == entity].copy().sort_values('date'),
            target_month=target_month,
            cutoff_day=cutoff_day,
            model_type='xgb',
            hist_share_p50=row.get('hist_share_p50'),
        )
        pred_remaining_xgb_dd = pred_total_xgb_dd - row['mtd_sales'] if pred_total_xgb_dd is not None and not pd.isna(pred_total_xgb_dd) else pd.NA
        _append_prediction(results, base_row, f'xgboost_daily_direct_c{cutoff_day}', pred_total_xgb_dd, pred_remaining_xgb_dd)

        pred_total_prophet_dd = predict_prophet_daily_direct(
            daily[daily['platform'] == entity].copy().sort_values('date'),
            target_month=target_month,
            cutoff_day=cutoff_day,
        )
        pred_remaining_prophet_dd = pred_total_prophet_dd - row['mtd_sales'] if pred_total_prophet_dd is not None and not pd.isna(pred_total_prophet_dd) else pd.NA
        _append_prediction(results, base_row, 'prophet_daily', pred_total_prophet_dd, pred_remaining_prophet_dd)

        train_snapshot = snapshot[(snapshot['platform'] == entity) & (snapshot['target_month'] < target_month)].copy()
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
            pred_total_xgb = float(row['mtd_sales'] + pred_remaining_xgb)
        else:
            pred_remaining_xgb = pd.NA
            pred_total_xgb = pd.NA
        _append_prediction(results, base_row, 'xgboost_residual', pred_total_xgb, pred_remaining_xgb)

    result_df = _finalize_prediction_output(pd.DataFrame(results), cutoff_day=cutoff_day)
    result_df.to_csv(output_dir / 'feature_store' / f'forecast_feature_snapshot_{target_month}_cutoff{cutoff_day}_{brand}_{platform}.csv', index=False)
    return result_df
