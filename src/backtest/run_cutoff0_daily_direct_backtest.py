from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.inference.run_forecast import prepare_daily_feature_table
from src.models.xgb_model import build_xgb_model
from src.models.prophet_model import build_prophet_holidays

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / 'data' / 'raw' / 'test_forecast_data.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'backtest_result'


def _get_complete_months(daily: pd.DataFrame) -> list[str]:
    month_end = (
        daily.groupby(['platform', 'year_month'], as_index=False)
        .agg(max_day=('day_seq_in_month', 'max'), days_in_month=('days_in_month', 'max'))
    )
    complete = month_end[month_end['max_day'] >= month_end['days_in_month']].copy()
    complete_total = complete[complete['platform'] == 'total']['year_month'].tolist()
    return sorted(complete_total)


def _month_event_flags(month_df: pd.DataFrame) -> pd.DataFrame:
    from src.feature_engineering.build_event_calendar_feature import add_event_features_to_month_df
    tmp = month_df[['year_month', 'year', 'month']].drop_duplicates().copy()
    tmp['days_in_month'] = pd.PeriodIndex(tmp['year_month'], freq='M').days_in_month
    tmp = add_event_features_to_month_df(tmp)
    return tmp[['year_month','is_cny_impact_month','pre_cny_flag','in_cny_flag','post_cny_flag','is_618_impact_month','pre_618_flag','in_618_flag','is_double11_impact_month','pre_double11_flag','in_double11_flag']]


def build_daily_direct_features(platform_daily: pd.DataFrame) -> pd.DataFrame:
    df = platform_daily.copy().sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
    df['is_month_end'] = (df['days_in_month'] - df['day_of_month'] < 3).astype(int)
    df['day_progress'] = df['day_of_month'] / df['days_in_month']

    monthly = (
        df.groupby('year_month', as_index=False)['sales'].sum()
        .rename(columns={'sales':'month_total'})
        .sort_values('year_month')
    )
    monthly['period'] = pd.PeriodIndex(monthly['year_month'], freq='M')
    monthly['last_month_total'] = monthly['month_total'].shift(1)
    monthly['last_3month_avg_total'] = monthly['month_total'].shift(1).rolling(3, min_periods=1).mean()
    monthly['last_6month_avg_total'] = monthly['month_total'].shift(1).rolling(6, min_periods=1).mean()

    ly_map = monthly[['period','month_total']].copy()
    ly_map['period_plus1y'] = ly_map['period'] + 12
    ly_map = ly_map.rename(columns={'month_total':'ly_same_month_total'})[['period_plus1y','ly_same_month_total']]
    monthly = monthly.merge(ly_map, left_on='period', right_on='period_plus1y', how='left').drop(columns=['period_plus1y'])
    monthly['ly_same_month_avg_daily'] = monthly['ly_same_month_total'] / monthly['period'].dt.days_in_month

    recent_daily = []
    for ym in monthly['year_month']:
        month_start = pd.Timestamp(f'{ym}-01')
        hist = df[df['date'] < month_start].copy()
        recent_daily.append(hist.tail(90)['sales'].mean() if len(hist) else np.nan)
    monthly['recent_90d_avg_daily'] = recent_daily

    event_flags = _month_event_flags(df[['year_month','year','month']].drop_duplicates())
    monthly = monthly.merge(event_flags, on='year_month', how='left')

    df = df.merge(monthly.drop(columns=['period']), on='year_month', how='left')
    if 'ly_same_month_total_y' in df.columns:
        df['ly_same_month_total'] = df['ly_same_month_total_y']
    elif 'ly_same_month_total_x' in df.columns:
        df['ly_same_month_total'] = df['ly_same_month_total_x']

    ly_daily = df[['date','sales']].copy()
    ly_daily['date_plus1y'] = ly_daily['date'] + pd.DateOffset(years=1)
    ly_daily = ly_daily.rename(columns={'sales':'ly_same_day_sales'})[['date_plus1y','ly_same_day_sales']]
    df = df.merge(ly_daily, left_on='date', right_on='date_plus1y', how='left').drop(columns=['date_plus1y'])

    return df


def predict_xgb_daily_direct(platform_daily: pd.DataFrame, target_month: str) -> pd.DataFrame | None:
    feat = build_daily_direct_features(platform_daily)
    train = feat[feat['year_month'] < target_month].copy()
    test = feat[feat['year_month'] == target_month].copy().sort_values('date')
    if train.empty or test.empty:
        return None

    feature_cols = [
        'month','day_of_month','day_of_week','is_weekend','is_month_start','is_month_end','day_progress','days_in_month',
        'last_month_total','last_3month_avg_total','last_6month_avg_total','ly_same_month_total','ly_same_month_avg_daily','recent_90d_avg_daily','ly_same_day_sales',
        'is_cny_impact_month','pre_cny_flag','in_cny_flag','post_cny_flag','is_618_impact_month','pre_618_flag','in_618_flag','is_double11_impact_month','pre_double11_flag','in_double11_flag'
    ]
    train = train.dropna(subset=[c for c in feature_cols if c in train.columns] + ['sales']).copy()
    test = test.dropna(subset=[c for c in feature_cols if c in test.columns]).copy()
    if len(train) < 90 or test.empty:
        return None

    model = build_xgb_model(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(train[feature_cols], train['sales'])
    test['pred_daily_sales'] = np.clip(model.predict(test[feature_cols]), 0, None)
    return test[['date','year_month','sales','pred_daily_sales']].rename(columns={'sales':'actual_daily_sales'})


def predict_prophet_daily_direct(platform_daily: pd.DataFrame, target_month: str) -> pd.DataFrame | None:
    from prophet import Prophet
    target_start = pd.Timestamp(f'{target_month}-01')
    target_end = target_start + pd.offsets.MonthEnd(0)

    train = platform_daily[platform_daily['date'] < target_start][['date','sales']].rename(columns={'date':'ds','sales':'y'}).copy()
    if len(train) < 60:
        return None

    holidays = build_prophet_holidays()
    y_max = float(train['y'].max()) if len(train) else 0.0
    y_q95 = float(train['y'].quantile(0.95)) if len(train) else 0.0
    cap_value = max(y_max, y_q95) * 1.8 if max(y_max, y_q95) > 0 else 1.0
    train['cap'] = cap_value
    train['floor'] = 0.0

    model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5.0,
        holidays=holidays,
        holidays_prior_scale=10.0,
    )
    model.fit(train)

    future = pd.DataFrame({'ds': pd.date_range(target_start, target_end, freq='D')})
    future['cap'] = cap_value
    future['floor'] = 0.0
    fcst = model.predict(future)

    actual = platform_daily[(platform_daily['date'] >= target_start) & (platform_daily['date'] <= target_end)][['date','sales']].copy()
    if actual.empty:
        return None
    out = actual.rename(columns={'date':'ds','sales':'actual_daily_sales'}).merge(fcst[['ds','yhat']], on='ds', how='left')
    out['pred_daily_sales'] = out['yhat'].clip(lower=0)
    return out.rename(columns={'ds':'date'})[['date','actual_daily_sales','pred_daily_sales']].assign(year_month=target_month)


def run_cutoff0_daily_direct_backtest(output_dir: str | Path = DEFAULT_OUTPUT, exclude_months=None):
    if exclude_months is None:
        exclude_months = ['2026-03']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = prepare_daily_feature_table(DEFAULT_INPUT)
    complete_months = [m for m in _get_complete_months(daily) if m not in set(exclude_months)]

    monthly_rows = []
    daily_rows = []
    for platform in sorted(daily['platform'].unique()):
        p_daily = daily[daily['platform'] == platform].copy().sort_values('date')
        for target_month in complete_months:
            xgb_out = predict_xgb_daily_direct(p_daily, target_month)
            if xgb_out is not None:
                daily_rows.append(xgb_out.assign(platform=platform, model_name='xgboost_daily_direct_c0'))
                pred_total = float(xgb_out['pred_daily_sales'].sum())
                actual_total = float(xgb_out['actual_daily_sales'].sum())
                monthly_rows.append({
                    'target_month': target_month, 'cutoff_day': 0, 'platform': platform, 'model_name': 'xgboost_daily_direct_c0',
                    'pred_total_sales': pred_total, 'actual_total_sales': actual_total,
                    'abs_error_total': abs(pred_total-actual_total), 'ape_total': None if actual_total == 0 else abs(pred_total-actual_total)/actual_total,
                    'bias_total': pred_total-actual_total,
                })
            prophet_out = predict_prophet_daily_direct(p_daily, target_month)
            if prophet_out is not None:
                daily_rows.append(prophet_out.assign(platform=platform, model_name='prophet_daily'))
                pred_total = float(prophet_out['pred_daily_sales'].sum())
                actual_total = float(prophet_out['actual_daily_sales'].sum())
                monthly_rows.append({
                    'target_month': target_month, 'cutoff_day': 0, 'platform': platform, 'model_name': 'prophet_daily',
                    'pred_total_sales': pred_total, 'actual_total_sales': actual_total,
                    'abs_error_total': abs(pred_total-actual_total), 'ape_total': None if actual_total == 0 else abs(pred_total-actual_total)/actual_total,
                    'bias_total': pred_total-actual_total,
                })

    monthly_df = pd.DataFrame(monthly_rows).sort_values(['platform','model_name','target_month']) if monthly_rows else pd.DataFrame()
    summary_df = (
        monthly_df.groupby(['cutoff_day','platform','model_name'], as_index=False)
        .agg(months=('target_month','nunique'), mae_total=('abs_error_total','mean'), mape_total=('ape_total','mean'), bias_total=('bias_total','mean'))
        .sort_values(['cutoff_day','platform','mape_total','mae_total'])
    ) if not monthly_df.empty else pd.DataFrame()

    if not monthly_df.empty:
        monthly_df.to_csv(output_dir / 'cutoff0_daily_direct_backtest.csv', index=False)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / 'cutoff0_daily_direct_summary.csv', index=False)
    if daily_rows:
        pd.concat(daily_rows, ignore_index=True).to_csv(output_dir / 'cutoff0_daily_direct_daily_predictions.csv', index=False)

    return monthly_df, summary_df


if __name__ == '__main__':
    monthly_df, summary_df = run_cutoff0_daily_direct_backtest()
    print('=== CUTOFF0 DAILY DIRECT SUMMARY ===')
    print(summary_df.to_string(index=False))
