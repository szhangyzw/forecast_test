from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet

from src.models.xgb_model import build_xgb_model
from src.models.prophet_model import build_prophet_holidays
from src.feature_engineering.build_event_calendar_feature import add_event_features_to_month_df


def _month_feature_table(platform_daily: pd.DataFrame) -> pd.DataFrame:
    df = platform_daily.copy().sort_values('date')
    monthly = df.groupby('year_month', as_index=False)['sales'].sum().rename(columns={'sales': 'month_total'})
    monthly['period'] = pd.PeriodIndex(monthly['year_month'], freq='M')
    monthly['last_month_total'] = monthly['month_total'].shift(1)
    monthly['last_3month_avg_total'] = monthly['month_total'].shift(1).rolling(3, min_periods=1).mean()
    monthly['last_6month_avg_total'] = monthly['month_total'].shift(1).rolling(6, min_periods=1).mean()

    ly = monthly[['period', 'month_total']].copy()
    ly['period_plus1y'] = ly['period'] + 12
    ly = ly.rename(columns={'month_total': 'ly_same_month_total'})[['period_plus1y', 'ly_same_month_total']]
    monthly = monthly.merge(ly, left_on='period', right_on='period_plus1y', how='left').drop(columns=['period_plus1y'])
    monthly['ly_same_month_avg_daily'] = monthly['ly_same_month_total'] / monthly['period'].dt.days_in_month

    event_df = monthly[['year_month']].copy()
    event_df['year'] = event_df['year_month'].str[:4].astype(int)
    event_df['month'] = event_df['year_month'].str[5:7].astype(int)
    event_df['days_in_month'] = pd.PeriodIndex(event_df['year_month'], freq='M').days_in_month
    event_df = add_event_features_to_month_df(event_df)
    monthly = monthly.merge(event_df.drop(columns=['year', 'month', 'days_in_month']), on='year_month', how='left')
    return monthly


def _build_base_feat(platform_daily: pd.DataFrame) -> pd.DataFrame:
    p_daily = platform_daily.copy().sort_values('date')
    monthly_feat = _month_feature_table(p_daily)

    recent_daily = []
    for ym in monthly_feat['year_month']:
        ms = pd.Timestamp(f'{ym}-01')
        hist = p_daily[p_daily['date'] < ms]
        recent_daily.append(hist.tail(90)['sales'].mean() if len(hist) else np.nan)
    monthly_feat['recent_90d_avg_daily'] = recent_daily

    ly_daily = p_daily[['date', 'sales']].copy()
    ly_daily['date_plus1y'] = ly_daily['date'] + pd.DateOffset(years=1)
    ly_daily = ly_daily.rename(columns={'sales': 'ly_same_day_sales'})[['date_plus1y', 'ly_same_day_sales']]

    feat = p_daily.copy()
    feat['day_of_month'] = feat['date'].dt.day
    feat['day_of_week'] = feat['date'].dt.dayofweek
    feat['is_weekend2'] = feat['day_of_week'].isin([5, 6]).astype(int)
    feat['is_month_start'] = (feat['day_of_month'] <= 3).astype(int)
    feat['is_month_end'] = (feat['days_in_month'] - feat['day_of_month'] < 3).astype(int)
    feat['day_progress'] = feat['day_of_month'] / feat['days_in_month']

    feat = feat.merge(monthly_feat, on='year_month', how='left', suffixes=('', '_m'))
    feat = feat.merge(ly_daily, left_on='date', right_on='date_plus1y', how='left').drop(columns=['date_plus1y'])
    if 'ly_same_month_total_y' in feat.columns:
        feat['ly_same_month_total'] = feat['ly_same_month_total_y']
    elif 'ly_same_month_total_x' in feat.columns:
        feat['ly_same_month_total'] = feat['ly_same_month_total_x']
    return feat


FEATURE_COLS_C0 = [
    'month', 'day_of_month', 'day_of_week', 'is_weekend2', 'is_month_start', 'is_month_end', 'day_progress', 'days_in_month',
    'last_month_total', 'last_3month_avg_total', 'last_6month_avg_total', 'ly_same_month_total', 'ly_same_month_avg_daily', 'recent_90d_avg_daily', 'ly_same_day_sales',
    'is_cny_impact_month', 'pre_cny_flag', 'in_cny_flag', 'post_cny_flag', 'is_618_impact_month', 'pre_618_flag', 'in_618_flag', 'is_double11_impact_month', 'pre_double11_flag', 'in_double11_flag'
]

FEATURE_COLS_CN = FEATURE_COLS_C0 + ['mtd_sales_anchor', 'mtd_avg_anchor', 'hist_share_p50_anchor']


def predict_tree_daily_direct(
    platform_daily: pd.DataFrame,
    target_month: str,
    cutoff_day: int,
    model_type: str,
    hist_share_p50: float | None = None,
) -> float | None:
    feat = _build_base_feat(platform_daily)
    target_start = pd.Timestamp(f'{target_month}-01')
    target_end = target_start + pd.offsets.MonthEnd(0)

    if cutoff_day <= 0:
        train = feat[feat['year_month'] < target_month].copy()
        test = feat[feat['year_month'] == target_month].copy().sort_values('date')
        train = train.dropna(subset=FEATURE_COLS_C0 + ['sales'])
        test = test.dropna(subset=FEATURE_COLS_C0)
        if len(train) < 90 or test.empty:
            return None
        if model_type == 'gbdt':
            model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
            model.fit(train[FEATURE_COLS_C0].values, train['sales'].values)
            pred = np.clip(model.predict(test[FEATURE_COLS_C0].values), 0, None)
        else:
            model = build_xgb_model(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(train[FEATURE_COLS_C0], train['sales'])
            pred = np.clip(model.predict(test[FEATURE_COLS_C0]), 0, None)
        return float(pred.sum())

    observed_end = target_start + pd.Timedelta(days=int(cutoff_day) - 1)
    pred_start = observed_end + pd.Timedelta(days=1)
    target_days = pd.date_range(pred_start, target_end, freq='D')
    if len(target_days) == 0:
        return None

    observed = platform_daily[(platform_daily['date'] >= target_start) & (platform_daily['date'] <= observed_end)].copy()
    if observed.empty:
        return None
    mtd_sales = float(observed['sales'].sum())
    mtd_avg_sales = float(observed['sales'].mean())

    train_rows = []
    hist_feat = feat[feat['date'] < target_start].copy()
    for ym in sorted(hist_feat['year_month'].unique()):
        ms = pd.Timestamp(f'{ym}-01')
        oe = ms + pd.Timedelta(days=int(cutoff_day) - 1)
        rem = feat[(feat['date'] > oe) & (feat['year_month'] == ym)].copy()
        obs = platform_daily[(platform_daily['date'] >= ms) & (platform_daily['date'] <= oe)].copy()
        if rem.empty or obs.empty:
            continue
        obs_mtd = float(obs['sales'].sum())
        obs_avg = float(obs['sales'].mean())
        rem['mtd_sales_anchor'] = obs_mtd
        rem['mtd_avg_anchor'] = obs_avg
        rem['hist_share_p50_anchor'] = np.nan
        train_rows.append(rem)

    if not train_rows:
        return None
    train = pd.concat(train_rows, ignore_index=True)
    test = feat[feat['date'].isin(target_days)].copy()
    test['mtd_sales_anchor'] = mtd_sales
    test['mtd_avg_anchor'] = mtd_avg_sales
    test['hist_share_p50_anchor'] = hist_share_p50

    train = train.dropna(subset=FEATURE_COLS_CN + ['sales'])
    test = test.dropna(subset=FEATURE_COLS_CN)
    if len(train) < 90 or test.empty:
        return None

    if model_type == 'gbdt':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
        model.fit(train[FEATURE_COLS_CN].values, train['sales'].values)
        pred = np.clip(model.predict(test[FEATURE_COLS_CN].values), 0, None)
    else:
        model = build_xgb_model(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(train[FEATURE_COLS_CN], train['sales'])
        pred = np.clip(model.predict(test[FEATURE_COLS_CN]), 0, None)
    return float(pred.sum() + mtd_sales)


def predict_prophet_daily_direct(platform_daily: pd.DataFrame, target_month: str, cutoff_day: int) -> float | None:
    target_start = pd.Timestamp(f'{target_month}-01')
    target_end = target_start + pd.offsets.MonthEnd(0)
    observed_end = target_start - pd.Timedelta(days=1) if cutoff_day <= 0 else target_start + pd.Timedelta(days=int(cutoff_day) - 1)
    pred_start = target_start if cutoff_day <= 0 else observed_end + pd.Timedelta(days=1)

    train = platform_daily[platform_daily['date'] <= observed_end][['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'}).copy()
    if len(train) < 60 or pred_start > target_end:
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

    future = pd.DataFrame({'ds': pd.date_range(pred_start, target_end, freq='D')})
    future['cap'] = cap_value
    future['floor'] = 0.0
    fcst = model.predict(future)
    observed_actual = 0.0
    if cutoff_day > 0:
        observed_actual = float(platform_daily[(platform_daily['date'] >= target_start) & (platform_daily['date'] <= observed_end)]['sales'].sum())
    return float(fcst['yhat'].clip(lower=0).sum() + observed_actual)
