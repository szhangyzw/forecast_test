from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet


CNY1 = pd.DataFrame({
    'holiday': 'cny',
    'ds': list(pd.date_range(start='2024-02-10', periods=8))
          + list(pd.date_range(start='2025-01-28', periods=8))
          + list(pd.date_range(start='2026-02-15', periods=9)),
    'lower_window': -6,
    'upper_window': 3,
})

CNY = pd.DataFrame({
    'holiday': 'cny',
    'ds': ['2024-02-10', '2025-01-28', '2026-02-17'],
    'lower_window': -6,
    'upper_window': 8,
})

NEW_YEAR = pd.DataFrame({
    'holiday': 'ny',
    'ds': ['2024-01-01', '2025-01-01', '2026-01-01', '2027-01-01'],
    'lower_window': 0,
    'upper_window': 2,
})

WOMENSDAY = pd.DataFrame({
    'holiday': 'wd',
    'ds': ['2024-03-08', '2025-03-08', '2026-03-08'],
    'lower_window': -3,
    'upper_window': 1,
})

QINGREN = pd.DataFrame({
    'holiday': 'qingren',
    'ds': [
        '2024-02-14', '2025-02-14', '2026-02-14', '2027-02-14',
        '2024-08-10', '2025-08-29', '2026-08-19',
        '2024-05-21', '2025-05-21', '2026-05-21', '2027-05-21'
    ],
    'lower_window': -3,
    'upper_window': 1,
})

EC_VIP = pd.DataFrame({
    'holiday': 'ec_vip',
    'ds': ['2024-08-08', '2025-08-08', '2026-08-08', '2027-08-08'],
    'lower_window': -2,
    'upper_window': 2,
})

EC_618 = pd.DataFrame({
    'holiday': 'ec_618',
    'ds': ['2024-06-18', '2025-06-18', '2026-06-18', '2027-06-18'],
    'lower_window': -20,
    'upper_window': 1,
})

EC_D11 = pd.DataFrame({
    'holiday': 'ec_d11',
    'ds': ['2024-11-11', '2025-11-11', '2026-11-11', '2027-11-11'],
    'lower_window': -20,
    'upper_window': 1,
})

EC_D12 = pd.DataFrame({
    'holiday': 'ec_d12',
    'ds': ['2024-12-12', '2025-12-12', '2026-12-12'],
    'lower_window': -3,
    'upper_window': 1,
})


def build_prophet_holidays() -> pd.DataFrame:
    holidays = pd.concat([CNY1, CNY, NEW_YEAR, WOMENSDAY, QINGREN, EC_VIP, EC_618, EC_D11, EC_D12], ignore_index=True)
    holidays['ds'] = pd.to_datetime(holidays['ds'].map(lambda x: str(x)[:10]))
    holidays = holidays.drop_duplicates().sort_values(['ds', 'holiday']).reset_index(drop=True)
    return holidays


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'}).copy()
    out['ds'] = pd.to_datetime(out['ds'])
    return out


def predict_prophet_full_month(
    daily_df: pd.DataFrame,
    target_month: str,
    growth: str = 'logistic',
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 5.0,
    holidays_prior_scale: float = 10.0,
) -> float:
    target_start = pd.Timestamp(f'{target_month}-01')
    target_end = target_start + pd.offsets.MonthEnd(0)

    train = prepare_prophet_df(daily_df)
    train = train[train['ds'] < target_start].copy()
    if len(train) < 60:
        return np.nan

    future = pd.DataFrame({'ds': pd.date_range(target_start, target_end, freq='D')})

    if growth == 'logistic':
        y_max = float(train['y'].max()) if len(train) else 0.0
        y_q95 = float(train['y'].quantile(0.95)) if len(train) else 0.0
        cap_value = max(y_max, y_q95) * 1.8 if max(y_max, y_q95) > 0 else 1.0
        floor_value = 0.0
        train['cap'] = cap_value
        train['floor'] = floor_value
        future['cap'] = cap_value
        future['floor'] = floor_value

    model = Prophet(
        growth=growth,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays=build_prophet_holidays(),
        holidays_prior_scale=holidays_prior_scale,
    )
    model.fit(train)
    fcst = model.predict(future)
    return float(fcst['yhat'].clip(lower=0).sum())
