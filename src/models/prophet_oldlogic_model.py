from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet


def build_prophet_holidays(start_year=2024, end_year=2027):
    cny1 = pd.DataFrame({
        'holiday': 'cny',
        'ds': list(pd.date_range(start='2024-02-10', periods=8))
              + list(pd.date_range(start='2025-01-28', periods=8))
              + list(pd.date_range(start='2026-02-15', periods=9)),
        'lower_window': -6,
        'upper_window': 3,
    })

    cny = pd.DataFrame({
        'holiday': 'cny',
        'ds': ['2024-02-10', '2025-01-28', '2026-02-17'],
        'lower_window': -6,
        'upper_window': 8,
    })

    new_year = pd.DataFrame({
        'holiday': 'ny',
        'ds': ['2024-01-01', '2025-01-01', '2026-01-01', '2027-01-01'],
        'lower_window': 0,
        'upper_window': 2,
    })

    womensday = pd.DataFrame({
        'holiday': 'wd',
        'ds': ['2024-03-08', '2025-03-08', '2026-03-08'],
        'lower_window': -3,
        'upper_window': 1,
    })

    qingren = pd.DataFrame({
        'holiday': 'qingren',
        'ds': [
            '2024-02-14', '2025-02-14', '2026-02-14', '2027-02-14',
            '2024-08-10', '2025-08-29', '2026-08-19',
            '2024-05-21', '2025-05-21', '2026-05-21', '2027-05-21'
        ],
        'lower_window': -3,
        'upper_window': 1,
    })

    ec_vip = pd.DataFrame({
        'holiday': 'ec_vip',
        'ds': ['2024-08-08', '2025-08-08', '2026-08-08', '2027-08-08'],
        'lower_window': -2,
        'upper_window': 2,
    })

    ec_618 = pd.DataFrame({
        'holiday': 'ec_618',
        'ds': ['2024-06-18', '2025-06-18', '2026-06-18', '2027-06-18'],
        'lower_window': -20,
        'upper_window': 1,
    })

    ec_d11 = pd.DataFrame({
        'holiday': 'ec_d11',
        'ds': ['2024-11-11', '2025-11-11', '2026-11-11', '2027-11-11'],
        'lower_window': -20,
        'upper_window': 1,
    })

    ec_d12 = pd.DataFrame({
        'holiday': 'ec_d12',
        'ds': ['2024-12-12', '2025-12-12', '2026-12-12'],
        'lower_window': -3,
        'upper_window': 1,
    })

    holidays = pd.concat([cny1, cny, new_year, womensday, qingren, ec_vip, ec_618, ec_d11, ec_d12], ignore_index=True)
    holidays['ds'] = pd.to_datetime(holidays['ds'].map(lambda x: str(x)[:10]))
    holidays = holidays.drop_duplicates().sort_values(['ds', 'holiday']).reset_index(drop=True)
    return holidays


def predict_prophet_oldlogic(
    train_daily: pd.DataFrame,
    monthly_pivot: pd.DataFrame,
    col: str,
    observed_actual: float,
    target_month: str,
    observed_end: pd.Timestamp,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
) -> float:
    train = train_daily[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'}).copy()
    train = train[train['ds'] <= observed_end]
    if len(train) < 60 or pred_start > pred_end:
        return np.nan

    holiday_start_year = max(2024, train['ds'].dt.year.min())
    holiday_end_year = max(pd.Timestamp(target_month + '-01').year, train['ds'].dt.year.max())
    holidays = build_prophet_holidays(holiday_start_year, holiday_end_year)

    y_max = float(train['y'].max()) if len(train) else 0.0
    y_q95 = float(train['y'].quantile(0.95)) if len(train) else 0.0
    cap_value = max(y_max, y_q95) * 1.8 if max(y_max, y_q95) > 0 else 1.0
    floor_value = 0.0
    train['cap'] = cap_value
    train['floor'] = floor_value

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

    future = pd.DataFrame({'ds': pd.date_range(pred_start, pred_end, freq='D')})
    future['cap'] = cap_value
    future['floor'] = floor_value
    fcst = model.predict(future)
    raw_pred = float(fcst['yhat'].clip(lower=0).sum())

    target_ts = pd.Period(target_month, freq='M')
    ym_last = f'{target_ts.year - 1}-{target_ts.month:02d}'
    ym_prev = f'{target_ts.year - 2}-{target_ts.month:02d}'
    last_year = float(monthly_pivot.loc[ym_last, col]) if ym_last in monthly_pivot.index else np.nan
    prev_year = float(monthly_pivot.loc[ym_prev, col]) if ym_prev in monthly_pivot.index else np.nan
    hist_candidates = [x for x in [last_year, prev_year] if not np.isnan(x)]

    month_days = pred_end.days_in_month
    observed_days = max((observed_end - pd.Timestamp(f'{target_month}-01')).days + 1, 0)
    remaining_days = len(pd.date_range(pred_start, pred_end, freq='D'))

    hist_center = np.nan
    if hist_candidates:
        hist_month_center = float(np.mean(hist_candidates))
        hist_center = max(hist_month_center * remaining_days / month_days, 0.0)

    run_rate = float(observed_actual / observed_days * remaining_days) if observed_days > 0 else np.nan
    anchors = [x for x in [hist_center, run_rate] if not np.isnan(x)]
    anchor_center = float(np.mean(anchors)) if anchors else raw_pred

    progress = min(max(observed_days / month_days, 0), 1)
    prophet_weight = 0.35 + 0.25 * progress
    blended = prophet_weight * raw_pred + (1 - prophet_weight) * anchor_center

    upper_candidates = [
        x for x in [
            hist_center * 1.35 if not np.isnan(hist_center) else np.nan,
            run_rate * 1.10 if not np.isnan(run_rate) else np.nan,
            raw_pred,
        ] if not np.isnan(x)
    ]
    lower_candidates = [
        x for x in [
            0.0,
            hist_center * 0.85 if not np.isnan(hist_center) else np.nan,
            run_rate * 0.80 if not np.isnan(run_rate) else np.nan,
        ] if not np.isnan(x)
    ]
    upper = min(upper_candidates) if upper_candidates else raw_pred
    lower = max(lower_candidates) if lower_candidates else 0.0
    return float(min(max(blended, lower), upper))
