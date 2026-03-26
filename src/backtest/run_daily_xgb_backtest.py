from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.inference.run_forecast import prepare_daily_feature_table
from src.models.xgb_model import build_xgb_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / 'data' / 'raw' / 'test_forecast_data.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'backtest_result'

FEATURES_DAILY = [
    'dow', 'is_weekend', 'day', 'month', 'days_in_month', 'day_progress',
    'lag1', 'lag7', 'lag14', 'lag28',
    'roll7', 'roll14', 'roll28',
    'ly_same_day', 'ly_same_month_avg',
    'is_cny_impact_month', 'pre_cny_flag', 'in_cny_flag', 'post_cny_flag',
    'is_618_impact_month', 'pre_618_flag', 'in_618_flag',
    'is_double11_impact_month', 'pre_double11_flag', 'in_double11_flag',
]

CNY_DATES = {
    2024: pd.Timestamp('2024-02-10'),
    2025: pd.Timestamp('2025-01-29'),
    2026: pd.Timestamp('2026-02-17'),
}


def _get_complete_months(daily: pd.DataFrame) -> list[str]:
    month_end = (
        daily.groupby(['platform', 'year_month'], as_index=False)
        .agg(max_day=('day_seq_in_month', 'max'), days_in_month=('days_in_month', 'max'))
    )
    complete = month_end[month_end['max_day'] >= month_end['days_in_month']].copy()
    complete_total = complete[complete['platform'] == 'total']['year_month'].tolist()
    return sorted(complete_total)


def _add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out['month_start'] = out['date'].values.astype('datetime64[M]')

    out['is_618_impact_month'] = out['month'].isin([5, 6]).astype(int)
    out['pre_618_flag'] = (out['month'] == 5).astype(int)
    out['in_618_flag'] = (out['month'] == 6).astype(int)

    out['is_double11_impact_month'] = out['month'].isin([10, 11]).astype(int)
    out['pre_double11_flag'] = (out['month'] == 10).astype(int)
    out['in_double11_flag'] = (out['month'] == 11).astype(int)

    out['is_cny_impact_month'] = 0
    out['pre_cny_flag'] = 0
    out['in_cny_flag'] = 0
    out['post_cny_flag'] = 0

    for idx, row in out.iterrows():
        y = int(row['year'])
        if y not in CNY_DATES:
            continue
        cny_date = CNY_DATES[y]
        month_period = pd.Period(row['date'], freq='M')
        cny_period = pd.Period(cny_date, freq='M')
        if month_period == cny_period - 1:
            out.at[idx, 'is_cny_impact_month'] = 1
            out.at[idx, 'pre_cny_flag'] = 1
        elif month_period == cny_period:
            out.at[idx, 'is_cny_impact_month'] = 1
            out.at[idx, 'in_cny_flag'] = 1
        elif month_period == cny_period + 1:
            out.at[idx, 'is_cny_impact_month'] = 1
            out.at[idx, 'post_cny_flag'] = 1
    return out.drop(columns=['month_start'])


def _build_daily_supervised(platform_daily: pd.DataFrame) -> pd.DataFrame:
    df = platform_daily.copy().sort_values('date').reset_index(drop=True)
    df['day_progress'] = df['day'] / df['days_in_month']

    for lag in [1, 7, 14, 28]:
        df[f'lag{lag}'] = df['sales'].shift(lag)

    for w in [7, 14, 28]:
        df[f'roll{w}'] = df['sales'].shift(1).rolling(w, min_periods=1).mean()

    ly_map = df[['date', 'sales']].copy()
    ly_map['date_plus1y'] = ly_map['date'] + pd.DateOffset(years=1)
    ly_map = ly_map.rename(columns={'sales': 'ly_same_day'})[['date_plus1y', 'ly_same_day']]
    df = df.merge(ly_map, left_on='date', right_on='date_plus1y', how='left').drop(columns=['date_plus1y'])

    month_avg = (
        df.groupby('year_month', as_index=False)['sales'].mean()
        .rename(columns={'sales': 'month_avg_sales'})
    )
    month_avg['period'] = pd.PeriodIndex(month_avg['year_month'], freq='M')
    ly_month = month_avg[['period', 'month_avg_sales']].copy()
    ly_month['period_plus1y'] = ly_month['period'] + 12
    ly_month = ly_month.rename(columns={'month_avg_sales': 'ly_same_month_avg'})[['period_plus1y', 'ly_same_month_avg']]
    df['period'] = pd.PeriodIndex(df['year_month'], freq='M')
    df = df.merge(ly_month, left_on='period', right_on='period_plus1y', how='left').drop(columns=['period_plus1y'])

    df = _add_event_flags(df)
    return df


def _make_feature_row(history_df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    hist = history_df.sort_values('date').reset_index(drop=True)
    year = target_date.year
    month = target_date.month
    day = target_date.day
    days_in_month = target_date.days_in_month
    dow = target_date.dayofweek

    row = {
        'date': target_date,
        'year': year,
        'month': month,
        'day': day,
        'days_in_month': days_in_month,
        'dow': dow,
        'is_weekend': int(dow in [5, 6]),
        'day_progress': day / days_in_month,
    }

    sales_series = hist['sales'].tolist()
    for lag in [1, 7, 14, 28]:
        row[f'lag{lag}'] = sales_series[-lag] if len(sales_series) >= lag else np.nan
    for w in [7, 14, 28]:
        row[f'roll{w}'] = float(np.mean(sales_series[-w:])) if len(sales_series) >= 1 else np.nan

    ly_day = hist.loc[hist['date'] == target_date - pd.DateOffset(years=1), 'sales']
    row['ly_same_day'] = float(ly_day.iloc[0]) if not ly_day.empty else np.nan

    ly_month = (target_date.to_period('M') - 12).strftime('%Y-%m')
    ly_month_rows = hist[hist['year_month'] == ly_month]
    row['ly_same_month_avg'] = float(ly_month_rows['sales'].mean()) if not ly_month_rows.empty else np.nan

    tmp = pd.DataFrame([row])
    tmp = _add_event_flags(tmp)
    row.update(tmp.iloc[0][[
        'is_cny_impact_month', 'pre_cny_flag', 'in_cny_flag', 'post_cny_flag',
        'is_618_impact_month', 'pre_618_flag', 'in_618_flag',
        'is_double11_impact_month', 'pre_double11_flag', 'in_double11_flag'
    ]].to_dict())
    return row


def _train_and_predict_month(platform_daily: pd.DataFrame, target_month: str):
    full = _build_daily_supervised(platform_daily)
    train = full[full['year_month'] < target_month].copy()
    actual = full[full['year_month'] == target_month].copy().sort_values('date')

    usable_cols = [c for c in FEATURES_DAILY if c in train.columns]
    train = train.dropna(subset=usable_cols + ['sales']).copy()
    if len(train) < 90 or actual.empty:
        return None

    model = build_xgb_model(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train[usable_cols], train['sales'])

    history_for_roll = full[full['date'] < actual['date'].min()][['date', 'year_month', 'sales', 'year', 'month', 'day', 'days_in_month', 'dow', 'is_weekend']].copy()
    preds = []
    for d in actual['date'].tolist():
        feature_row = _make_feature_row(history_for_roll, d)
        feature_df = pd.DataFrame([feature_row])
        if feature_df[usable_cols].isna().any().any():
            pred = np.nan
        else:
            pred = float(model.predict(feature_df[usable_cols])[0])
            if pred < 0:
                pred = 0.0
        preds.append(pred)
        append_row = {
            'date': d,
            'year_month': d.strftime('%Y-%m'),
            'sales': pred,
            'year': d.year,
            'month': d.month,
            'day': d.day,
            'days_in_month': d.days_in_month,
            'dow': d.dayofweek,
            'is_weekend': int(d.dayofweek in [5, 6]),
        }
        history_for_roll = pd.concat([history_for_roll, pd.DataFrame([append_row])], ignore_index=True)

    result = actual[['date', 'year_month', 'sales']].copy().rename(columns={'sales': 'actual_daily_sales'})
    result['pred_daily_sales'] = preds
    return result


def run_daily_xgb_backtest(output_dir: str | Path = DEFAULT_OUTPUT, exclude_months=None):
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
            out = _train_and_predict_month(p_daily, target_month)
            if out is None:
                continue
            out['platform'] = platform
            daily_rows.append(out)

            pred_total = out['pred_daily_sales'].sum(skipna=True)
            actual_total = out['actual_daily_sales'].sum(skipna=True)
            monthly_rows.append({
                'target_month': target_month,
                'platform': platform,
                'model_name': 'xgboost_daily_recursive',
                'pred_total_sales': pred_total,
                'actual_total_sales': actual_total,
                'abs_error_total': abs(pred_total - actual_total),
                'ape_total': None if actual_total == 0 else abs(pred_total - actual_total) / actual_total,
                'bias_total': pred_total - actual_total,
            })

    monthly_df = pd.DataFrame(monthly_rows).sort_values(['platform', 'target_month'])
    daily_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()

    summary_df = (
        monthly_df.groupby(['platform', 'model_name'], as_index=False)
        .agg(
            months=('target_month', 'nunique'),
            mae_total=('abs_error_total', 'mean'),
            mape_total=('ape_total', 'mean'),
            bias_total=('bias_total', 'mean'),
        )
        .sort_values(['platform', 'mape_total', 'mae_total'])
    ) if not monthly_df.empty else pd.DataFrame()

    monthly_df.to_csv(output_dir / 'daily_xgb_monthly_backtest.csv', index=False)
    if not daily_df.empty:
        daily_df.to_csv(output_dir / 'daily_xgb_daily_predictions.csv', index=False)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / 'daily_xgb_summary.csv', index=False)

    return monthly_df, summary_df


if __name__ == '__main__':
    monthly_df, summary_df = run_daily_xgb_backtest()
    print('=== DAILY XGB MONTHLY SUMMARY ===')
    print(summary_df.to_string(index=False))
