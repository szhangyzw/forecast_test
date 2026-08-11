# E-commerce Sales Forecasting

A Python-based rolling sales forecasting project designed to estimate full-month e-commerce sales at the beginning, middle, and end of a month.

The project explores a practical statistical decision problem: **how should a forecasting method be selected when the amount of information available changes throughout the month?** It supports forecasting and historical backtesting at any `cutoff_day` from 0 to 31, allowing different models to be compared under the same information constraints.

## Project Motivation

E-commerce teams often need a full-month sales estimate before the month has ended. At the beginning of a month, little or no current-month sales information is available, so forecasts depend mainly on historical patterns, seasonality, and event effects. As the month progresses, actual month-to-date sales provide increasingly useful evidence.

Instead of assuming that one model is always optimal, this project compares statistical, time-series, and machine-learning approaches at different forecasting stages. The results can provide an analytical reference for monthly target tracking, promotional monitoring, and inventory planning.

## Main Objectives

- Compare multiple forecasting methods within a unified framework.
- Evaluate model performance by month, platform, brand, and cutoff point.
- Simulate realistic forecasting conditions using only information available at each cutoff.
- Recommend models according to platform, month characteristics, and forecasting stage.
- Support both full-month forecasting and remaining-period forecasting through one interface.
- Provide a foundation for future prediction intervals and dashboard development.

## Forecasting Approaches

The framework currently includes the following model families:

1. Historical average
2. Same month in the previous year
3. Year-on-year growth extrapolation
4. Seasonality- and event-adjusted forecasting
5. Month-to-date progress extrapolation
6. Prophet-based forecasting
7. XGBoost-based forecasting
8. GBDT-based forecasting
9. Residual learning
10. Similar-month retrieval
11. Three-model weighted ensemble

Some model families contain different implementations for full-month and remaining-period forecasting.

## Backtesting Design

The most recent evaluation covers **12 consecutive historical months, from March 2025 to February 2026**, at five forecasting cutoffs: Days 0, 5, 10, 15, and 25.

For each historical test month, the framework uses only information that would have been available at the selected cutoff. This design simulates a real forecasting situation and reduces future-information leakage.

The current stage-based model selection is:

- **XGBoost** at Days 0 and 5, when little or no current-month information is available.
- **MTD progress percentage** at Days 10, 15, and 25, when actual month-to-date sales provide a stronger basis for projection.

The signed percentage error is defined as:

```text
diff% = Forecast / Actual - 1
```

A positive value indicates over-forecasting, while a negative value indicates under-forecasting. For example, `9%` means that the forecast was 9% above the actual full-month result.

## Detailed Backtesting Results

| Test month | Day 0: XGBoost | Day 5: XGBoost | Day 10: MTD progress | Day 15: MTD progress | Day 25: MTD progress |
|---|---:|---:|---:|---:|---:|
| 2025-03 | 9% | 1% | 5% | 3% | 0% |
| 2025-04 | -8% | -5% | -1% | -2% | 0% |
| 2025-05 | -9% | -7% | -4% | 4% | -3% |
| 2025-06 | 5% | -11% | 6% | 5% | 0% |
| 2025-07 | 10% | 0% | 5% | 6% | 4% |
| 2025-08 | 9% | -7% | -2% | -1% | 1% |
| 2025-09 | -3% | 7% | 3% | 5% | 1% |
| 2025-10 | -8% | -9% | 3% | 2% | -1% |
| 2025-11 | 7% | -10% | -3% | 1% | 1% |
| 2025-12 | 11% | -6% | 2% | 4% | 4% |
| 2026-01 | -11% | 0% | 0% | 0% | -1% |
| 2026-02 | -7% | -10% | 6% | 0% | 0% |

## Summary of Backtesting Performance

Because `diff%` is a signed error rather than a prediction interval, model accuracy is summarised below using **mean absolute percentage error**, calculated as the average of `|diff%|` across the 12 test months.

| Cutoff | Selected method | Mean signed error | Mean absolute error | Median absolute error | Maximum absolute error | Months within ±5% |
|---|---|---:|---:|---:|---:|---:|
| Day 0 | XGBoost | +0.4% | 8.1% | 8.5% | 11% | 2 of 12 |
| Day 5 | XGBoost | -4.8% | 6.1% | 7.0% | 11% | 4 of 12 |
| Day 10 | MTD progress | +1.7% | 3.3% | 3.0% | 6% | 10 of 12 |
| Day 15 | MTD progress | +2.3% | 2.8% | 2.5% | 6% | 11 of 12 |
| Day 25 | MTD progress | +0.5% | 1.3% | 1.0% | 4% | 12 of 12 |

## Interpretation of the Results

- **Accuracy improved as more current-month information became available.** Mean absolute error decreased from 8.1% on Day 0 to 1.3% on Day 25.
- **Day 0 XGBoost was broadly unbiased across the full test period.** Its mean signed error was +0.4%, although individual monthly errors ranged from -11% to +11%.
- **Day 5 XGBoost showed a tendency to under-forecast.** Its mean signed error was -4.8%, suggesting that this stage requires further feature or calibration work.
- **MTD progress became much more reliable from Day 10 onward.** Ten of 12 Day 10 forecasts, 11 of 12 Day 15 forecasts, and all 12 Day 25 forecasts were within ±5% of actual sales.
- **Late-month forecasts were the most stable.** On Day 25, the maximum absolute error was 4%, and 9 of 12 months were within ±2%.
- **The results support stage-specific model selection.** A history- and feature-based model is required when current-month data are scarce, while a progress-based method becomes more effective after a meaningful portion of the month has been observed.

These results describe the current 12-month backtest and should not be interpreted as universal accuracy guarantees. Performance may change with a longer evaluation period, different platforms or brands, or months affected by unusual holidays and promotional events.

## Cutoff-Day Definition

| `cutoff_day` | Meaning |
|---|---|
| `0` | Use data available through the end of the previous month to forecast the entire target month. |
| `1–31` | Use data available through Day N of the target month to forecast the remaining days and estimate the full-month total. |

### Rules

- `cutoff_day < 0`: invalid input
- `cutoff_day = 0`: full-month forecasting scenario
- `cutoff_day > number of days in the month`: automatically adjusted to the final calendar day
- `cutoff_day = final day of the month`: remaining forecast period equals zero

## Models Available at Different Cutoffs

### Cutoff = 0: Full-Month Forecast

- `history_average`
- `last_year_same_month`
- `yoy_growth_extrapolation`
- `prophet_fullmonth`
- `xgboost_v2_event`
- `gbdt_daily_direct_c0`
- `xgboost_daily_direct_c0`
- `prophet_daily`

### Cutoff > 0: In-Month and Remaining-Sales Forecast

- `history_average`
- `mtd_progress_p50`
- `prophet_mixed`
- `xgboost_residual`
- `gbdt_daily_direct_c{cutoff}`
- `xgboost_daily_direct_c{cutoff}`
- `prophet_daily`

## Repository Structure

```text
sales_forecast/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── feature_store/
│   └── backtest_result/
├── docs/
├── notebooks/
├── src/
│   ├── data_prep/
│   ├── feature_engineering/
│   ├── models/
│   ├── backtest/
│   ├── inference/
│   ├── report/
│   └── utils/
├── tests/
├── main.py
├── requirements.txt
└── .gitignore
```

## Minimum Input Data

The raw dataset should contain at least:

- `date`
- `platform`
- `sales`

The current test pipeline can automatically map the following aliases:

- `trans_dt` → `date`
- `amount` → `sales`

Depending on the forecasting scope, a `brand` field may also be used.

## Local Setup

1. Download or clone the `sales_forecast/` directory.
2. Preserve the internal directory structure.
3. Place the raw dataset at:

```text
sales_forecast/data/raw/test_forecast_data.csv
```

4. Install the dependencies listed in `requirements.txt`.
5. Run the required command from the project root directory.

## Usage

### 1. Check Project Status

```bash
python main.py --action status
```

### 2. Run a Single Forecast

```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 15
```

### 3. Run a Forecast by Brand, Platform, and Cutoff

```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand brand1 --platform jd
```

Replace `jd` with the platform label used in the input dataset.

### 4. Forecast All Brands for One Platform

```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand total --platform jd
```

### 5. Forecast One Brand Across All Platforms

```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand brand1 --platform total
```

### 6. Backtest a Single Cutoff

```bash
python main.py --action backtest --cutoff-day 7
```

### 7. Backtest by Brand, Platform, and Multiple Cutoffs

```bash
python main.py --action backtest --cutoff-days 0,19 --brand brand1 --platform jd
```

### 8. Backtest Multiple Cutoffs

```bash
python main.py --action backtest --cutoff-days 0,7,14,21,28
```

### 9. Run the Standard Fixed-Point Backtest

```bash
python main.py --action backtest --cutoff-days 0,5,10,15,20,25
```

## Daily-Direct Forecasting

### Cutoff = 0

The model uses information available through the end of the previous month to predict each day of the next month directly. Daily predictions are then aggregated into a full-month estimate. The method does not use recursive day-by-day forecasting.

### Cutoff > 0

The model uses information available through Day N of the current month to predict each remaining day directly. The predicted remaining sales are added to the actual month-to-date sales to produce the estimated full-month total. This method also avoids recursive forecasting.

## Current Model Recommendations

Based on the current 12-month evaluation:

- Use **XGBoost** as the current candidate at Days 0 and 5.
- Use **MTD progress percentage** as the current candidate at Days 10, 15, and 25.
- Continue comparing alternative models by platform and brand, because the overall recommendation may not be optimal for every subgroup.
- Treat the Day 5 under-forecasting bias as an area for further model improvement and validation.

The recommendations are derived from the current test period and should be recalculated when new monthly data become available.

## Output Files

Typical outputs are saved in:

```text
data/backtest_result/backtest_v1.csv
data/backtest_result/backtest_v1_summary.csv
data/backtest_result/final_model_recommendation_by_platform_cutoff.xlsx
data/backtest_result/final_model_recommendation_by_platform_cutoff.csv
```

## My Contribution and Learning

I designed the comparison framework, developed the Python forecasting workflow, created the cutoff-based backtesting logic, compared model errors, and interpreted the results. AI-assisted tools were used to support research and coding, while the project methodology, model comparison, backtesting design, and interpretation were developed and validated by me.

The project helped me understand that:

- a forecasting model should be evaluated under the same information constraints as the real decision problem;
- a more complex algorithm is not automatically more accurate;
- model suitability depends on timing, available data, seasonality, and unusual events;
- forecast errors are useful because they reveal model limitations and guide further investigation.

This project strengthened my interest in statistical modelling, time-series analysis, machine learning, and data-driven decision-making. It provided an early opportunity to explore how quantitative evidence can be used to compare alternatives under uncertainty.

## Limitations

- Current findings are based on a limited historical test period.
- Exceptional events may not be fully represented in the training data.
- Approximate error ranges summarise current backtesting results and are not prediction intervals.
- Model rankings may change across platforms, brands, and future datasets.
- Business impact has not been evaluated through a controlled operational experiment.

## Future Development

- Add explicit holiday and promotional-event features.
- Introduce prediction intervals and uncertainty estimates.
- Automate model-selection rule generation.
- Export model-recommendation tables automatically.
- Extend testing to a longer historical period.
- Develop a Streamlit dashboard.
- Add current backtesting charts and summary snapshots to this README.
