# Data Dictionary

## 1. sales_raw_daily

| field | type | description |
|---|---|---|
| date | date | 日期 |
| platform | string | 平台：京东 / 阿里 |
| sales | float | 销量或销售额 |
| brand | string | 品牌，可选 |
| category | string | 类目，可选 |
| price | float | 价格，可选 |
| is_promo | int/bool | 是否活动日，可选 |
| promo_name | string | 活动名称，可选 |
| traffic | float | 流量，可选 |
| inventory | float | 库存，可选 |

## 2. sales_daily_agg

| field | type | description |
|---|---|---|
| date | date | 日期 |
| platform | string | 平台 |
| sales | float | 当日聚合销量 |
| year_month | string | YYYY-MM |
| dow | int | 周几 |
| is_weekend | int/bool | 是否周末 |
| is_holiday | int/bool | 是否节假日 |
| is_promo | int/bool | 是否活动 |
| days_in_month | int | 当月总天数 |
| day_seq_in_month | int | 当月第几天 |

## 3. forecast_feature_snapshot

粒度：`target_month × cutoff_day × platform`

| field | type | description |
|---|---|---|
| target_month | string | 目标月份 |
| cutoff_day | int | 截止日 |
| platform | string | 平台 |
| mtd_sales | float | 截止当日累计销量 |
| mtd_avg_sales | float | MTD日均 |
| days_elapsed | int | 已过去天数 |
| days_remaining | int | 剩余天数 |
| ly_same_month_total | float | 去年同月总销量 |
| ly_same_month_same_day | float | 去年同月同日累计 |
| last_3month_avg | float | 近3月均值 |
| hist_share_p50 | float | 历史同阶段累计占比中位数 |

## 4. backtest_prediction_result

| field | type | description |
|---|---|---|
| target_month | string | 目标月份 |
| cutoff_day | int | 截止日 |
| platform | string | 平台 |
| model_name | string | 模型名 |
| pred_value | float | 预测值 |
| actual_value | float | 实际值 |
| abs_error | float | 绝对误差 |
| ape | float | 绝对百分比误差 |
| bias | float | 偏差 |
