# Ecommerce Sales Forecast

一个面向电商平台（月初 / 月中 / 月末）滚动销量预估项目，当前已支持：
- **任意 `cutoff_day`（0~31）** 的预测与回测
- 多模型统一纳入同一套 backtest / summary
- 整月预测与月中剩余预测共用一套入口

---

## 项目目标

- 多模型并行预测：历史同期、同比外推、MTD 节奏外推、Prophet、XGBoost、GBDT、daily_direct、残差模型、Similar-month retrieval、Three-model weighted ensemble
- 历史回测评估：按月份、平台、cutoff 时点评估模型表现
- 动态选模：根据平台、月份类型、预测阶段推荐最优模型
- 支持后续扩展区间预测与 dashboard
- 

---

## 回测结果

- MONTH START： 推荐Prophet Error-Approx. ±12% 
- MID-MONTH： 推荐MTD progress%  Error-Approx. ±5%   
- MONTH END： 推荐MTD progress% Error-Approx. ±2%

---

## 当前已接入的主要模型

### cutoff = 0（整月预测）
- `history_average`
- `last_year_same_month`
- `yoy_growth_extrapolation`
- `prophet_fullmonth`
- `xgboost_v2_event`
- `gbdt_daily_direct_c0`
- `xgboost_daily_direct_c0`
- `prophet_daily`

### cutoff > 0（月中预测 / 剩余销量预测）
- `history_average`
- `mtd_progress_p50`
- `prophet_mixed`
- `xgboost_residual`
- `gbdt_daily_direct_c{cutoff}`
- `xgboost_daily_direct_c{cutoff}`
- `prophet_daily`

---

## cutoff_day 口径说明

| cutoff_day | 含义 |
|---|---|
| `0` | 截至上月月底，预测下个月整月 |
| `1~31` | 截至当月第 N 天，预测当月剩余天数 |

### 规则
- `cutoff_day < 0`：非法
- `cutoff_day = 0`：整月预测场景
- `cutoff_day > 当月天数`：自动按当月最后一天处理
- `cutoff_day = 当月最后一天`：剩余预测区间为 0

---

## 目录结构

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

---

## 最小原始数据字段

建议至少包含：
- `date`
- `platform`
- `sales`

当前默认测试数据会自动从以下别名映射：
- `trans_dt -> date`
- `amount -> sales`

## 本地独立运行说明

当前项目已改为默认从项目内部读取原始数据，请将 raw data 放到：

```bash
sales_forecast/data/raw/test_forecast_data.csv
```

也就是说，下载 `sales_forecast/` 目录后，只要保留其内部目录结构，并把原始数据放进 `data/raw/`，即可在本地独立运行。

---

## 运行方式

### 1）查看状态
```bash
python main.py --action status
```

### 2）单次预测（单个 cutoff）
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 15
```

### 3）三维参数化单次预测（brand + platform + cutoff）
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand brand1 --platform 京东
```

### 4）预测某平台总品牌
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand total --platform 京东
```

### 5）预测某品牌全平台
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand brand1 --platform total
```

### 6）单个 cutoff 回测
```bash
python main.py --action backtest --cutoff-day 7
```

### 7）三维参数化回测
```bash
python main.py --action backtest --cutoff-days 0,19 --brand brand1 --platform 京东
```

### 8）多个 cutoff 回测
```bash
python main.py --action backtest --cutoff-days 0,7,14,21,28
```

### 9）传统固定点回测（等价示例）
```bash
python main.py --action backtest --cutoff-days 0,5,10,15,20,25
```

---

## daily_direct 方法说明

### cutoff = 0
使用截至上月月底的数据，**直接预测下个月每天销量**，再按月汇总评估。
不做逐日递推。

### cutoff > 0
使用截至当月第 N 天的已知信息，**直接预测剩余每天销量**，再与已发生销量相加得到整月预测。
同样不做逐日递推。

---

## 当前推荐经验（基于现阶段回测）

### cutoff = 0
- total：`prophet_daily` 常较稳
- 京东：`last_year_same_month` 常较稳
- 阿里：`yoy_growth_extrapolation` / `prophet_daily` 可重点比较

### cutoff > 0
- `mtd_progress_p50` 当前通常最稳
- `xgboost_residual` 在部分早期 cutoff 可作为增强候选
- `prophet_mixed` 可以保留为候选，但当前通常不是最优

> 注意：以上是当前测试数据下的经验结论，不代表换数据后仍完全成立。

---

## 输出文件

典型输出位置：
- `data/backtest_result/backtest_v1.csv`
- `data/backtest_result/backtest_v1_summary.csv`
- `data/backtest_result/final_model_recommendation_by_platform_cutoff.xlsx`
- `data/backtest_result/final_model_recommendation_by_platform_cutoff.csv`

---

## 后续可补充

- 动态选模规则自动生成
- 推荐模型表自动导出
- Streamlit dashboard
- README 中加入最新回测快照
