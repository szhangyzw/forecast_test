# sales_forecast 三维参数化调用说明

当前版本已支持以下三个维度联合调用：

- `brand`
- `platform`
- `cutoff_day` / `cutoff_days`

适用于：
- 单次预测（forecast）
- 历史回测（backtest）

---

## 一、参数定义

### 1）brand
可取值：
- 具体品牌名，例如：`brand1`
- `total`：总品牌
- `all`：全部品牌（当前 forecast 已支持切片聚合；backtest 更推荐单个切片逐个跑）

### 2）platform
可取值：
- 具体平台名，例如：`京东` / `阿里`
- `total`：总平台
- `all`：全部平台（当前 forecast 已支持切片聚合；backtest 更推荐单个切片逐个跑）

### 3）cutoff_day
可取值：
- `0`：截至上个月月底，预测下个月整月
- `1~31`：截至当月第 N 天，预测当月剩余天数

### 4）cutoff_days
多个 cutoff，逗号分隔，例如：
- `0,7,14,21,28`
- `0,5,10,15,20,25`

---

## 二、单次预测示例（forecast）

### 1）总品牌 + 总平台 + cutoff=19
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand total --platform total
```

### 2）brand1 + 京东 + cutoff=19
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand brand1 --platform 京东
```

### 3）brand1 + 总平台 + cutoff=19
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 19 --brand brand1 --platform total
```

### 4）总品牌 + 阿里 + cutoff=0（整月预测）
```bash
python main.py --action forecast --target-month 2026-03 --cutoff-day 0 --brand total --platform 阿里
```

---

## 三、历史回测示例（backtest）

### 1）总品牌 + 京东 + cutoff=19
```bash
python main.py --action backtest --cutoff-day 19 --brand total --platform 京东
```

### 2）brand1 + 京东 + cutoff=0,19
```bash
python main.py --action backtest --cutoff-days 0,19 --brand brand1 --platform 京东
```

### 3）总品牌 + 总平台 + 固定点回测
```bash
python main.py --action backtest --cutoff-days 0,5,10,15,20,25 --brand total --platform total
```

### 4）总品牌 + 总平台 + 任意 cutoff 组合
```bash
python main.py --action backtest --cutoff-days 0,7,13,21,28 --brand total --platform total
```

---

## 四、结果理解

### forecast 输出
会返回该切片对应实体，例如：
- `total`
- `京东`
- `brand:brand1`
- `brand1@京东`

forecast 当前会同时输出多个模型结果，并增加推荐标记字段，例如：
- `model_name`
- `pred_total_sales`
- `pred_remaining_sales`
- `recommended_model`
- `is_recommended`

### backtest 输出
会输出该切片下的模型回测对比表，包括：
- `cutoff_day`
- `platform`（实际是切片实体）
- `model_name`
- `months`
- `mape_total`
- `mae_total`
- 等指标

---

## 五、推荐用法

### 做业务预测时
优先使用单一切片：
- `brand=total, platform=total`
- 或某个明确品牌 / 平台

### 做模型评估时
建议：
- 一次只评估一个品牌-平台切片
- 避免一上来用 `all/all` 做特别大的长跑回测

---

## 六、注意事项

1. 某些品牌 / 平台切片历史样本不足时，回测可能返回空表
2. Prophet 节点较慢，多 cutoff 组合回测耗时会明显增加
3. `cutoff_day` 超过当月天数时，会按当月最后一天处理
4. `cutoff_day=0` 与 `cutoff_day>0` 使用的是不同预测逻辑

---

## 七、一句话总结

现在 `sales_forecast` 已支持：

> **brand + platform + cutoff_day 三维参数化调用**

可用于：
- 任意品牌 / 平台 / 截止日的单次预测
- 任意品牌 / 平台 / cutoff 组合的历史回测
