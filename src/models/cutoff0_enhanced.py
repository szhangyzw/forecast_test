"""cutoff=0 增强候选模型：季节+事件校正 YOY / 相似月份检索 / ensemble。"""

from __future__ import annotations

import math

import pandas as pd


EVENT_FLAG_COLS = [
    'is_618_impact_month',
    'pre_618_flag',
    'in_618_flag',
    'is_double11_impact_month',
    'pre_double11_flag',
    'in_double11_flag',
    'is_cny_impact_month',
    'pre_cny_flag',
    'in_cny_flag',
    'post_cny_flag',
]

NUMERIC_HINT_COLS = [
    'days_in_month',
    'last_month_total',
    'last_3month_avg',
    'last_6month_avg',
    'recent_yoy_growth',
    'last_3month_yoy_avg',
    'last_6month_std',
    'month_num_in_year',
]


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_target_row(target_row: pd.Series | pd.DataFrame | dict) -> pd.Series:
    if isinstance(target_row, pd.DataFrame):
        if target_row.empty:
            return pd.Series(dtype='object')
        return target_row.iloc[0]
    if isinstance(target_row, dict):
        return pd.Series(target_row)
    return target_row


def _months_gap(a: str, b: str) -> int:
    pa = pd.Period(str(a), freq='M')
    pb = pd.Period(str(b), freq='M')
    return abs((pa.year - pb.year) * 12 + (pa.month - pb.month))


def _month_cyclic_distance(m1: int, m2: int) -> float:
    diff = abs(int(m1) - int(m2))
    return min(diff, 12 - diff) / 6.0


def _build_yoy_history(hist_monthly: pd.DataFrame) -> pd.DataFrame:
    if hist_monthly is None or hist_monthly.empty:
        return pd.DataFrame()

    df = hist_monthly.copy().sort_values('year_month').reset_index(drop=True)
    if 'period' not in df.columns:
        df['period'] = pd.PeriodIndex(df['year_month'], freq='M')

    yoy = []
    for _, row in df.iterrows():
        ly = _safe_float(row.get('ly_same_month_total'))
        cur = _safe_float(row.get('month_total_sales'))
        if ly is None or cur is None or ly <= 0:
            yoy.append(pd.NA)
        else:
            yoy.append(cur / ly - 1.0)
    df['yoy_growth'] = yoy
    return df


def predict_seasonal_event_adjusted_yoy(
    hist_monthly: pd.DataFrame,
    target_row: pd.Series | pd.DataFrame | dict,
    *,
    recent_window: int = 6,
) -> float | None:
    """季节+事件校正 YOY（v2）。

    核心增强：
    1. 同月 / 同事件 regime / 最近月份加权；
    2. 估计近期 bias，并做保守修正；
    3. 对高波动月份自动收缩 toward 去年同期；
    4. 对极端 yoy 做截尾。
    """
    target = _coerce_target_row(target_row)
    ly_target = _safe_float(target.get('ly_same_month_total'))
    if ly_target is None or ly_target <= 0:
        return None

    hist = _build_yoy_history(hist_monthly)
    if hist.empty:
        return None

    target_month = str(target.get('year_month'))
    hist = hist[hist['year_month'] < target_month].copy()
    hist = hist[pd.notna(hist['yoy_growth'])].copy()
    if hist.empty:
        return None

    target_month_num = int(target.get('month_num_in_year', pd.Period(target_month, freq='M').month))
    target_days_in_month = _safe_float(target.get('days_in_month'))
    target_vol = _safe_float(target.get('last_6month_std'))
    target_last3 = _safe_float(target.get('last_3month_avg'))

    def _same_event_signature(row: pd.Series) -> float:
        score = 0.0
        for col in EVENT_FLAG_COLS:
            t = int(target.get(col, 0) or 0)
            r = int(row.get(col, 0) or 0)
            if t == 1 and r == 1:
                score += 1.0
        return score

    def _month_regime(row: pd.Series) -> str:
        if int(row.get('in_cny_flag', 0) or 0) == 1:
            return 'cny_in'
        if int(row.get('pre_cny_flag', 0) or 0) == 1:
            return 'cny_pre'
        if int(row.get('post_cny_flag', 0) or 0) == 1:
            return 'cny_post'
        if int(row.get('in_618_flag', 0) or 0) == 1:
            return '618_in'
        if int(row.get('pre_618_flag', 0) or 0) == 1:
            return '618_pre'
        if int(row.get('in_double11_flag', 0) or 0) == 1:
            return 'double11_in'
        if int(row.get('pre_double11_flag', 0) or 0) == 1:
            return 'double11_pre'
        return 'normal'

    target_regime = _month_regime(target)

    weights = []
    values = []
    weighted_bias_terms = []
    for _, row in hist.iterrows():
        yoy = _safe_float(row.get('yoy_growth'))
        if yoy is None:
            continue

        same_month = 1.0 if int(row.get('month_num_in_year', 0) or 0) == target_month_num else 0.0
        event_match = _same_event_signature(row)
        same_regime = 1.0 if _month_regime(row) == target_regime else 0.0
        month_gap = _months_gap(row['year_month'], target_month)
        recency_weight = 1.0 / (1.0 + month_gap / max(recent_window, 1))

        days_match = 0.0
        row_days = _safe_float(row.get('days_in_month'))
        if target_days_in_month is not None and row_days is not None:
            days_match = max(0.0, 1.0 - abs(target_days_in_month - row_days) / 3.0)

        weight = 1.0
        weight += 2.2 * same_month
        weight += 1.2 * same_regime
        weight += 0.8 * event_match
        weight += 1.3 * recency_weight
        weight += 0.4 * days_match

        row_ly = _safe_float(row.get('ly_same_month_total'))
        row_total = _safe_float(row.get('month_total_sales'))
        row_last3 = _safe_float(row.get('last_3month_avg'))
        if row_ly is not None and row_ly > 0 and row_total is not None:
            naive_yoy_pred = row_ly * (1.0 + yoy)
            bias_ratio = (naive_yoy_pred - row_total) / row_ly
            weighted_bias_terms.append((weight, bias_ratio))
        elif row_last3 is not None and row_last3 > 0 and row_total is not None:
            bias_ratio = (row_last3 * (1.0 + yoy) - row_total) / row_last3
            weighted_bias_terms.append((weight, bias_ratio))

        values.append(yoy)
        weights.append(weight)

    if not values:
        return None

    yoy_series = pd.Series(values, dtype='float64')
    lower = yoy_series.quantile(0.1)
    upper = yoy_series.quantile(0.9)
    clipped_values = yoy_series.clip(lower=lower, upper=upper)

    weight_series = pd.Series(weights, dtype='float64')
    if weight_series.sum() <= 0:
        blended_yoy = float(clipped_values.mean())
    else:
        blended_yoy = float((clipped_values * weight_series).sum() / weight_series.sum())

    if weighted_bias_terms:
        bias_weight_sum = sum(w for w, _ in weighted_bias_terms)
        recent_bias = sum(w * b for w, b in weighted_bias_terms) / bias_weight_sum if bias_weight_sum > 0 else 0.0
        blended_yoy = blended_yoy - 0.5 * recent_bias

    volatility_ratio = None
    if target_last3 is not None and target_last3 > 0 and target_vol is not None:
        volatility_ratio = target_vol / max(target_last3, 1.0)
    if volatility_ratio is not None:
        shrink = min(max(volatility_ratio, 0.0), 1.0)
        blended_yoy = (1.0 - 0.35 * shrink) * blended_yoy

    blended_yoy = max(min(blended_yoy, 2.0), -0.8)
    return float(max(ly_target * (1.0 + blended_yoy), 0.0))


def predict_similar_month_retrieval(
    hist_monthly: pd.DataFrame,
    target_row: pd.Series | pd.DataFrame | dict,
    *,
    top_k: int = 5,
    recent_month_limit: int = 24,
) -> float | None:
    """相似月份检索模型。

    优先使用“相似月份的 yoy 倍率”迁移到目标月去年同期；
    若目标月没有去年同期，则退化为“相似月份销量水平”迁移。
    """
    target = _coerce_target_row(target_row)
    target_month = str(target.get('year_month'))
    target_period = pd.Period(target_month, freq='M')
    ly_target = _safe_float(target.get('ly_same_month_total'))
    target_last3 = _safe_float(target.get('last_3month_avg'))

    hist = _build_yoy_history(hist_monthly)
    if hist.empty:
        return None
    hist = hist[hist['year_month'] < target_month].copy()
    if hist.empty:
        return None

    hist['month_gap'] = hist['year_month'].apply(lambda x: _months_gap(x, target_month))
    hist = hist.sort_values('month_gap').head(max(recent_month_limit, top_k)).copy()
    if hist.empty:
        return None

    def _numeric_distance(row: pd.Series, col: str, scale: float = 1.0) -> float:
        a = _safe_float(target.get(col))
        b = _safe_float(row.get(col))
        if a is None or b is None:
            return 0.0
        denom = max(abs(a), abs(b), scale, 1.0)
        return abs(a - b) / denom

    candidate_preds: list[tuple[float, float]] = []
    for _, row in hist.iterrows():
        dist = 0.0
        dist += 2.0 * _month_cyclic_distance(int(target.get('month_num_in_year', target_period.month)), int(row.get('month_num_in_year', 0) or 0))
        dist += 0.6 * _numeric_distance(row, 'days_in_month', 3.0)
        dist += 0.7 * _numeric_distance(row, 'last_month_total')
        dist += 0.9 * _numeric_distance(row, 'last_3month_avg')
        dist += 0.6 * _numeric_distance(row, 'last_6month_avg')
        dist += 0.7 * _numeric_distance(row, 'recent_yoy_growth', 0.3)
        dist += 0.5 * _numeric_distance(row, 'last_3month_yoy_avg', 0.3)
        dist += 0.4 * _numeric_distance(row, 'last_6month_std')

        for col in EVENT_FLAG_COLS:
            dist += 0.8 * abs(int(target.get(col, 0) or 0) - int(row.get(col, 0) or 0))

        recency_bonus = 1.0 / (1.0 + _safe_float(row.get('month_gap') or 0) / 6.0)
        weight = math.exp(-dist) * (1.0 + 0.6 * recency_bonus)

        pred = None
        row_ly = _safe_float(row.get('ly_same_month_total'))
        row_total = _safe_float(row.get('month_total_sales'))
        row_last3 = _safe_float(row.get('last_3month_avg'))

        if ly_target is not None and ly_target > 0 and row_ly is not None and row_ly > 0 and row_total is not None:
            yoy_ratio = row_total / row_ly
            yoy_ratio = max(min(yoy_ratio, 3.5), 0.2)
            pred = ly_target * yoy_ratio
        elif target_last3 is not None and target_last3 > 0 and row_last3 is not None and row_last3 > 0 and row_total is not None:
            level_ratio = row_total / row_last3
            level_ratio = max(min(level_ratio, 3.5), 0.2)
            pred = target_last3 * level_ratio

        if pred is None or pd.isna(pred):
            continue
        candidate_preds.append((float(weight), float(pred)))

    if not candidate_preds:
        return None

    candidate_preds = sorted(candidate_preds, key=lambda x: x[0], reverse=True)[: max(top_k, 1)]
    weight_sum = sum(w for w, _ in candidate_preds)
    if weight_sum <= 0:
        return float(sum(pred for _, pred in candidate_preds) / len(candidate_preds))
    return float(sum(w * pred for w, pred in candidate_preds) / weight_sum)


def estimate_preheat_bias_factor(
    hist_monthly: pd.DataFrame,
    target_row: pd.Series | pd.DataFrame | dict,
    *,
    base_model_col: str = 'seasonal_event_adjusted_yoy_pred',
    min_history: int = 2,
    default_factor: float = 1.0,
    max_uplift: float = 1.35,
    min_uplift: float = 0.90,
    pooled: bool = True,
) -> float:
    """估计预热月（5月/10月）bias correction 因子。

    默认使用 pooled preheat factor：
    - 只要目标月是预热月，就从历史所有预热月（5月+10月）共同估计 factor
    - factor = 历史预热月 actual / base_pred 的稳健中位数，并做截尾
    """
    target = _coerce_target_row(target_row)
    if not (int(target.get('pre_618_flag', 0) or 0) == 1 or int(target.get('pre_double11_flag', 0) or 0) == 1):
        return float(default_factor)

    if hist_monthly is None or hist_monthly.empty or base_model_col not in hist_monthly.columns:
        return float(default_factor)

    hist = hist_monthly.copy()
    target_month = str(target.get('year_month'))
    if 'year_month' in hist.columns:
        hist = hist[hist['year_month'] < target_month].copy()

    # A-1: pooled preheat factor，合并 5 月 / 10 月预热月样本
    preheat_mask = (
        (hist.get('pre_618_flag', 0) == 1) |
        (hist.get('pre_double11_flag', 0) == 1) |
        (hist.get('month_num_in_year', 0).isin([5, 10]))
    )
    hist_pooled = hist[preheat_mask].copy()

    # 若 pooled=False，可退回到同类型预热月；当前默认不走这条。
    if not pooled:
        if int(target.get('pre_618_flag', 0) or 0) == 1:
            hist_pooled = hist[(hist.get('pre_618_flag', 0) == 1) | (hist.get('month_num_in_year', 0) == 5)].copy()
        elif int(target.get('pre_double11_flag', 0) or 0) == 1:
            hist_pooled = hist[(hist.get('pre_double11_flag', 0) == 1) | (hist.get('month_num_in_year', 0) == 10)].copy()

    if hist_pooled.empty:
        return float(default_factor)

    ratios = []
    for _, row in hist_pooled.iterrows():
        actual = _safe_float(row.get('month_total_sales'))
        base_pred = _safe_float(row.get(base_model_col))
        if actual is None or base_pred is None or base_pred <= 0:
            continue
        ratios.append(actual / base_pred)

    if len(ratios) < min_history:
        return float(default_factor)

    ratio_s = pd.Series(ratios, dtype='float64')
    factor = float(ratio_s.clip(lower=ratio_s.quantile(0.1), upper=ratio_s.quantile(0.9)).median())
    return float(min(max(factor, min_uplift), max_uplift))



def apply_preheat_bias_correction(
    base_prediction: float | None,
    hist_monthly: pd.DataFrame,
    target_row: pd.Series | pd.DataFrame | dict,
    **kwargs,
) -> float | None:
    """对 cutoff=0 的基础预测做预热月 bias correction。"""
    base_pred = _safe_float(base_prediction)
    if base_pred is None:
        return None
    factor = estimate_preheat_bias_factor(hist_monthly, target_row, **kwargs)
    return float(max(base_pred * factor, 0.0))



DEFAULT_PREHEAT_UPLIFT = 1.25


def apply_fixed_preheat_uplift(
    base_prediction: float | None,
    target_row: pd.Series | pd.DataFrame | dict,
    uplift_factor: float = DEFAULT_PREHEAT_UPLIFT,
) -> float | None:
    """A-3：固定 uplift，只在 5 月/10 月预热月生效。"""
    base_pred = _safe_float(base_prediction)
    if base_pred is None:
        return None
    target = _coerce_target_row(target_row)
    is_preheat = int(target.get('pre_618_flag', 0) or 0) == 1 or int(target.get('pre_double11_flag', 0) or 0) == 1
    if not is_preheat:
        return float(base_pred)
    return float(max(base_pred * float(uplift_factor), 0.0))



def predict_cutoff0_triplet_ensemble(
    *,
    pred_last_year_same_month: float | None,
    pred_yoy_growth_extrapolation: float | None,
    pred_seasonal_event_adjusted_yoy: float | None,
    target_row: pd.Series | pd.DataFrame | dict | None = None,
    entity_name: str | None = None,
) -> float | None:
    """cutoff=0 三模型小型 ensemble（分平台动态权重版）。

    经验策略：
    - total：更偏向 last_year_same_month
    - 京东：seasonal / LY 更均衡
    - 阿里：seasonal / YOY 权重更高
    - 事件月 / 高波动月：进一步提高 seasonal 权重
    """
    target = _coerce_target_row(target_row) if target_row is not None else pd.Series(dtype='object')
    entity = str(entity_name or target.get('platform', '') or '').strip()

    preds = {
        'last_year_same_month': _safe_float(pred_last_year_same_month),
        'yoy_growth_extrapolation': _safe_float(pred_yoy_growth_extrapolation),
        'seasonal_event_adjusted_yoy': _safe_float(pred_seasonal_event_adjusted_yoy),
    }
    preds = {k: v for k, v in preds.items() if v is not None}
    if not preds:
        return None
    if len(preds) == 1:
        return float(next(iter(preds.values())))

    if entity == 'total':
        weights = {
            'last_year_same_month': 0.58,
            'yoy_growth_extrapolation': 0.14,
            'seasonal_event_adjusted_yoy': 0.28,
        }
    elif '京东' in entity:
        weights = {
            'last_year_same_month': 0.42,
            'yoy_growth_extrapolation': 0.16,
            'seasonal_event_adjusted_yoy': 0.42,
        }
    elif '阿里' in entity:
        weights = {
            'last_year_same_month': 0.28,
            'yoy_growth_extrapolation': 0.22,
            'seasonal_event_adjusted_yoy': 0.50,
        }
    else:
        weights = {
            'last_year_same_month': 0.45,
            'yoy_growth_extrapolation': 0.18,
            'seasonal_event_adjusted_yoy': 0.37,
        }

    event_intensity = sum(int(target.get(col, 0) or 0) for col in EVENT_FLAG_COLS) if len(target) else 0
    if event_intensity > 0:
        weights['seasonal_event_adjusted_yoy'] += 0.10
        weights['last_year_same_month'] -= 0.06
        weights['yoy_growth_extrapolation'] -= 0.04

    vol = _safe_float(target.get('last_6month_std')) if len(target) else None
    last3 = _safe_float(target.get('last_3month_avg')) if len(target) else None
    if vol is not None and last3 is not None and last3 > 0:
        vol_ratio = vol / max(last3, 1.0)
        if vol_ratio > 0.25:
            weights['seasonal_event_adjusted_yoy'] += 0.06
            weights['last_year_same_month'] -= 0.04
            weights['yoy_growth_extrapolation'] -= 0.02

    active_weights = {k: max(weights[k], 0.01) for k in preds.keys()}
    total_w = sum(active_weights.values())
    if total_w <= 0:
        return float(sum(preds.values()) / len(preds))

    return float(sum(preds[k] * active_weights[k] for k in preds.keys()) / total_w)
