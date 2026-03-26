"""预测阶段推荐模型规则。"""

from __future__ import annotations


def recommend_model(entity: str, cutoff_day: int) -> str:
    """根据当前实体和 cutoff 阶段给出推荐模型名。

    说明：
    - cutoff=0：整月预测
    - cutoff>0：月中/月底剩余预测
    当前规则基于 README 中已有经验规则固化，后续可替换为回测驱动的动态规则。
    """
    entity = str(entity or '').strip()

    if cutoff_day > 0:
        return 'mtd_progress_p50'

    if entity == 'total':
        return 'yoy_growth_extrapolation'
    if '京东' in entity:
        return 'last_year_same_month'
    if '阿里' in entity:
        return 'yoy_growth_extrapolation'
    return 'xgboost_v2_event'
