"""预测阶段推荐模型规则。"""

from __future__ import annotations

from src.config_runtime import get_preferred_model_override


def recommend_model(entity: str, cutoff_day: int, *, brand: str | None = None, month: int | None = None, platform: str | None = None) -> str:
    """根据当前实体和 cutoff 阶段给出推荐模型名。

    优先级：brand+month+platform > brand+month > brand > default。
    若 config 未设置 preferred_model，则回退到内置默认推荐逻辑。
    """
    entity = str(entity or '').strip()

    override = get_preferred_model_override(brand=brand, month=month, platform=platform)
    if override:
        return str(override)

    if cutoff_day > 0:
        return 'mtd_progress_p50'

    if entity == 'total':
        return 'yoy_growth_extrapolation'
    if '京东' in entity:
        return 'last_year_same_month'
    if '阿里' in entity:
        return 'yoy_growth_extrapolation'
    return 'xgboost_daily_direct_c0'
