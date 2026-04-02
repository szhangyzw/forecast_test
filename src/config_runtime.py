from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'config' / 'preheat_rules.yaml'


@lru_cache(maxsize=1)
def load_runtime_rules() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def _match_rule(rule: dict[str, Any], *, brand: str | None, month: int | None, platform: str | None) -> bool:
    if rule.get('brand') is not None and str(rule.get('brand')) != str(brand):
        return False
    if rule.get('month') is not None and int(rule.get('month')) != int(month):
        return False
    if rule.get('platform') is not None and str(rule.get('platform')) != str(platform):
        return False
    return True


def _specificity(rule: dict[str, Any]) -> int:
    return sum(1 for k in ('brand', 'month', 'platform') if rule.get(k) is not None)


def _best_rule(rules: list[dict[str, Any]], *, brand: str | None, month: int | None, platform: str | None) -> dict[str, Any] | None:
    matched = [r for r in rules if _match_rule(r, brand=brand, month=month, platform=platform)]
    if not matched:
        return None
    matched.sort(key=_specificity, reverse=True)
    return matched[0]


def get_preheat_uplift(*, brand: str | None, month: int | None, platform: str | None) -> float:
    cfg = load_runtime_rules()
    preheat = cfg.get('preheat', {}) if isinstance(cfg, dict) else {}
    default_uplift = float(preheat.get('default_uplift', 1.0))
    rules = preheat.get('uplift_rules', []) or []
    rule = _best_rule(rules, brand=brand, month=month, platform=platform)
    if not rule:
        return default_uplift
    return float(rule.get('uplift', default_uplift))


def get_preferred_model_override(*, brand: str | None, month: int | None, platform: str | None) -> str | None:
    cfg = load_runtime_rules()
    pref = cfg.get('preferred_model', {}) if isinstance(cfg, dict) else {}
    default_model = pref.get('default')
    rules = pref.get('rules', []) or []
    rule = _best_rule(rules, brand=brand, month=month, platform=platform)
    if not rule:
        return default_model
    return rule.get('preferred_model', default_model)
