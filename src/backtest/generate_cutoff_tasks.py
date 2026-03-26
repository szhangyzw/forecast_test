"""生成回测任务。"""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ForecastTask:
    target_month: str
    cutoff_day: int
    platform: str


def generate_tasks(months: Iterable[str], cutoff_days: Iterable[int], platforms: Iterable[str]):
    return [ForecastTask(m, c, p) for m in months for c in cutoff_days for p in platforms]
