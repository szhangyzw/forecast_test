"""模型融合模块。"""

from typing import Sequence


def simple_average(predictions: Sequence[float]) -> float:
    if not predictions:
        raise ValueError("predictions 不能为空")
    return float(sum(predictions) / len(predictions))
