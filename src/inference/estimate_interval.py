"""误差区间估计占位。"""


def estimate_interval(point_forecast: float, lower_error: float, upper_error: float):
    return {
        "point": point_forecast,
        "lower": point_forecast * (1 + lower_error),
        "upper": point_forecast * (1 + upper_error),
    }
