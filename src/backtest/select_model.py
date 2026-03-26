"""动态选模占位。"""


def select_best_model(summary_df):
    return summary_df.sort_values("mape").head(1)
