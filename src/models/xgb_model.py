"""XGBoost 模型。"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor


FEATURES_CUTOFF0 = [
    "month",
    "days_in_month",
    "last_month_total",
    "last_3month_avg",
    "last_6month_avg",
    "ly_same_month_total",
    "recent_yoy_growth",
    "last_3month_yoy_avg",
    "last_6month_std",
    "month_num_in_year",
    "is_cny_impact_month",
    "pre_cny_flag",
    "in_cny_flag",
    "post_cny_flag",
    "days_to_cny",
    "days_from_cny",
    "is_618_impact_month",
    "pre_618_flag",
    "in_618_flag",
    "is_double11_impact_month",
    "pre_double11_flag",
    "in_double11_flag",
]

FEATURES_CUTOFFN = [
    "month",
    "days_in_month",
    "days_elapsed",
    "days_remaining",
    "mtd_sales",
    "mtd_avg_sales",
    "mtd_progress",
    "last_7d_avg",
    "last_30d_avg",
    "ly_same_month_same_day_mtd",
    "ly_same_month_total",
    "hist_share_p50",
    "hist_share_mean",
]


def build_xgb_model(**kwargs) -> XGBRegressor:
    default_params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    default_params.update(kwargs)
    return XGBRegressor(**default_params)


def train_xgb(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> XGBRegressor | None:
    df = train_df.copy()
    usable_cols = [c for c in feature_cols if c in df.columns]
    if not usable_cols:
        return None
    df = df.dropna(subset=usable_cols + [target_col]).copy()
    if len(df) < 6:
        return None

    X = df[usable_cols]
    y = df[target_col]
    model = build_xgb_model()
    model.fit(X, y)
    model._oc_feature_cols = usable_cols
    return model


def predict_xgb(model: XGBRegressor | None, pred_df: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series([pd.NA] * len(pred_df), index=pred_df.index)

    feature_cols = list(getattr(model, '_oc_feature_cols', []))
    if not feature_cols:
        return pd.Series([pd.NA] * len(pred_df), index=pred_df.index)
    X = pred_df[feature_cols].copy()
    if X.isna().any().any():
        return pd.Series([pd.NA] * len(pred_df), index=pred_df.index)
    preds = model.predict(X)
    return pd.Series(preds, index=pred_df.index)
