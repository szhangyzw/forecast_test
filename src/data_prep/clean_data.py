"""数据清洗模块。"""

from pathlib import Path
import pandas as pd


COLUMN_MAPPING = {
    "trans_dt": "date",
    "amount": "sales",
}


REQUIRED_COLUMNS = ["date", "platform", "sales"]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {c: COLUMN_MAPPING[c] for c in df.columns if c in COLUMN_MAPPING}
    df = df.rename(columns=rename_map)
    return df


def clean_raw_data(path: str | Path) -> pd.DataFrame:
    df = load_raw_data(path)
    df = standardize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["platform"] = df["platform"].astype(str).str.strip()

    if "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.strip()

    df = df.dropna(subset=["date", "platform", "sales"])
    return df
