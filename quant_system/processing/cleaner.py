"""行情数据清洗与标准化工具。"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def standardize_ohlcv(
    df: pd.DataFrame,
    date_col: str,
    code_col: str,
    rename_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    将原始行情数据统一为标准格式：
    - 日期列转为 datetime，并设置为索引；
    - 根据 rename_map 重命名列（常用于把第三方字段映射到 open/close 等标准名）；
    - 自动按日期排序，便于后续补全和复权；
    - 除代码与日期列之外的字段全部转为浮点数。
    参数：
        df：原始 DataFrame。
        date_col：日期字段名称（重命名前的列名）。
        code_col：证券代码字段名称（重命名前的列名）。
        rename_map：可选的重命名映射，键为原列名，值为目标列名。
    返回：
        以日期为索引、数值列已统一为 float 的 DataFrame。
    """
    rename_map = rename_map or {}
    renamed_df = df.copy()
    if rename_map:
        renamed_df = renamed_df.rename(columns=rename_map)

    date_field = rename_map.get(date_col, date_col)
    code_field = rename_map.get(code_col, code_col)

    if date_field not in renamed_df.columns:
        raise ValueError(f"缺少日期列：{date_field}")
    if code_field not in renamed_df.columns:
        raise ValueError(f"缺少代码列：{code_field}")

    renamed_df[date_field] = pd.to_datetime(renamed_df[date_field], errors="coerce")
    renamed_df = renamed_df.dropna(subset=[date_field])

    numeric_cols = [c for c in renamed_df.columns if c not in {date_field, code_field}]
    for col in numeric_cols:
        renamed_df[col] = pd.to_numeric(renamed_df[col], errors="coerce")

    if code_field in renamed_df.columns:
        renamed_df = renamed_df.sort_values(by=[date_field, code_field])
    else:
        renamed_df = renamed_df.sort_values(by=[date_field])

    standardized = renamed_df.set_index(date_field)
    standardized.index = pd.DatetimeIndex(standardized.index)
    standardized = standardized.sort_index()
    return standardized


def fill_missing_trading_days(
    df: pd.DataFrame,
    calendar: Sequence[pd.Timestamp | str] | pd.Index,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    按给定交易日历补齐缺失日期，并按指定方式填充。
    参数：
        df：已按日期索引的行情数据。
        calendar：交易日序列，可为列表、DatetimeIndex 或其它可迭代日期。
        method：填充方式，当前支持 "ffill"（向前填充）。
    返回：
        索引对齐到交易日历后的 DataFrame。
    """
    if method not in {"ffill"}:
        raise ValueError("仅支持 ffill 方式填充缺失交易日")

    target_index = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    aligned = df.copy()
    if not isinstance(aligned.index, pd.DatetimeIndex):
        aligned.index = pd.to_datetime(aligned.index, errors="coerce")
    aligned = aligned.reindex(target_index, method="ffill")
    return aligned


def apply_adjustment(
    df: pd.DataFrame,
    method: str = "forward",
) -> pd.DataFrame:
    """
    复权处理入口，预留扩展能力。
    参数：
        df：已对齐的原始行情数据。
        method：复权方式占位，目前默认 "forward"。
    返回：
        暂不做任何变换，直接返回传入的 DataFrame。
    """
    _ = method  # 预留参数，未来接入前复权/后复权逻辑
    return df


def prepare_for_analysis(
    df: pd.DataFrame,
    date_col: str = "trade_date",
    code_col: str = "code",
    rename_map: Mapping[str, str] | None = None,
    calendar: Sequence[pd.Timestamp | str] | pd.Index | None = None,
    adjust_method: str = "forward",
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    面向策略/回测的统一清洗入口：
    1. standardize_ohlcv：统一列名、类型与索引；
    2. apply_adjustment：预留复权处理；
    3. 若缺失 pct_chg 列，则用前收盘计算涨跌幅；
    4. 确保输出包含 open/high/low/close/preclose/pct_chg/volume/amount。
    参数：
        df：原始行情数据。
        date_col：日期列名。
        code_col：代码列名。
        rename_map：可选的重命名映射。
        calendar：若提供则按照交易日历补齐缺口。
        adjust_method：复权方式占位。
        fill_method：补全交易日的填充方式，占位仅支持 ffill。
    返回：
        列结构统一、索引为日期的 DataFrame。
    """
    standardized = standardize_ohlcv(
        df,
        date_col=date_col,
        code_col=code_col,
        rename_map=rename_map,
    )

    if calendar is not None:
        standardized = fill_missing_trading_days(standardized, calendar, method=fill_method)

    adjusted = apply_adjustment(standardized, method=adjust_method).copy()

    if "pct_chg" in adjusted.columns:
        adjusted["pct_chg"] = pd.to_numeric(adjusted["pct_chg"], errors="coerce")
    else:
        adjusted["pct_chg"] = pd.to_numeric(
            (adjusted["close"] / adjusted["preclose"] - 1.0) * 100.0,
            errors="coerce",
        )

    if adjusted["pct_chg"].isna().any() and {"close", "preclose"}.issubset(adjusted.columns):
        computed = (adjusted["close"] / adjusted["preclose"] - 1.0) * 100.0
        adjusted["pct_chg"] = adjusted["pct_chg"].fillna(pd.to_numeric(computed, errors="coerce"))

    required_cols = [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "pct_chg",
        "volume",
        "amount",
    ]
    for col in required_cols:
        if col not in adjusted.columns:
            adjusted[col] = float("nan")
        adjusted[col] = pd.to_numeric(adjusted[col], errors="coerce")

    ordered_cols = []
    if "code" in adjusted.columns:
        ordered_cols.append("code")
    ordered_cols.extend([c for c in required_cols if c not in ordered_cols])
    remaining_cols = [c for c in adjusted.columns if c not in ordered_cols]
    adjusted = adjusted[ordered_cols + remaining_cols]

    adjusted = adjusted.sort_index()
    return adjusted
