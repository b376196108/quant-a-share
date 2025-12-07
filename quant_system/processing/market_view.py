"""市场概览与情绪统计。"""

from __future__ import annotations

import pandas as pd

from quant_system.data.storage import get_latest_trade_date, load_stock_daily_with_industry


def _label_market_sentiment(up_ratio: float) -> str:
    """
    根据上涨占比标记市场情绪。
    规则：
        - 上涨占比 ≥ 0.65 ：高潮
        - 上涨占比 ≤ 0.35 ：冰点
        - 其余视为普通
    """
    if up_ratio >= 0.65:
        return "高潮"
    if up_ratio <= 0.35:
        return "冰点"
    return "普通"


def calc_market_overview(trade_date: str | None = None) -> pd.DataFrame:
    """
    统计指定交易日的全市场情绪概览。
    指标包含：
        总股票数、上涨/下跌/涨停/跌停家数、平均与中位数涨幅、总成交额(亿元)、上涨占比、市场情绪。
    参数：
        trade_date：交易日，格式 YYYY-MM-DD；为空时自动取 stock_daily 中最新交易日。
    返回：
        仅一行的 DataFrame，索引为交易日，列名全为中文。
    """
    if trade_date is None:
        trade_date = get_latest_trade_date(table="stock_daily")
        if trade_date is None:
            raise RuntimeError("数据库中没有日线记录，无法生成市场概览")

    df = load_stock_daily_with_industry(trade_date)
    if df.empty:
        raise RuntimeError(f"{trade_date} 当日没有任何日线数据，无法生成市场概览")

    data = df.copy()
    for col in ["close", "preclose", "amount", "pct_chg"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "pct_chg" not in data.columns:
        data["pct_chg"] = (data["close"] / data["preclose"] - 1.0) * 100.0
    else:
        computed = (data["close"] / data["preclose"] - 1.0) * 100.0
        data["pct_chg"] = data["pct_chg"].fillna(pd.to_numeric(computed, errors="coerce"))

    total_count = len(data)
    if total_count == 0:
        raise RuntimeError(f"{trade_date} 当日没有有效的行情数据")

    up_mask = data["pct_chg"] > 0
    down_mask = data["pct_chg"] < 0
    limit_up_mask = data["pct_chg"] > 9.8
    limit_down_mask = data["pct_chg"] < -9.8

    up_count = int(up_mask.sum())
    down_count = int(down_mask.sum())
    limit_up_count = int(limit_up_mask.sum())
    limit_down_count = int(limit_down_mask.sum())

    avg_pct = float(data["pct_chg"].mean())
    median_pct = float(data["pct_chg"].median())
    amount_sum = float(data["amount"].sum(min_count=1)) if "amount" in data.columns else float("nan")

    up_ratio = up_count / total_count if total_count > 0 else 0.0
    sentiment = _label_market_sentiment(up_ratio)

    overview = pd.DataFrame(
        {
            "总股票数": [total_count],
            "上涨家数": [up_count],
            "下跌家数": [down_count],
            "涨停家数": [limit_up_count],
            "跌停家数": [limit_down_count],
            "平均涨幅(%)": [round(avg_pct, 2)],
            "中位涨幅(%)": [round(median_pct, 2)],
            "总成交额(亿元)": [round(amount_sum / 1e8, 2)] if not pd.isna(amount_sum) else [float("nan")],
            "上涨占比": [round(up_ratio, 3)],
            "市场情绪": [sentiment],
        },
        index=pd.DatetimeIndex([pd.to_datetime(trade_date)], name="交易日期"),
    )
    return overview


def calc_overview_between(start: str, end: str) -> pd.DataFrame:
    """
    生成一段日期内的每日市场情绪序列。
    参数：
        start：起始日期，格式 YYYY-MM-DD。
        end：结束日期，格式 YYYY-MM-DD。
    返回：
        以日期为索引的 DataFrame，每行对应当日的市场概览（列结构与 calc_market_overview 保持一致）。
    """
    start_dt = pd.to_datetime(start, errors="coerce")
    end_dt = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("起止日期格式不合法，需使用 YYYY-MM-DD")
    if start_dt > end_dt:
        raise ValueError("开始日期不能晚于结束日期")

    frames: list[pd.DataFrame] = []
    for day in pd.date_range(start_dt, end_dt, freq="D"):
        day_str = day.strftime("%Y-%m-%d")
        try:
            daily_overview = calc_market_overview(day_str)
        except RuntimeError:
            continue
        frames.append(daily_overview)

    if not frames:
        columns = [
            "总股票数",
            "上涨家数",
            "下跌家数",
            "涨停家数",
            "跌停家数",
            "平均涨幅(%)",
            "中位涨幅(%)",
            "总成交额(亿元)",
            "上涨占比",
            "市场情绪",
        ]
        return pd.DataFrame(columns=columns)

    result = pd.concat(frames, axis=0).sort_index()
    result.index.name = "交易日期"
    return result
