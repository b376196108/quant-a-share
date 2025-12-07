"""行业情绪统计与标签计算。

Notebook 示例：
    from quant_system.processing.industry_sentiment import calc_industry_sentiment

    df_today = calc_industry_sentiment()  # 默认最新交易日
    df_today.head()

    df_20251205 = calc_industry_sentiment("2025-12-05", min_stock_count=5)
    df_20251205.head()
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quant_system.data.storage import (
    get_latest_trade_date,
    load_stock_daily_with_industry,
)


SentimentLabel = Literal["高潮", "普通", "冰点"]


def _label_industry_sentiment(up_ratio: float, limit_up_ratio: float) -> SentimentLabel:
    """
    根据上涨占比和涨停占比打行业情绪标签。

    参数：
        up_ratio: 行业内上涨家数 / 股票总数
        limit_up_ratio: 行业内涨停家数 / 股票总数

    返回：
        "高潮" / "普通" / "冰点"
    """
    if up_ratio >= 0.7 and limit_up_ratio >= 0.05:
        return "高潮"
    if up_ratio <= 0.3 and limit_up_ratio <= 0.0:
        return "冰点"
    return "普通"


def calc_industry_sentiment(
    trade_date: str | None = None,
    min_stock_count: int = 5,
) -> pd.DataFrame:
    """
    计算指定交易日的行业情绪统计结果。

    功能：
        - 从 SQLite 读取指定交易日的全市场日线 + 行业信息；
        - 计算每个行业的涨跌统计、平均涨幅、成交额等；
        - 打上行业情绪标签（高潮 / 普通 / 冰点）；
        - 按“情绪从高到低 + 平均涨幅从高到低”进行排序；
        - 列名全部使用中文，方便直接展示。

    参数：
        trade_date: 交易日字符串，格式 "YYYY-MM-DD"。若为 None，则自动使用 stock_daily 表中的最新交易日。
        min_stock_count: 行业内最少股票数量过滤条件，小于此门槛的行业将被丢弃。

    返回：
        一个按情绪排序的 DataFrame，主要字段包括：
            - 行业
            - 股票数
            - 上涨家数
            - 下跌家数
            - 涨停家数
            - 跌停家数
            - 平均涨幅(%)
            - 中位涨幅(%)
            - 总成交额(亿元)
            - 上涨占比
            - 情绪
    """
    if trade_date is None:
        trade_date = get_latest_trade_date(table="stock_daily")
        if trade_date is None:
            raise RuntimeError("数据库中没有任何 stock_daily 数据，无法计算行业情绪。")

    df = load_stock_daily_with_industry(trade_date)
    if df.empty:
        raise RuntimeError(f"{trade_date} 当日没有任何日线数据。")

    df = df.copy()
    for col in ["close", "preclose", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pct_chg"] = (df["close"] / df["preclose"] - 1.0) * 100.0
    df["industry"] = df["industry"].fillna("未分类")

    grouped = df.groupby("industry", dropna=False)

    def _agg_func(g: pd.DataFrame) -> pd.Series:
        stock_count = len(g)
        up_mask = g["pct_chg"] > 0
        down_mask = g["pct_chg"] < 0
        limit_up_mask = g["pct_chg"] > 9.8
        limit_down_mask = g["pct_chg"] < -9.8

        up_count = int(up_mask.sum())
        down_count = int(down_mask.sum())
        limit_up_count = int(limit_up_mask.sum())
        limit_down_count = int(limit_down_mask.sum())

        avg_pct = float(g["pct_chg"].mean())
        median_pct = float(g["pct_chg"].median())
        total_amount = float(g["amount"].sum()) if "amount" in g.columns else np.nan

        up_ratio = up_count / stock_count if stock_count > 0 else 0.0
        limit_up_ratio = limit_up_count / stock_count if stock_count > 0 else 0.0

        sentiment = _label_industry_sentiment(up_ratio, limit_up_ratio)

        return pd.Series(
            {
                "股票数": stock_count,
                "上涨家数": up_count,
                "下跌家数": down_count,
                "涨停家数": limit_up_count,
                "跌停家数": limit_down_count,
                "平均涨幅(%)": round(avg_pct, 2),
                "中位涨幅(%)": round(median_pct, 2),
                "总成交额(亿元)": round(total_amount / 1e8, 2)
                if not np.isnan(total_amount)
                else np.nan,
                "上涨占比": round(up_ratio, 3),
                "情绪": sentiment,
            }
        )

    result = grouped.apply(_agg_func)
    result.index.name = "行业"
    result.reset_index(inplace=True)

    if min_stock_count > 1:
        result = result[result["股票数"] >= min_stock_count]

    sentiment_order = {"高潮": 2, "普通": 1, "冰点": 0}
    result["情绪排序值"] = result["情绪"].map(sentiment_order).fillna(0).astype(int)

    result = result.sort_values(
        by=["情绪排序值", "平均涨幅(%)"],
        ascending=[False, False],
    ).reset_index(drop=True)

    result.drop(columns=["情绪排序值"], inplace=True)

    return result
