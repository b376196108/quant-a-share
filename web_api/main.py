from datetime import date, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from quant_system.data.fetcher import get_index_data
from quant_system.data.storage import get_latest_trade_date
from quant_system.processing.industry_sentiment import calc_industry_sentiment
from quant_system.processing.market_view import calc_market_overview

app = FastAPI(title="Quant A-Share API")

# 允许前端 http://localhost:5173 调用
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/overview")
def get_market_overview(trade_date: str | None = None):
    """
    返回全市场情绪概览（对应 processing.market_view.calc_market_overview）
    trade_date 为空时使用最近一个交易日。
    """
    df = calc_market_overview(trade_date=trade_date)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No market overview data")

    # 只取第一行
    row = df.iloc[0].to_dict()
    return jsonable_encoder(row)


@app.get("/api/industry-sentiment")
def get_industry_sentiment(
    trade_date: str | None = None,
    limit: int = 20,
    min_stock_count: int = 5,
):
    """
    返回行业情绪列表（对应 processing.industry_sentiment.calc_industry_sentiment）
    默认按平均涨幅降序，并只返回前 limit 个行业。
    """
    df = calc_industry_sentiment(trade_date=trade_date, min_stock_count=min_stock_count)

    if df is None or df.empty:
        return []

    # 你可以按需要换排序字段，比如按“情绪”排序
    df_sorted = df.sort_values("平均涨幅(%)", ascending=False)

    records = df_sorted.head(limit).to_dict(orient="records")
    return jsonable_encoder(records)


@app.get("/api/index-kline")
def get_index_kline(
    symbol: str = "sh.000001",
    start: str | None = None,
    end: str | None = None,
    limit: int = 90,
):
    """
    返回指定指数的日线 K 线数据，默认为上证指数。
    参数：symbol 指数代码，start/end 日期字符串，limit 返回的最近 K 线数。
    返回：包含 time/open/high/low/close/volume/ma5/ma20 的列表。
    """
    end_date = end
    if end_date is None:
        latest_trade_date = get_latest_trade_date("index_daily")
        end_date = latest_trade_date or date.today().strftime("%Y-%m-%d")

    start_date = start or (pd.to_datetime(end_date) - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )

    df = get_index_data(symbol, start_date, end_date)
    if df is None or df.empty:
        raise HTTPException(
            status_code=404, detail="No index data found for given range."
        )

    df = df.reset_index()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()

    if limit > 0:
        df = df.tail(limit)

    records = []
    for _, row in df.iterrows():
        date_value = row.get("date")
        if pd.isna(date_value):
            continue
        time_str = pd.to_datetime(date_value).strftime("%Y-%m-%d")

        close = float(row.get("close", 0) or 0)
        ma5_val = row.get("ma5", close)
        ma20_val = row.get("ma20", close)

        records.append(
            {
                "time": time_str,
                "open": float(row.get("open", 0) or 0),
                "high": float(row.get("high", 0) or 0),
                "low": float(row.get("low", 0) or 0),
                "close": close,
                "volume": float(row.get("volume", 0) or 0),
                "ma5": float(ma5_val) if pd.notna(ma5_val) else close,
                "ma20": float(ma20_val) if pd.notna(ma20_val) else close,
            }
        )

    return jsonable_encoder(records)
