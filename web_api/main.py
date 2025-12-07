from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from quant_system.processing.market_view import calc_market_overview
from quant_system.processing.industry_sentiment import calc_industry_sentiment

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
