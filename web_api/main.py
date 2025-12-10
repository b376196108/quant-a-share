from datetime import date, timedelta
from typing import Any, Dict, List
import math
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from quant_system.data.fetcher import get_index_data, get_stock_data
from quant_system.data.storage import get_latest_trade_date
from quant_system.processing.industry_sentiment import calc_industry_sentiment
from quant_system.processing.market_view import calc_market_overview
from quant_system.strategy.plugins import load_all_plugins
from quant_system.strategy.registry import list_strategies, StrategyMeta
from quant_system.backtest.engine import run_single_backtest

app = FastAPI(title="Quant A-Share API")

def _load_cors_origins() -> list[str]:
    """
    Load allowed origins from env `CORS_ALLOW_ORIGINS` (comma separated).
    Falls back to common local hosts used by Vite preview/dev builds.
    """
    raw = os.getenv("CORS_ALLOW_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://webui:5173",
        "http://webui",
    ]


origins = _load_cors_origins()
# Wildcard origins cannot be combined with credentials in CORS responses.
allow_credentials = "*" not in origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时加载所有策略插件，完成注册
load_all_plugins()


# ====================== 通用模型定义 ======================

class StrategyMetaOut(BaseModel):
    id: str
    name: str
    category: str
    description: str = ""
    tags: List[str] = []
    default_params: Dict[str, Any] = {}
    param_schema: Dict[str, Any] = {}


class BacktestStrategyIn(BaseModel):
    """前端勾选的单个策略项。"""
    id: str
    params: Dict[str, Any] = {}


class BacktestRequest(BaseModel):
    """
    回测请求体，对接黄框里的参数：
      - symbol: 前端输入的股票代码（建议 6 位，如 600519）
      - start_date/end_date: 字符串 "YYYY-MM-DD"
      - strategies: 勾选的策略列表
      - mode: "AND" / "OR" / "VOTING"
      - initial_capital: 初始资金
      - fee_rate_bps: 手续费（万分比），前端填 2.5 就是 0.00025
      - slippage: 滑点（元/股），目前先按绝对值处理
    """
    symbol: str
    start_date: str
    end_date: str
    strategies: List[BacktestStrategyIn]
    mode: str = "OR"
    initial_capital: float = 100_000.0
    fee_rate_bps: float = 2.5
    slippage: float = 0.01


def normalize_symbol(code: str) -> str:
    """
    把前端输入的 6 位代码转换成带前缀的 9 位格式：
      600xxx / 601xxx / 603xxx / 688xxx -> sh.XXXXXX
      其余 6 位数字 -> sz.XXXXXX
    如果已经是 sh.XXXXXX / sz.XXXXXX 则原样返回。
    """
    code = code.strip()
    if len(code) == 9 and "." in code:
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        else:
            return f"sz.{code}"
    return code


def _safe_float(val: Any, default: float = 0.0) -> float:
    """
    Convert a value to float; if it is NaN/Inf or invalid, return a safe default
    so the JSON response stays serializable.
    """
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _safe_int(val: Any, default: int = 0) -> int:
    """Convert value to int with fallback for invalid inputs."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ====================== 策略相关 API ======================

@app.get("/api/backtest/strategies", response_model=List[StrategyMetaOut])
async def get_backtest_strategies() -> List[StrategyMetaOut]:
    """
    返回当前已注册的所有策略元信息，用于前端“策略选择”面板。
    """
    metas: List[StrategyMeta] = list_strategies()
    return [StrategyMetaOut(**m.to_dict()) for m in metas]


@app.post("/api/backtest/run")
async def run_backtest(req: BacktestRequest):
    """
    运行单标的、多策略组合回测。

    前端只需把黄框里的内容封装成 BacktestRequest 发过来，
    这里负责：
      - 代码格式转换（600519 -> sh.600519）
      - fee_rate 单位转换（万分比 -> 小数）
      - 拆出 strategy_ids / strategy_params
      - 调用 run_single_backtest() 执行回测
      - 整理返回给前端的净值曲线 & 绩效指标 & 交易明细
    """
    if not req.strategies:
        raise HTTPException(status_code=400, detail="至少需要选择一个策略")

    symbol_full = normalize_symbol(req.symbol)

    # 预拉一份原始价格序列，方便前端画 K 线（如果失败不影响回测本身）
    price_df = None
    try:
        price_df = get_stock_data(
            symbol_full, req.start_date, req.end_date, freq="d", fields=None, adjust="2"
        )
    except Exception:
        price_df = None

    strategy_ids = [s.id for s in req.strategies]
    strategy_params: Dict[str, Dict[str, Any]] = {
        s.id: (s.params or {}) for s in req.strategies
    }

    try:
        result = run_single_backtest(
            symbol=symbol_full,
            start_date=req.start_date,
            end_date=req.end_date,
            strategy_ids=strategy_ids,
            mode=req.mode,
            strategy_params=strategy_params,
            initial_cash=req.initial_capital,
            fee_rate=req.fee_rate_bps / 10_000.0,  # 万分比 -> 小数
            slippage=req.slippage,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 绩效指标（确保都是 Python float，便于 JSON 序列化）
    stats = {k: _safe_float(v) for k, v in result.stats.items()}

    # 净值曲线：转换成 list[{date, equity}]，方便前端画图
    equity_curve = [
        {"date": idx.strftime("%Y-%m-%d"), "equity": _safe_float(val)}
        for idx, val in result.equity_curve.items()
    ]

    price_series = []
    if price_df is not None and not price_df.empty:
        price_df = price_df.reset_index()
        if "date" in price_df.columns:
            price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
        price_df = price_df.dropna(subset=["date"]).sort_values("date")

        price_series = [
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "open": _safe_float(row.get("open", 0)),
                "high": _safe_float(row.get("high", 0)),
                "low": _safe_float(row.get("low", 0)),
                "close": _safe_float(row.get("close", 0)),
                "volume": _safe_float(row.get("volume", 0)),
            }
            for _, row in price_df.iterrows()
        ]

    # 交易明细：如果前端暂时不用，可以先不画表，但数据我们先给全
    trades = []
    if result.trades is not None and not result.trades.empty:
        for _, row in result.trades.iterrows():
            dt = row.get("date")
            trades.append(
                {
                    "date": dt.strftime("%Y-%m-%d") if isinstance(dt, pd.Timestamp) else None,
                    "action": row.get("action"),
                    "price": _safe_float(row.get("price", 0)),
                    "shares": _safe_int(row.get("shares", 0)),
                    "fee": _safe_float(row.get("fee", 0)),
                    "cash_after": _safe_float(row.get("cash_after", 0)),
                    "position_after": _safe_int(row.get("position_after", 0)),
                }
            )

    # 基于交易明细估算一个胜率，方便前端直接展示
    win_count = 0
    loss_count = 0
    last_buy_price = None
    last_buy_shares = 0
    last_buy_fee = 0.0
    for t in trades:
        action = (t.get("action") or "").lower()
        if action == "buy":
            last_buy_price = _safe_float(t.get("price"))
            last_buy_shares = _safe_int(t.get("shares"))
            last_buy_fee = _safe_float(t.get("fee"))
        elif action == "sell" and last_buy_price is not None and last_buy_shares > 0:
            sell_fee = _safe_float(t.get("fee"))
            sell_price = _safe_float(t.get("price"))
            profit = (sell_price - _safe_float(last_buy_price)) * last_buy_shares - (
                _safe_float(last_buy_fee) + sell_fee
            )
            if profit >= 0:
                win_count += 1
            else:
                loss_count += 1
            last_buy_price = None
            last_buy_shares = 0
            last_buy_fee = 0.0

    trade_count = win_count + loss_count
    if trade_count > 0:
        stats["win_rate"] = win_count / trade_count
    stats["trade_count"] = trade_count

    resp = {
        "symbol_input": req.symbol,
        "symbol": symbol_full,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "mode": req.mode,
        "strategy_ids": strategy_ids,
        "stats": stats,
        "equity_curve": equity_curve,
        "trades": trades,
        "price_series": price_series,
    }
    return jsonable_encoder(resp)


# ====================== 原有市场总览 & 指数 K 线 API ======================

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

    # 按情绪高->低，再按涨幅降序
    sentiment_order = {"高潮": 2, "普通": 1, "冰点": 0}
    df = df.copy()
    df["情绪排序值"] = df["情绪"].map(sentiment_order).fillna(0).astype(int)
    df_sorted = df.sort_values(
        by=["情绪排序值", "平均涨幅(%)"],
        ascending=[False, False],
    )

    records = df_sorted.head(limit).drop(columns=["情绪排序值"]).to_dict(orient="records")
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
