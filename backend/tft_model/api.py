from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .predict import get_latest_price, get_recent_history, model_predict, tft_price_forecast

router = APIRouter()


class PredictResponse(BaseModel):
    signal: str        # 买入 / 卖出 / 中性
    confidence: float  # 置信度 [0,1]


class StockHistoryPoint(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ma5: float
    ma20: float


class StockForecastPoint(BaseModel):
    date: str
    predicted_close: float
    change_pct: float
    lower: float | None = None
    upper: float | None = None


class StockForecastResponse(BaseModel):
    symbol: str
    name: str | None = None
    last_date: str
    last_close: float
    method: str
    signal: str
    confidence: float
    forecast: list[StockForecastPoint]
    history: list[StockHistoryPoint] | None = None


@router.get("/api/predict", response_model=PredictResponse)
async def predict(ts_code: str, date: Optional[str] = None) -> PredictResponse:
    """
    接收股票代码和日期，返回预测信号和置信度。

    - ts_code: 前端输入的股票代码，例如 "600519" / "000001.SZ" 等；
    - date:   预测基准日期，格式 "YYYY-MM-DD"；为空则默认用今天。
    """
    try:
        signal, confidence = model_predict(ts_code, date)
    except ValueError as exc:
        # 例如：代码无数据、样本太少等
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        # 避免内部错误信息泄露到前端
        raise HTTPException(status_code=500, detail="预测服务内部错误，请稍后重试")

    return PredictResponse(signal=signal, confidence=confidence)


@router.get("/api/forecast", response_model=StockForecastResponse)
async def forecast(symbol: str, days: int = 5) -> StockForecastResponse:
    """
    统一预测接口：
    - 方向信号：TFT 三分类模型（model_predict）
    - 价格路径：TFT 回归模型（tft_price_forecast），失败时回退到 baseline 外推
    """
    # 1. 方向预测
    try:
        signal, confidence = model_predict(symbol, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="方向预测失败，请稍后重试")

    # 2. 最近收盘价 + 历史
    try:
        last_date, last_close = get_latest_price(symbol, None)
        history = get_recent_history(symbol, lookback_days=90)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="获取最近收盘价失败")

    # 3. 价格路径：优先回归 TFT，失败回退 baseline 外推
    try:
        reg_points = tft_price_forecast(symbol, None, horizon=days)
        method_desc = "tft_reg_close_q10_50_90"
        forecast_points: list[StockForecastPoint] = []
        for p in reg_points:
            pred_close = float(p["predicted_close"])
            change_pct = pred_close / last_close - 1.0
            lower = float(p.get("lower")) if p.get("lower") is not None else None
            upper = float(p.get("upper")) if p.get("upper") is not None else None
            forecast_points.append(
                StockForecastPoint(
                    date=p["date"],
                    predicted_close=pred_close,
                    change_pct=change_pct,
                    lower=lower,
                    upper=upper,
                )
            )
    except Exception:
        method_desc = f"baseline_5d_return_signal({signal}, conf={confidence:.2f})"
        forecast_points = []

        base_date = dt.datetime.strptime(last_date, "%Y-%m-%d").date()
        daily_pct = 0.0
        if signal == "买入":
            daily_pct = 0.002 * (0.5 + confidence)
        elif signal == "卖出":
            daily_pct = -0.002 * (0.5 + confidence)

        for i in range(1, max(days, 1) + 1):
            d = base_date + dt.timedelta(days=i)
            pred = float(last_close * (1 + daily_pct * i))
            change_pct = pred / last_close - 1.0

            band = 0.02 + (1.0 - confidence) * 0.05
            lower = float(pred * (1 - band))
            upper = float(pred * (1 + band))

            forecast_points.append(
                StockForecastPoint(
                    date=d.strftime("%Y-%m-%d"),
                    predicted_close=pred,
                    change_pct=change_pct,
                    lower=lower,
                    upper=upper,
                )
            )

    return StockForecastResponse(
        symbol=symbol,
        name=None,
        last_date=last_date,
        last_close=last_close,
        method=method_desc,
        forecast=forecast_points,
        signal=signal,
        confidence=confidence,
        history=history,
    )
