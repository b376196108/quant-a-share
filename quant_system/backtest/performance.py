"""回测绩效指标与结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """
    回测结果数据类。

    属性：
        equity_curve：账户总资产曲线。
        returns：日收益率序列。
        trades：逐笔交易记录。
        stats：汇总指标（total_return、annual_return、max_drawdown、sharpe 等）。
    """

    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    stats: Dict[str, float]


def calc_max_drawdown(equity: pd.Series) -> float:
    """计算最大回撤，返回负数（如 -0.25）。"""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())


def calc_annual_return(returns: pd.Series, trading_days: int = 252) -> float:
    """
    根据日收益率估算年化收益率。
    采用几何收益率：((1+总收益) ** (年度交易日/样本长度) - 1)。
    """
    if returns.empty:
        return 0.0
    total_return = float((1.0 + returns).prod() - 1.0)
    periods = len(returns)
    if periods <= 0:
        return 0.0
    return float((1.0 + total_return) ** (trading_days / periods) - 1.0)


def calc_sharpe(returns: pd.Series, risk_free: float = 0.0, trading_days: int = 252) -> float:
    """
    计算夏普比率：((平均超额收益) / 标准差) * sqrt(年度交易日)。
    risk_free 为年化无风险收益率。
    """
    if returns.empty:
        return 0.0
    daily_rf = risk_free / trading_days
    excess = returns - daily_rf
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(trading_days))
