"""单标的日线回测引擎（最小可用版本）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quant_system.backtest.performance import (
    BacktestResult,
    calc_annual_return,
    calc_max_drawdown,
    calc_sharpe,
)


@dataclass
class BacktestConfig:
    """回测配置。"""

    initial_cash: float = 100_000.0
    fee_rate: float = 0.0005
    slippage: float = 0.0  # 以“元”为单位加在价格上
    allow_short: bool = False  # 当前仅实现做多，做空占位


class BacktestEngine:
    """单标的、日线、仅多头的简单回测引擎。"""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    def _validate_df(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume", "signal"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"回测缺少必要列：{', '.join(sorted(missing))}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("回测数据索引需为 DatetimeIndex")

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        约定：
            - df.index 为 DatetimeIndex，按时间升序。
            - 必须至少包含列：'open', 'high', 'low', 'close', 'volume', 'signal'
            - signal ∈ {-1, 0, 1}，视为“目标仓位方向”（这里只做多：1=满仓多头，0=空仓）。
        """
        self._validate_df(df)
        cfg = self.config

        cash = float(cfg.initial_cash)
        position = 0  # 持仓股数

        equity_list: list[float] = []
        trade_records: list[dict[str, object]] = []

        for date, row in df.iterrows():
            raw_signal = int(row.get("signal", 0))
            signal = raw_signal
            if not cfg.allow_short and signal < 0:
                signal = 0

            price = float(row["close"])
            if np.isnan(price) or price <= 0:
                equity_list.append(cash + position * price if not np.isnan(price) else cash)
                continue

            # 买入
            if position == 0 and signal == 1:
                trade_price = price + cfg.slippage
                shares = int(cash // trade_price)
                if shares > 0:
                    cost = trade_price * shares
                    fee = cost * cfg.fee_rate
                    cash -= cost + fee
                    position += shares
                    trade_records.append(
                        {
                            "date": date,
                            "action": "buy",
                            "price": trade_price,
                            "shares": shares,
                            "fee": fee,
                            "cash_after": cash,
                            "position_after": position,
                        }
                    )

            # 卖出/平仓
            elif position > 0 and signal == 0:
                trade_price = price - cfg.slippage
                amount = trade_price * position
                fee = amount * cfg.fee_rate
                cash += amount - fee
                trade_records.append(
                    {
                        "date": date,
                        "action": "sell",
                        "price": trade_price,
                        "shares": position,
                        "fee": fee,
                        "cash_after": cash,
                        "position_after": 0,
                    }
                )
                position = 0

            equity = cash + position * price
            equity_list.append(equity)

        equity_curve = pd.Series(equity_list, index=df.index, name="equity")
        returns = equity_curve.pct_change().fillna(0.0)

        stats = {
            "total_return": float(equity_curve.iloc[-1] / cfg.initial_cash - 1.0)
        } if not equity_curve.empty else {"total_return": 0.0}
        stats["annual_return"] = calc_annual_return(returns)
        stats["max_drawdown"] = calc_max_drawdown(equity_curve)
        stats["sharpe"] = calc_sharpe(returns)

        trades_df = pd.DataFrame(trade_records)
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades_df,
            stats=stats,
        )


__all__ = ["BacktestEngine", "BacktestConfig"]
