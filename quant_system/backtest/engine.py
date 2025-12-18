# quant_system/backtest/engine.py
"""
单标的日线回测引擎 + 多策略组合封装。

分两层：
1）BacktestEngine / BacktestConfig
    - 保持你原来的实现：输入已经带有 `signal` 列的 DataFrame，输出 BacktestResult。
2）run_single_backtest / combine_signals
    - 负责：
        * 从数据模块取日线行情（get_stock_data）
        * 通过策略注册表创建策略实例（create_strategy）
        * 生成每个策略的信号并按组合方式合成（AND / OR / Voting）
        * 调用 BacktestEngine.run() 完成真实回测
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_system.backtest.performance import (
    BacktestResult,
    calc_annual_return,
    calc_max_drawdown,
    calc_sharpe,
)
from quant_system.data.fetcher import get_stock_data
from quant_system.strategy.base_strategy import StrategyContext
from quant_system.strategy.registry import create_strategy


# =====================================================================
# 一、底层回测引擎（保留你原来的实现）
# =====================================================================


@dataclass
class BacktestConfig:
    """回测配置。"""

    initial_cash: float = 100_000.0
    fee_rate: float = 0.0005
    stamp_duty_rate: float = 0.0005  # sell-side stamp duty (0.05% by default)
    slippage: float = 0.0  # 以“元”为单位加在价格上
    allow_short: bool = False  # 当前仅实现做多，做空占位


    execution_price: str = "open"  # "open" | "close"; default: next-day open execution
    mark_to_market_price: str = "close"  # equity valuation price; default: close


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

        exec_col = str(cfg.execution_price or "close").strip().lower()
        if exec_col not in {"open", "close"}:
            exec_col = "close"
        mtm_col = str(cfg.mark_to_market_price or "close").strip().lower()
        if mtm_col not in {"open", "close"}:
            mtm_col = "close"

        for date, row in df.iterrows():
            raw_signal = int(row.get("signal", 0))
            signal = raw_signal
            if not cfg.allow_short and signal < 0:
                signal = 0

            close_price = float(row.get("close", np.nan))
            if np.isnan(close_price) or close_price <= 0:
                # close 无效：无法退化到可靠价格，跳过当日撮合与估值计算
                equity_list.append(equity_list[-1] if equity_list else cash)
                continue

            exec_price = float(row.get(exec_col, np.nan))
            if np.isnan(exec_price) or exec_price <= 0:
                exec_price = close_price

            mtm_price = float(row.get(mtm_col, np.nan))
            if np.isnan(mtm_price) or mtm_price <= 0:
                mtm_price = close_price

            # 买入：当前空仓且目标信号为 1 → 全仓买入
            if position == 0 and signal == 1:
                trade_price = exec_price + cfg.slippage
                if not np.isnan(trade_price) and trade_price > 0:
                    # Ensure cash covers fees (buy side only charges fee_rate)
                    shares = int(cash // (trade_price * (1.0 + cfg.fee_rate)))
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
                                "stamp_duty": 0.0,
                                "cash_after": cash,
                                "position_after": position,
                            }
                        )

            # 卖出/平仓：当前有仓位且目标信号为 0 → 全部卖出
            elif position > 0 and signal == 0:
                trade_price = exec_price - cfg.slippage
                if not np.isnan(trade_price) and trade_price > 0:
                    amount = trade_price * position
                    fee = amount * cfg.fee_rate
                    stamp = amount * cfg.stamp_duty_rate
                    total_fee = fee + stamp
                    cash += amount - total_fee
                    trade_records.append(
                        {
                            "date": date,
                            "action": "sell",
                            "price": trade_price,
                            "shares": position,
                            "fee": total_fee,
                            "stamp_duty": stamp,
                            "cash_after": cash,
                            "position_after": 0,
                        }
                    )
                    position = 0

            equity = cash + position * mtm_price
            equity_list.append(equity)

        equity_curve = pd.Series(equity_list, index=df.index, name="equity")
        returns = equity_curve.pct_change().fillna(0.0)

        if not equity_curve.empty:
            stats = {
                "total_return": float(equity_curve.iloc[-1] / cfg.initial_cash - 1.0)
            }
        else:
            stats = {"total_return": 0.0}

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


# =====================================================================
# 二、多策略信号组合 & 单标的回测封装
# =====================================================================


def combine_signals(signals: List[pd.Series], mode: str = "OR") -> pd.Series:
    """
    将多个策略信号按指定模式合成一个最终 signal 序列。

    参数：
        signals : 若干个 pd.Series，每个取值一般在 {-1, 0, 1}
        mode    : "AND" / "OR" / "VOTING"

    返回：
        combined : pd.Series，index 为日期，name="signal"
    """
    if not signals:
        raise ValueError("signals 为空，至少需要一个策略信号")

    # 对齐索引，空值按 0 处理
    sig_df = pd.concat(signals, axis=1).fillna(0)
    sig_df = sig_df.astype(float)

    mode_upper = mode.upper()

    if mode_upper == "AND":
        # 所有策略都看多（>0）才 1；其余视为 0
        combined = (sig_df.gt(0).all(axis=1)).astype(int)

    elif mode_upper == "VOTING":
        # 多数表决：对每个策略取 sign，再按行求和
        sign_matrix = np.sign(sig_df.values)
        sum_sign = sign_matrix.sum(axis=1)
        combined = pd.Series(0, index=sig_df.index)
        combined[sum_sign > 0] = 1
        combined[sum_sign < 0] = -1

    else:  # 默认 OR
        # 只要有一个策略看多（>0）就 1，否则 0
        combined = (sig_df.gt(0).any(axis=1)).astype(int)

    combined.name = "signal"
    return combined


def run_single_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy_ids: List[str],
    mode: str = "OR",
    strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0005,
    slippage: float = 0.0,
    stamp_duty_rate: float = 0.0005,
    execution_price: str = "open",
    adjustflag: str = "2",
) -> BacktestResult:
    """
    高层封装：单只股票 + 多策略组合回测。

    这就是后面 FastAPI /api/backtest/run 可以直接调用的核心函数。

    参数：
        symbol          : 股票代码（如 "600519"）
        start_date      : 回测开始日期 "YYYY-MM-DD"
        end_date        : 回测结束日期 "YYYY-MM-DD"
        strategy_ids    : 参与组合的策略 id 列表（如 ["connors_rsi2"]）
        mode            : 组合方式，"AND" / "OR" / "VOTING"
        strategy_params : 每个策略的参数字典，key=策略 id，value=参数 dict，可为 None
        initial_cash    : 初始资金
        fee_rate        : 手续费率（万分之 2.5 就填 0.00025）
        slippage        : 单边滑点（元）

    返回：
        BacktestResult（与 BacktestEngine.run 一致）
    """
    if not strategy_ids:
        raise ValueError("strategy_ids 为空，至少选择一个策略")

    params_map: Dict[str, Dict[str, Any]] = strategy_params or {}

    # 1) 拉取日线行情
    df = get_stock_data(symbol, start_date, end_date, adjust=adjustflag)
    if df is None or df.empty:
        raise ValueError(f"没有 {symbol} 在 {start_date}~{end_date} 的历史数据")

    # 确保索引是 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
        else:
            raise ValueError("stock 数据既不是 DatetimeIndex 索引，也没有 'date' 列")

    df = df.sort_index()

    # 2) 逐个策略生成信号
    signals: List[pd.Series] = []
    ctx = StrategyContext(
        code=symbol,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        extra={},
    )

    for sid in strategy_ids:
        params = params_map.get(sid) or {}
        strategy = create_strategy(sid, params)

        sig = strategy.generate_signals(df, context=ctx)

        # 统一整理成 Series
        if isinstance(sig, pd.DataFrame):
            if "signal" not in sig.columns:
                raise ValueError(f"策略 {sid} 返回的 DataFrame 中没有 'signal' 列")
            ser = sig["signal"]
        elif isinstance(sig, pd.Series):
            ser = sig
        else:
            # 万一返回的是 list / ndarray，强制转成 Series
            ser = pd.Series(sig, index=df.index)

        ser = ser.reindex(df.index).fillna(0)
        ser = ser.astype(float)
        signals.append(ser)

    # 3) 按组合方式合成一个总信号
    combined_signal = combine_signals(signals, mode=mode)

    # 可选：T+1 生效（避免前视偏差），当前版本先按 T+1 处理
    combined_signal = combined_signal.shift(1).fillna(0)

    # 4) 构造回测输入 DataFrame：在行情上加一个 signal 列
    bt_df = df.copy()
    bt_df["signal"] = combined_signal

    # 5) 调用底层 BacktestEngine
    engine = BacktestEngine(
        config=BacktestConfig(
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            stamp_duty_rate=stamp_duty_rate,
            slippage=slippage,
            allow_short=False,
            execution_price=execution_price,
            mark_to_market_price="close",
        )
    )
    result = engine.run(bt_df)
    return result


__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "combine_signals",
    "run_single_backtest",
]
