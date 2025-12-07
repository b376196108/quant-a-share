from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 确保可导入 quant_system 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.indicators.engine import calculate_indicators  # noqa: E402
from quant_system.strategy import get_strategy, list_strategies  # noqa: E402
from quant_system.strategy.base_strategy import StrategyContext  # noqa: E402
from quant_system.backtest.engine import BacktestConfig, BacktestEngine  # noqa: E402


def _load_stock_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    优先调用数据模块；失败时返回一份模拟数据，保证示例可运行。
    """
    try:
        from quant_system.data.fetcher import get_stock_data

        df = get_stock_data(code=code, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            return df
        print("[demo] 数据接口返回空，将使用模拟数据。")
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] 读取真实数据失败：{exc}，将使用模拟数据。")

    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    base_price = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
    close = pd.Series(base_price).clip(lower=1.0).round(2)
    open_price = close * (1 + np.random.normal(0, 0.002, size=len(dates)))
    high = np.maximum(open_price, close) * (1 + np.random.uniform(0, 0.002, size=len(dates)))
    low = np.minimum(open_price, close) * (1 - np.random.uniform(0, 0.002, size=len(dates)))
    volume = np.random.randint(1_000_000, 5_000_000, size=len(dates))

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


def main() -> None:
    code = "sh.600519"
    start_date = "2023-01-01"
    end_date = "2024-12-31"

    print("可用策略：", list_strategies())
    StrategyCls = get_strategy("ma_rsi_long_only")

    strategy_params = {
        "fast_ma": 20,
        "slow_ma": 30,
        "rsi_lower": 25.0,
        "rsi_upper": 80.0,
    }
    strategy = StrategyCls(params=strategy_params)
    print("=== 使用的策略参数 ===")
    for k, v in strategy_params.items():
        print(f"{k}: {v}")
    print()

    df = _load_stock_data(code, start_date, end_date)
    df_with_ind = calculate_indicators(df, indicators=strategy.required_indicators())

    ctx = StrategyContext(code=code, initial_cash=100_000)
    signals = strategy.generate_signals(df_with_ind, context=ctx)
    df_with_ind["signal"] = signals

    engine = BacktestEngine(BacktestConfig(initial_cash=ctx.initial_cash, fee_rate=ctx.fee_rate, slippage=ctx.slippage))
    result = engine.run(df_with_ind)

    print("=== 回测结果 ===")
    print(f"总收益率: {result.stats.get('total_return', 0):.2%}")
    print(f"年化收益: {result.stats.get('annual_return', 0):.2%}")
    print(f"最大回撤: {result.stats.get('max_drawdown', 0):.2%}")
    print(f"夏普比率: {result.stats.get('sharpe', 0):.2f}")
    print("\n交易记录示例：")
    if not result.trades.empty:
        print(result.trades.head())
    else:
        print("暂无交易记录（可能未触发信号）。")


if __name__ == "__main__":
    main()
