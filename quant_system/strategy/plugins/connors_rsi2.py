"""
Connors RSI(2) 极限反转策略（V1：贴近期刊/书籍原始规则版本）

对应文档《策略一：Connors RSI(2) 极限反转策略》：

- 环境过滤：收盘价 > 200 日简单移动平均线 (SMA200)，只在长期上升趋势中做多；
- 入场信号：RSI(2) < 5（默认），部分激进可放宽到 < 10；
- 变体：CumRSI(2) = RSI(2) 当天 + 前一天，当 CumRSI(2) < 35 时买入（本版本中为可选增强，默认关闭）；
- 出场信号：收盘价 > 5 日简单移动平均线 (SMA5)，视为价格已回归短期均值；
- 止损：原版不设硬止损，依赖高胜率 + 分散持仓，本策略保持此设定，ATR 止损预留给回测引擎统一实现。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base_strategy import BaseStrategy, StrategyContext
from ..registry import StrategyMeta, register_strategy


# --------- 工具函数：RSI 计算（基于 Wilder 思路的简化实现） ---------


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    简单 RSI 计算：
        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度

    这里使用 rolling mean 近似 Wilder 的平滑方法，对 N=2 的短周期影响很小。
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)  # 前期空值用 50 这种中性值填充，避免极端噪音


# --------- 策略元信息 ---------


META = StrategyMeta(
    id="connors_rsi2",
    name="Connors RSI(2) 极限反转",
    category="reversal",  # 反转 / 均值回归策略
    description=(
        "适用：长期上升趋势（收盘价 > SMA200）的短线回撤/急跌。\n"
        "信号：RSI(2) < 5 触发买入，反弹到 SMA5 附近退出。\n"
        "高胜率：慢牛/趋势向上且回撤后快速修复的品种；熊市慎用。"
    ),
    tags=["mean_reversion", "short_term", "rsi"],
    # 默认参数 = 原文经典设置：
    default_params={
        "rsi_period": 2,            # RSI 周期 = 2
        "use_cum_rsi": False,       # ✅ 默认关闭累积 RSI，先走原版 RSI(2)<5
        "rsi_threshold": 5.0,       # RSI(2) < 5 视为极度超卖
        "cum_rsi_threshold": 35.0,  # 可选：CumRSI(2) < 35（两日 RSI 之和）
        "sma_short": 5,             # 出场短期均线 = SMA5
        "sma_long": 200,            # 环境过滤长期均线 = SMA200
    },
    # 预留给前端参数面板使用的 schema
    param_schema={
        "rsi_period": {"type": "int", "min": 1, "max": 5, "step": 1},
        "use_cum_rsi": {"type": "bool"},
        "rsi_threshold": {"type": "float", "min": 1.0, "max": 20.0, "step": 0.5},
        "cum_rsi_threshold": {"type": "float", "min": 10.0, "max": 80.0, "step": 1.0},
        "sma_short": {"type": "int", "min": 3, "max": 20, "step": 1},
        "sma_long": {"type": "int", "min": 100, "max": 250, "step": 5},
    },
)


@register_strategy(META)
class ConnorsRsi2Strategy(BaseStrategy):
    """
    Connors RSI(2) 极限反转策略（V1 原始规则版，只做多）。

    约定：
        - 输入 df 至少包含列：open/high/low/close/volume；
        - 本策略内部自行计算 RSI(2)、SMA5、SMA200，不依赖外部指标列；
        - 输出为 0/1 持仓序列（不做空），执行时机由回测引擎控制。
    """

    name = "connors_rsi2"

    def required_indicators(self) -> list[str]:
        """
        当前版本内部计算指标，因此不声明额外依赖。
        未来若切到统一指标引擎，可在此返回 ["rsi", "sma"] 等。
        """
        return []

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Optional[StrategyContext] = None,
    ) -> pd.Series:
        # ---------------- 参数处理 ----------------
        params: Dict[str, Any] = {**META.default_params, **(self.params or {})}

        rsi_period = int(params["rsi_period"])
        rsi_threshold = float(params["rsi_threshold"])
        cum_rsi_threshold = float(params["cum_rsi_threshold"])
        sma_short_n = int(params["sma_short"])
        sma_long_n = int(params["sma_long"])
        use_cum = bool(params["use_cum_rsi"])

        # ---------------- 数据准备 ----------------
        df = df.sort_index().copy()
        close = df["close"].astype(float)

        # 1）计算 RSI(2)、SMA5、SMA200
        rsi = compute_rsi(close, period=rsi_period)
        sma_short = close.rolling(window=sma_short_n, min_periods=sma_short_n).mean()
        sma_long = close.rolling(window=sma_long_n, min_periods=sma_long_n).mean()

        # 2）环境过滤：仅在长期多头趋势中做多（收盘价 > SMA200）
        up_trend = close > sma_long

        # 3）入场条件：
        #    默认：RSI(2) < 5 （原文规则）
        #    可选：CumRSI(2) = RSI(2)+前一日RSI(2) < 35（更稳健）
        if use_cum:
            cum_rsi = rsi + rsi.shift(1)
            entry_raw = cum_rsi < cum_rsi_threshold
        else:
            entry_raw = rsi < rsi_threshold

        entry_cond = up_trend & entry_raw

        # 4）出场条件：收盘价 > SMA5
        exit_cond = close > sma_short

        # 5）根据入场/出场条件生成 0/1 持仓信号（不做空）
        signal = pd.Series(0, index=df.index, dtype="int8")
        position = 0

        for i, idx in enumerate(df.index):
            if position == 0 and entry_cond.iat[i]:
                # 空仓且满足入场条件 → 开多
                position = 1
            elif position == 1 and exit_cond.iat[i]:
                # 持多且满足出场条件 → 平仓
                position = 0
            signal.iat[i] = position

        signal.name = "signal"
        return signal
