"""日线单标的策略基类与上下文定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class StrategyContext:
    """
    策略运行上下文。

    参数：
        code：标的代码。
        initial_cash：初始资金。
        fee_rate：手续费率。
        slippage：滑点（元）。
        extra：预留的额外上下文字段。
    """

    code: str
    initial_cash: float = 100_000.0
    fee_rate: float = 0.0005
    slippage: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    日线单标的策略基类。

    约定：
        - 输入：带有 open/high/low/close/volume 等基础行情列 + 指标列 的 DataFrame，
                行索引为 DatetimeIndex。
        - 输出：一个包含 'signal' 列的 DataFrame 或 Series：
            signal = 1  → 开多 / 持有多头
            signal = 0  → 空仓
            signal = -1 → 平多 or 做空（当前只实现多头，所以 -1 视为平仓）
    """

    name: str = "base"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = params or {}

    @abstractmethod
    def required_indicators(self) -> list[str]:
        """
        返回本策略需要的指标名称列表，用于配合 indicators.engine.calculate_indicators。
        例如：["sma", "macd", "rsi"] 或 ["ultimate_features"]。
        """
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Optional[StrategyContext] = None,
    ) -> pd.Series:
        """
        根据已经附加指标的行情 DataFrame 生成信号序列。

        要求：
            - 返回值 index 与 df.index 对齐
            - 值域只允许 {-1, 0, 1}
        """
        raise NotImplementedError
