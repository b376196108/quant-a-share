"""日线单标的策略基类与上下文定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, List

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
    
class StrategyCategory(str, Enum):
    """策略大类，对齐前端：趋势 / 反转 / 波动率 / 成交量。"""
    TREND = "trend"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    VOLUME = "volume"


@dataclass
class StrategyParamMeta:
    """单个参数的元数据，用于前端渲染表单和后端校验。"""
    name: str                # 参数字段名，例如 "rsi_period"
    label: str               # 中文名，例如 "RSI周期"
    type: str                # "int" | "float" | "bool" | "str"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    description: str = ""


@dataclass
class StrategyMeta:
    """策略元信息，既给前端用，也给回测引擎用。"""
    id: str                      # 全局唯一ID，例如 "connors_rsi2"
    name: str                    # 中文名称
    category: StrategyCategory   # 策略大类
    description: str             # 简要说明
    params: Dict[str, StrategyParamMeta] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # 可选：如 ["均值回归", "高胜率"]


class PluggableStrategy(BaseStrategy):
    """
    带有元信息的策略基类，所有“策略插件”都继承它。
    注意：仍然沿用 BaseStrategy 的 generate_signals 接口（返回 signal 序列）。
    """

    meta: StrategyMeta  # 每个子类必须定义

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        # 先用元信息里的默认值，再用外部传入覆盖
        merged: Dict[str, Any] = {
            name: p.default for name, p in self.meta.params.items()
        }
        if params:
            merged.update(params)
        super().__init__(merged)

    @classmethod
    def get_meta(cls) -> Dict[str, Any]:
        """
        给 API / 前端用：返回可 JSON 化的元信息字典。
        """
        data = asdict(cls.meta)
        data["category"] = cls.meta.category.value
        # params 从对象 -> dict
        data["params"] = {
            name: asdict(p) for name, p in cls.meta.params.items()
        }
        return data
