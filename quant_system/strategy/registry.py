# quant_system/strategy/registry.py
"""
策略注册表：统一管理所有可用的日线单标的策略。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Type

from .base_strategy import BaseStrategy


@dataclass
class StrategyMeta:
    """
    策略元信息，用于前端展示和参数配置。

    字段说明：
        id          : 唯一标识（英文短名，如 "connors_rsi2"）
        name        : 中文名，用于 UI 展示
        category    : 策略类别（如 "trend" / "reversal" / "volatility" / "volume"）
        description : 简短说明
        tags        : 若干标签（如 ["mean_reversion", "short_term"]）
        default_params : 默认参数（回测时若没传就用这里）
        param_schema   : 参数结构描述，给前端或配置中心用（可选）
    """

    id: str
    name: str
    category: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """方便 FastAPI 直接返回 JSON。"""
        return asdict(self)


# ---------------- 内部注册表 ----------------

_STRATEGY_CLASSES: Dict[str, Type[BaseStrategy]] = {}
_STRATEGY_META: Dict[str, StrategyMeta] = {}


def register_strategy(meta: StrategyMeta):
    """
    用作类装饰器，将策略类注册到全局表中。

    示例：
        @register_strategy(StrategyMeta(...))
        class MyStrategy(BaseStrategy):
            ...
    """

    def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        if not issubclass(cls, BaseStrategy):
            raise TypeError(f"{cls.__name__} 不是 BaseStrategy 的子类")

        sid = meta.id
        if sid in _STRATEGY_CLASSES:
            raise ValueError(f"策略 id 重复: {sid}")

        _STRATEGY_CLASSES[sid] = cls
        _STRATEGY_META[sid] = meta
        return cls

    return decorator


def list_strategies(category: Optional[str] = None) -> List[StrategyMeta]:
    """
    返回所有策略的元信息列表，可按类别过滤。

    以后 /api/backtest/strategies 可以直接：
        from quant_system.strategy.registry import list_strategies
        ...
    """
    if category is None:
        return list(_STRATEGY_META.values())
    return [m for m in _STRATEGY_META.values() if m.category == category]


def get_strategy_meta(strategy_id: str) -> StrategyMeta:
    try:
        return _STRATEGY_META[strategy_id]
    except KeyError:
        raise KeyError(f"未找到策略: {strategy_id!r}")


def create_strategy(strategy_id: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    """
    根据 id 创建策略实例，在回测模块里用。
    """
    try:
        cls = _STRATEGY_CLASSES[strategy_id]
    except KeyError:
        raise KeyError(f"未找到策略: {strategy_id!r}")

    return cls(params=params)

