"""策略插件注册与加载。"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import Dict, Type

from .base_strategy import BaseStrategy, StrategyContext

_STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """装饰器：注册策略，使用类属性 name 作为 key。"""
    name = getattr(cls, "name", cls.__name__).lower()
    _STRATEGY_REGISTRY[name] = cls
    return cls


def _auto_discover_plugins() -> None:
    """扫描 strategy.plugins 包，自动 import 其中的模块，以触发 @register_strategy。"""
    from . import plugins  # noqa: F401

    pkg = plugins.__name__
    for m in iter_modules(plugins.__path__):
        import_module(f"{pkg}.{m.name}")


def list_strategies() -> list[str]:
    """返回当前已注册的策略名称列表。"""
    if not _STRATEGY_REGISTRY:
        _auto_discover_plugins()
    return sorted(_STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> Type[BaseStrategy]:
    """按名称获取策略类，若不存在则抛出 KeyError。"""
    if not _STRATEGY_REGISTRY:
        _auto_discover_plugins()
    key = name.lower()
    if key not in _STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {name}")
    return _STRATEGY_REGISTRY[key]


__all__ = [
    "BaseStrategy",
    "StrategyContext",
    "register_strategy",
    "list_strategies",
    "get_strategy",
]

