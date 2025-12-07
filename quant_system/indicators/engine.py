"""技术指标引擎：注册表 + 统一计算入口。"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type, Union

import pandas as pd

from quant_system.indicators.base_indicator import BaseIndicator


_INDICATOR_REGISTRY: Dict[str, Type[BaseIndicator]] = {}
_PLUGIN_DIR = Path(__file__).resolve().parent / "plugins"
_PLUGIN_PACKAGE = "quant_system.indicators.plugins"


def register_indicator(cls: Type[BaseIndicator]) -> Type[BaseIndicator]:
    """
    装饰器：将指标类注册到全局注册表。

    - 以 cls.name（小写）为 key；
    - 若重复注册则打印提示并覆盖旧值。
    """
    name = getattr(cls, "name", "") or ""
    key = name.strip().lower()
    if not key:
        raise ValueError("指标类必须定义非空的 name 属性")

    if key in _INDICATOR_REGISTRY:
        print(f"[indicators] 指标 {key} 已存在，将覆盖旧注册。")
    _INDICATOR_REGISTRY[key] = cls
    return cls


def get_indicator(name: str) -> Type[BaseIndicator]:
    """按名称获取指标类，不存在则抛出 ValueError。"""
    key = name.strip().lower()
    if key not in _INDICATOR_REGISTRY:
        raise ValueError(f"未找到名称为 {name} 的指标插件")
    return _INDICATOR_REGISTRY[key]


def list_indicators() -> List[str]:
    """返回当前已注册的指标名称列表（按名称排序）。"""
    return sorted(_INDICATOR_REGISTRY.keys())


def _auto_import_plugins() -> None:
    """遍历 plugins 目录，导入所有子模块以触发注册。"""
    if not _PLUGIN_DIR.exists():
        return

    for module_info in pkgutil.iter_modules([str(_PLUGIN_DIR)]):
        if module_info.name.startswith("_"):
            continue
        module_path = f"{_PLUGIN_PACKAGE}.{module_info.name}"
        if module_path in sys.modules:
            continue
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[indicators] 导入插件 {module_path} 失败：{exc}")


def calculate_indicators(
    df: pd.DataFrame,
    indicators: Sequence[Union[str, BaseIndicator]] | None = None,
    **indicator_kwargs: Any,
) -> pd.DataFrame:
    """
    在日线行情 DataFrame 上按序追加指定指标列。

    参数：
        df：要求包含 open/high/low/close/volume 等基础字段，索引为日期。
        indicators：
            - None：对全部已注册指标依次计算（数据量大时谨慎使用）；
            - 序列：元素可为指标名称（字符串）或已实例化的 BaseIndicator 对象。
        indicator_kwargs：
            - 预留给未来扩展，当前仅在实例化字符串指标时作为统一参数传入。

    返回：
        附加指标列后的 DataFrame，按传入顺序链式计算。

    说明：
        - 当前仅面向日线数据；
        - 只计算指标，不产生交易信号。
    """
    BaseIndicator.validate_input(df)
    result = df.copy()

    if indicators is None:
        indicator_list: List[Union[str, BaseIndicator]] = list_indicators()
    else:
        indicator_list = list(indicators)

    for item in indicator_list:
        if isinstance(item, str):
            indicator_cls = get_indicator(item)
            if indicator_kwargs:
                try:
                    indicator = indicator_cls(**indicator_kwargs)
                except TypeError as exc:
                    raise TypeError(f"实例化指标 {item} 失败，请检查参数：{indicator_kwargs}") from exc
            else:
                indicator = indicator_cls()
        elif isinstance(item, BaseIndicator):
            indicator = item
        else:
            raise TypeError(f"指标需为名称字符串或 BaseIndicator 实例，收到：{type(item)}")

        result = indicator.compute(result)

    return result


# 模块导入时自动加载插件，确保注册表可用
_auto_import_plugins()

__all__ = [
    "calculate_indicators",
    "get_indicator",
    "list_indicators",
    "register_indicator",
    "BaseIndicator",
]
