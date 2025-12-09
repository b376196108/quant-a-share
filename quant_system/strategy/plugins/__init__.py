# quant_system/strategy/plugins/__init__.py
"""
策略插件包。

提供 `load_all_plugins()` 动态加载本目录下的所有策略模块，
确保它们的 @register_strategy 装饰器被执行。
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import List


def load_all_plugins() -> List[ModuleType]:
    """
    动态导入本包下的所有 .py 策略文件（排除以下划线开头的模块）。

    用法示例（在回测或 API 启动时调用一次即可）：
        from quant_system.strategy.plugins import load_all_plugins
        from quant_system.strategy.registry import list_strategies

        load_all_plugins()
        metas = [m.to_dict() for m in list_strategies()]
    """
    modules: List[ModuleType] = []

    # __path__ 是包级变量，指向当前 package 的搜索路径
    for info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        modules.append(module)

    return modules
