"""技术指标基类，定义统一的计算接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class BaseIndicator(ABC):
    """
    技术指标抽象基类。

    约定：
        - 入参 df 必须至少包含 open、high、low、close、volume 字段；
        - 索引建议为 DatetimeIndex，且按时间升序排列；
        - compute 在 df 上追加指标列后返回 DataFrame（可为原对象或拷贝）。

    子类需设置类属性 name 作为唯一标识，并实现 compute。
    """

    name: ClassVar[str] = ""

    @staticmethod
    def validate_input(df: pd.DataFrame) -> None:
        """校验行情基础字段是否齐全。"""
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"缺少基础行情字段：{', '.join(sorted(missing))}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("指标计算要求 df 索引为 DatetimeIndex")

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在传入的行情 DataFrame 上追加本指标产生的列。

        参数：
            df：包含 open/high/low/close/volume 等基础字段的日线数据，索引为日期。
        返回：
            包含新增指标列的 DataFrame。
        """
        raise NotImplementedError
