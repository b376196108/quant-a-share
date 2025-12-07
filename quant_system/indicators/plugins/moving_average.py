"""简单移动平均线插件。"""

from __future__ import annotations

import pandas as pd

from quant_system.indicators.base_indicator import BaseIndicator
from quant_system.indicators.engine import register_indicator


@register_indicator
class SimpleMovingAverage(BaseIndicator):
    """
    简单移动平均线（SMA）插件。

    入参要求：
        - df 至少包含 price_col（默认 close）列，索引为日期。
    新增列：
        - SMA_{window}：price_col 的 rolling(window).mean()。
    """

    name = "sma"

    def __init__(self, window: int = 20, price_col: str = "close", **_: object) -> None:
        self.window = int(window)
        self.price_col = price_col

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        BaseIndicator.validate_input(df)
        if self.price_col not in df.columns:
            raise ValueError(f"SMA 计算缺少价格列：{self.price_col}")

        result = df.copy()
        col_name = f"SMA_{self.window}"
        result[col_name] = result[self.price_col].rolling(self.window).mean()
        return result
