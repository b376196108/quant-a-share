"""MACD 指标插件。"""

from __future__ import annotations

import pandas as pd

from quant_system.indicators.base_indicator import BaseIndicator
from quant_system.indicators.engine import register_indicator


@register_indicator
class MACDIndicator(BaseIndicator):
    """
    MACD（指数平滑异同移动平均）插件。

    入参要求：
        - df 包含 close 列与基础行情字段，索引为日期。
    新增列：
        - MACD_{fast}_{slow}_{signal}
        - MACDh_{fast}_{slow}_{signal}（柱状图）
        - MACDs_{fast}_{slow}_{signal}（信号线）
    """

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **_: object) -> None:
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        BaseIndicator.validate_input(df)
        try:
            import pandas_ta as ta  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 环境依赖提示
            raise ImportError("需要安装 pandas_ta 才能计算 MACD，请执行 pip install pandas-ta") from exc

        result = df.copy()
        macd_df = result.ta.macd(fast=self.fast, slow=self.slow, signal=self.signal)
        if macd_df is None or macd_df.empty:
            return result

        return result.join(macd_df)
