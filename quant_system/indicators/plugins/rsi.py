"""RSI 指标插件。"""

from __future__ import annotations

import pandas as pd

from quant_system.indicators.base_indicator import BaseIndicator
from quant_system.indicators.engine import register_indicator


@register_indicator
class RSIIndicator(BaseIndicator):
    """
    相对强弱指标（RSI）插件。

    入参要求：
        - df 包含 close 列与基础行情字段，索引为日期。
    新增列：
        - RSI_{length}
    """

    name = "rsi"

    def __init__(self, length: int = 14, **_: object) -> None:
        self.length = int(length)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        BaseIndicator.validate_input(df)
        try:
            import pandas_ta as ta  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 环境依赖提示
            raise ImportError("需要安装 pandas_ta 才能计算 RSI，请执行 pip install pandas-ta") from exc

        result = df.copy()
        rsi_series = result.ta.rsi(length=self.length)
        col_name = f"RSI_{self.length}"
        result[col_name] = rsi_series
        return result
