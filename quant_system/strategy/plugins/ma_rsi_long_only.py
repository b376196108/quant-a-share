"""均线金叉 + RSI 过滤的多头示例策略。"""

from __future__ import annotations

import pandas as pd

from quant_system.strategy import register_strategy
from quant_system.strategy.base_strategy import BaseStrategy, StrategyContext


@register_strategy
class MaRsiLongOnly(BaseStrategy):
    """
    简单多头策略：
        - 快速均线向上穿越慢速均线，且 RSI > rsi_lower 时开多/持有；
        - 快速均线下穿慢速均线，或 RSI > rsi_upper 时平仓；
        - 其余时间保持前一日信号。

    主要参数：
        fast_ma：快线窗口
        slow_ma：慢线窗口
        rsi_lower：RSI 低阈值（低于该值才允许开仓）
        rsi_upper：RSI 高阈值（高于该值则平仓）
    """

    name = "ma_rsi_long_only"
    default_params = {
        "fast_ma": 5,
        "slow_ma": 20,
        "rsi_lower": 30.0,
        "rsi_upper": 70.0,
    }

    def required_indicators(self) -> list[str]:
        # 依赖基础 SMA 和 RSI 插件，默认会至少生成 SMA_20 和 RSI_14。
        return ["sma", "rsi"]

    def _ensure_series(
        self,
        df: pd.DataFrame,
        col_name: str,
        window: int,
        price_col: str = "close",
    ) -> pd.Series:
        """若缺失指定均线列则临时计算。"""
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce")
        return pd.to_numeric(df[price_col].rolling(window).mean(), errors="coerce")

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: StrategyContext | None = None,
    ) -> pd.Series:
        fast_ma = int(self.params.get("fast_ma", self.default_params["fast_ma"]))
        slow_ma = int(self.params.get("slow_ma", self.default_params["slow_ma"]))
        rsi_lower = float(self.params.get("rsi_lower", self.default_params["rsi_lower"]))
        rsi_upper = float(self.params.get("rsi_upper", self.default_params["rsi_upper"]))
        rsi_length = int(self.params.get("rsi_length", 14))

        close_col = "close"
        fast_col = f"SMA_{fast_ma}"
        slow_col = f"SMA_{slow_ma}"
        rsi_col = f"RSI_{rsi_length}"

        fast_sma = self._ensure_series(df, fast_col, fast_ma, close_col)
        slow_sma = self._ensure_series(df, slow_col, slow_ma, close_col)
        if rsi_col in df.columns:
            rsi = pd.to_numeric(df[rsi_col], errors="coerce")
        else:
            try:
                import pandas_ta as ta  # noqa: F401

                rsi = pd.to_numeric(df.ta.rsi(length=rsi_length), errors="coerce")
            except Exception:
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0.0).rolling(rsi_length).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_length).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

        golden = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        dead = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

        signals = pd.Series(0, index=df.index, dtype=int)
        prev = 0
        for i in range(len(df)):
            current = prev
            if pd.isna(fast_sma.iloc[i]) or pd.isna(slow_sma.iloc[i]) or pd.isna(rsi.iloc[i]):
                signals.iloc[i] = current
                prev = current
                continue

            # 优先处理止盈/平仓条件
            if dead.iloc[i] or rsi.iloc[i] > rsi_upper:
                current = 0
            elif golden.iloc[i] and rsi.iloc[i] > rsi_lower:
                current = 1

            signals.iloc[i] = current
            prev = current

        return signals.astype(int)
