"""终极特征插件：聚合多项常用技术特征。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from quant_system.indicators.base_indicator import BaseIndicator
from quant_system.indicators.engine import register_indicator


@register_indicator
class UltimateFeatures(BaseIndicator):
    """
    聚合 Supertrend、Squeeze Pro、MACD/VWMACD、EWO、RSI 等特征。

    新增列（若计算成功）：
        - SUPERTd_{st_length}_{st_multiplier}：Supertrend 方向
        - SQZPRO_ON：Squeeze Pro 挤压状态
        - MACDh_{fast}_{slow}_{signal}：传统 MACD 柱状图
        - VWMACDh_{fast}_{slow}_{signal}：成交量加权 MACD 柱状图
        - EWO_{fast}_{slow}：Elliott Wave Oscillator
        - RSI_{rsi_length}：相对强弱指标
        - MA5_UP_MA20：5 日均线上穿 20 日均线（布尔转 float）
        - vol_adj_ratio：波动率标准化量比
    """

    name = "ultimate_features"

    def __init__(
        self,
        st_length: int = 10,
        st_multiplier: float = 3.0,
        sqz_length: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_length: int = 14,
        ewo_fast: int = 5,
        ewo_slow: int = 35,
        vwma_fast: int = 12,
        vwma_slow: int = 26,
        vwma_signal: int = 9,
        vol_window: int = 20,
        **_: Any,
    ) -> None:
        self.st_length = int(st_length)
        self.st_multiplier = float(st_multiplier)
        self.sqz_length = int(sqz_length)
        self.macd_fast = int(macd_fast)
        self.macd_slow = int(macd_slow)
        self.macd_signal = int(macd_signal)
        self.rsi_length = int(rsi_length)
        self.ewo_fast = int(ewo_fast)
        self.ewo_slow = int(ewo_slow)
        self.vwma_fast = int(vwma_fast)
        self.vwma_slow = int(vwma_slow)
        self.vwma_signal = int(vwma_signal)
        self.vol_window = int(vol_window)

    def _safe_assign(self, target: pd.DataFrame, data: Dict[str, Any]) -> None:
        """按列写入，已有列则覆盖，避免 join 冲突。"""
        for col, series in data.items():
            target[col] = series

    def _calc_vwma(self, price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
        """手写 VWMA，避免 pandas_ta 版本差异。"""
        num = (price * volume).rolling(window).sum()
        den = volume.rolling(window).sum()
        return num / den

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        BaseIndicator.validate_input(df)
        try:
            import pandas_ta as ta  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 环境依赖提示
            raise ImportError("需要安装 pandas_ta 才能计算终极特征，请执行 pip install pandas-ta") from exc

        result = df.copy()

        # Supertrend 方向
        st_df = result.ta.supertrend(length=self.st_length, multiplier=self.st_multiplier)
        if st_df is not None and not st_df.empty:
            st_dir_col = next((c for c in st_df.columns if c.startswith("SUPERTd_")), None)
            if st_dir_col:
                self._safe_assign(result, {st_dir_col: st_df[st_dir_col]})
                result["SUPERTd"] = st_df[st_dir_col]

        # Squeeze Pro 状态
        try:
            sqz_df = result.ta.sqzpro(length=self.sqz_length)
        except AttributeError:
            print("[indicators] 当前 pandas_ta 版本缺少 sqzpro，已跳过 Squeeze Pro 特征。")
            sqz_df = None
        if sqz_df is not None and not sqz_df.empty:
            for col in ["SQZPRO_ON", "SQZPRO_OFF", "SQZPRO_NOISE"]:
                if col in sqz_df.columns:
                    result[col] = sqz_df[col]

                # 经典 MACD
        macd_df = result.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd_df is not None and not macd_df.empty:
            self._safe_assign(result, macd_df.to_dict(orient="series"))

            # 取 MACDh_* 列做一个通用别名 MACDh，方便后续使用
            macdh_col = next((c for c in macd_df.columns if c.startswith("MACDh_")), None)
            if macdh_col:
                result["MACDh"] = macd_df[macdh_col]


        # 成交量加权 MACD
        vwma_fast = self._calc_vwma(result["close"], result["volume"], self.vwma_fast)
        vwma_slow = self._calc_vwma(result["close"], result["volume"], self.vwma_slow)
        vw_macd = vwma_fast - vwma_slow
        vw_signal = vw_macd.ewm(span=self.vwma_signal, adjust=False).mean()
        vw_hist = vw_macd - vw_signal
        self._safe_assign(
            result,
            {
                f"VWMACD_{self.vwma_fast}_{self.vwma_slow}_{self.vwma_signal}": vw_macd,
                f"VWMACDs_{self.vwma_fast}_{self.vwma_slow}_{self.vwma_signal}": vw_signal,
                f"VWMACDh_{self.vwma_fast}_{self.vwma_slow}_{self.vwma_signal}": vw_hist,
            },
        )
                # 通用别名：VWMACDh
        result["VWMACDh"] = vw_hist


        # Elliott Wave Oscillator
        ema_fast = result["close"].ewm(span=self.ewo_fast, adjust=False).mean()
        ema_slow = result["close"].ewm(span=self.ewo_slow, adjust=False).mean()
        ewo = (ema_fast - ema_slow) / result["close"] * 100.0

        ewo_col = f"EWO_{self.ewo_fast}_{self.ewo_slow}"
        result[ewo_col] = ewo
        # 通用别名：EWO
        result["EWO"] = ewo


        # RSI
        rsi_series = result.ta.rsi(length=self.rsi_length)
        rsi_col = f"RSI_{self.rsi_length}"
        result[rsi_col] = rsi_series
        # 通用别名：RSI
        result["RSI"] = rsi_series


        # 均线关系
        ma5 = result["close"].rolling(5).mean()
        ma20 = result["close"].rolling(20).mean()
        result["MA5_UP_MA20"] = (ma5 > ma20).astype(float)

        # 波动率标准化量比
        vol_ratio = result["volume"] / result["volume"].rolling(self.vol_window).mean()
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
        volatility = result["close"].pct_change().rolling(self.vol_window).std()
        vol_adj = vol_ratio / volatility
        vol_adj = vol_adj.replace([np.inf, -np.inf], np.nan)
        result["vol_adj_ratio"] = vol_adj

        return result
