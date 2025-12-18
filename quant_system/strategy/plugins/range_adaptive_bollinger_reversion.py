"""
横盘震荡（均值回归）高胜率策略（真实数据回测版）

核心思想（结合 docs/横盘震荡1策略.docx + docs/横盘震荡2.docx 精华）：
- 用 ADX + CI + Bandwidth 压缩 + 成交量，确认“横盘震荡”环境，只在最适合均值回归的状态出手
- 用 Bollinger 下轨 + RSI 超卖寻找入场；用回到中轨快速止盈，提高胜率
- 用 Bandwidth 单日放大作为“突破/出横盘”信号，强制退出并进入冷却期

注意：
- 仅做多；输出为“当日收盘计算的目标仓位(0/1)”，不在策略内 shift，T+1 由回测层处理。
- 不引入任何新依赖（仅 pandas / numpy）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from quant_system.strategy import register_strategy as legacy_register_strategy
from quant_system.strategy.base_strategy import BaseStrategy, StrategyContext

from ..registry import StrategyMeta, register_strategy


def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _is_valid_price(x: float) -> bool:
    return bool(np.isfinite(x) and x > 0)


def _bollinger(close: pd.Series, window: int, n_std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(np.isfinite(rsi), np.nan)

    # Wilder 常见处理：无涨无跌时 RSI 取 50；仅有涨(无跌)时 100；仅有跌(无涨)时 0
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    only_gain = (avg_gain > 0.0) & (avg_loss == 0.0)
    only_loss = (avg_gain == 0.0) & (avg_loss > 0.0)
    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask(only_gain, 100.0)
    rsi = rsi.mask(only_loss, 0.0)
    return rsi


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = (-low.diff())

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)
    alpha = 1.0 / float(period)

    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False, min_periods=period
    ).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False, min_periods=period
    ).mean()

    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm_smooth / atr_safe)
    minus_di = 100.0 * (minus_dm_smooth / atr_safe)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx


def _choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    denom = (hh - ll).replace(0.0, np.nan)

    ratio = (sum_tr / denom).replace(0.0, np.nan)
    ci = 100.0 * (np.log10(ratio) / np.log10(float(period)))
    return ci


def _rolling_percent_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank_last(x: np.ndarray) -> float:
        last = x[-1]
        if not np.isfinite(last):
            return np.nan
        arr = x[np.isfinite(x)]
        if arr.size <= 1:
            return 0.0
        cnt = float(np.sum(arr <= last))
        return float((cnt - 1.0) / float(arr.size - 1))

    return series.rolling(window=window, min_periods=window).apply(_rank_last, raw=True)


META = StrategyMeta(
    id="range_adaptive_bollinger_reversion",
    name="横盘震荡-布林均值回归(高胜率)",
    category="reversal",
    description=(
        "仅做多的横盘震荡均值回归：用 ADX/CI/Bollinger Bandwidth/量能确认横盘，"
        "在下轨+RSI超卖介入，回到中轨即止盈；带宽放大视为突破，强制退出并冷却。"
    ),
    tags=["range", "mean_reversion", "bollinger", "rsi", "adx", "ci"],
    default_params={
        # Bollinger / RSI
        "boll_window": 20,
        "boll_n_std": 2.0,
        "rsi_period": 14,
        "rsi_entry": 40.0,
        # Regime filter: ADX / CI / Bandwidth / Volume / Trend
        "adx_period": 14,
        "adx_max": 25.0,
        "require_adx_falling": True,
        "ci_period": 14,
        "ci_min": 61.8,
        "bw_prank_window": 126,  # ~6 months trading days
        "bw_prank_max": 0.2,     # bottom 20%
        "bw_jump_stop": 0.2,     # +20% day-over-day bandwidth jump => breakout
        "cooldown_days_after_breakout": 10,
        "volume_ma_window": 5,
        "require_above_ma200": True,
        # Trade management
        "max_hold_days": 5,
        "band_walk_days": 2,
        # Entry candle filter
        "enable_falling_knife_filter": True,
        "lower_shadow_body_ratio_max": 2.0,
        # Data hygiene filters (optional)
        "avoid_st": True,
        "avoid_suspended": True,
    },
    param_schema={
        "boll_window": {"type": "int", "min": 5, "max": 60, "step": 1},
        "boll_n_std": {"type": "float", "min": 1.0, "max": 4.0, "step": 0.25},
        "rsi_period": {"type": "int", "min": 2, "max": 50, "step": 1},
        "rsi_entry": {"type": "float", "min": 5.0, "max": 50.0, "step": 0.5},
        "adx_period": {"type": "int", "min": 5, "max": 50, "step": 1},
        "adx_max": {"type": "float", "min": 5.0, "max": 50.0, "step": 0.5},
        "require_adx_falling": {"type": "bool"},
        "ci_period": {"type": "int", "min": 5, "max": 50, "step": 1},
        "ci_min": {"type": "float", "min": 30.0, "max": 90.0, "step": 0.5},
        "bw_prank_window": {"type": "int", "min": 20, "max": 252, "step": 1},
        "bw_prank_max": {"type": "float", "min": 0.05, "max": 0.5, "step": 0.01},
        "bw_jump_stop": {"type": "float", "min": 0.05, "max": 1.0, "step": 0.01},
        "cooldown_days_after_breakout": {"type": "int", "min": 0, "max": 60, "step": 1},
        "volume_ma_window": {"type": "int", "min": 2, "max": 60, "step": 1},
        "require_above_ma200": {"type": "bool"},
        "max_hold_days": {"type": "int", "min": 1, "max": 30, "step": 1},
        "band_walk_days": {"type": "int", "min": 1, "max": 10, "step": 1},
        "enable_falling_knife_filter": {"type": "bool"},
        "lower_shadow_body_ratio_max": {"type": "float", "min": 0.1, "max": 10.0, "step": 0.1},
        "avoid_st": {"type": "bool"},
        "avoid_suspended": {"type": "bool"},
    },
)


@register_strategy(META)
@legacy_register_strategy
class RangeAdaptiveBollingerReversion(BaseStrategy):
    name = META.id

    def required_indicators(self) -> list[str]:
        return []

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Optional[StrategyContext] = None,
    ) -> pd.Series:
        params: Dict[str, Any] = {**META.default_params, **(self.params or {})}

        boll_window = int(params["boll_window"])
        boll_n_std = float(params["boll_n_std"])
        rsi_period = int(params["rsi_period"])
        rsi_entry = float(params["rsi_entry"])

        adx_period = int(params["adx_period"])
        adx_max = float(params["adx_max"])
        require_adx_falling = bool(params["require_adx_falling"])
        ci_period = int(params["ci_period"])
        ci_min = float(params["ci_min"])
        bw_prank_window = int(params["bw_prank_window"])
        bw_prank_max = float(params["bw_prank_max"])
        bw_jump_stop = float(params["bw_jump_stop"])
        cooldown_days_after_breakout = int(params["cooldown_days_after_breakout"])
        volume_ma_window = int(params["volume_ma_window"])
        require_above_ma200 = bool(params["require_above_ma200"])

        max_hold_days = int(params["max_hold_days"])
        band_walk_days = int(params["band_walk_days"])
        enable_falling_knife_filter = bool(params["enable_falling_knife_filter"])
        lower_shadow_body_ratio_max = float(params["lower_shadow_body_ratio_max"])

        avoid_st = bool(params["avoid_st"])
        avoid_suspended = bool(params["avoid_suspended"])

        df = df.sort_index().copy()
        open_ = _to_float_series(df, "open")
        high = _to_float_series(df, "high")
        low = _to_float_series(df, "low")
        close = _to_float_series(df, "close")
        volume = _to_float_series(df, "volume")

        upper, middle, lower = _bollinger(close, window=boll_window, n_std=boll_n_std)
        rsi = _rsi_wilder(close, period=rsi_period)
        adx = _adx_wilder(high, low, close, period=adx_period)
        ci = _choppiness_index(high, low, close, period=ci_period)

        bandwidth = (upper - lower) / middle.replace(0.0, np.nan)
        bw_prank = _rolling_percent_rank_last(bandwidth, window=bw_prank_window)

        bw_prev = bandwidth.shift(1)
        breakout_today = (bw_prev > 0) & ((bandwidth - bw_prev) / bw_prev >= bw_jump_stop)

        vol_ma = volume.rolling(window=volume_ma_window, min_periods=volume_ma_window).mean()
        volume_ok = volume > vol_ma

        ma200 = close.rolling(window=200, min_periods=200).mean()
        trend_ok = (close > ma200) if require_above_ma200 else pd.Series(True, index=df.index)

        adx_ok = adx < adx_max
        if require_adx_falling:
            adx_ok = adx_ok & (adx < adx.shift(1))

        ci_ok = ci > ci_min
        bw_ok = bw_prank <= bw_prank_max

        st_ok = pd.Series(True, index=df.index)
        if avoid_st and "is_st" in df.columns:
            st_ok = _to_float_series(df, "is_st").fillna(0.0) != 1.0

        trade_ok = pd.Series(True, index=df.index)
        if avoid_suspended and "tradestatus" in df.columns:
            trade_ok = _to_float_series(df, "tradestatus").fillna(1.0) == 1.0

        regime_ok = (adx_ok & ci_ok & bw_ok & volume_ok & trend_ok & st_ok & trade_ok).fillna(False)

        knife_ok = pd.Series(True, index=df.index)
        if enable_falling_knife_filter:
            body = (close - open_).abs()
            candle_low_body = pd.concat([open_, close], axis=1).min(axis=1)
            lower_shadow = (candle_low_body - low).clip(lower=0.0)
            # body == 0 => not allowed (avoid random entry)
            knife_bad = (body <= 0.0) | ((close < open_) & (lower_shadow <= lower_shadow_body_ratio_max * body))
            knife_ok = (~knife_bad).fillna(False)

        entry_cond = (
            regime_ok
            & knife_ok
            & (close <= lower)
            & (rsi <= rsi_entry)
            & (~breakout_today)
        ).fillna(False)

        # 目标仓位信号（当天收盘决定 -> 次日执行由回测层 shift(1) 完成）
        signal = pd.Series(0, index=df.index, dtype="int8")

        pos_today = 0
        hold_days = 0
        band_walk_count = 0
        cooldown = 0

        for i in range(len(df)):
            c = float(close.iat[i]) if i < len(close) else np.nan
            m = float(middle.iat[i]) if i < len(middle) else np.nan
            l = float(lower.iat[i]) if i < len(lower) else np.nan
            breakout = bool(breakout_today.iat[i]) if pd.notna(breakout_today.iat[i]) else False

            next_pos = pos_today

            if pos_today == 1:
                hold_days += 1

                if _is_valid_price(c) and np.isfinite(l):
                    if c < l:
                        band_walk_count += 1
                    else:
                        band_walk_count = 0

                exit_now = False
                if _is_valid_price(c) and np.isfinite(m) and c >= m:
                    exit_now = True
                if hold_days >= max_hold_days:
                    exit_now = True
                if band_walk_days > 0 and band_walk_count >= band_walk_days:
                    exit_now = True
                if breakout:
                    exit_now = True

                if exit_now:
                    next_pos = 0
                    hold_days = 0
                    band_walk_count = 0

            else:
                hold_days = 0
                band_walk_count = 0

                if cooldown == 0 and bool(entry_cond.iat[i]):
                    next_pos = 1

            signal.iat[i] = int(next_pos)

            if breakout:
                cooldown = max(0, cooldown_days_after_breakout)
            elif cooldown > 0:
                cooldown -= 1

            pos_today = next_pos

        signal.name = "signal"
        return signal.astype(int)
