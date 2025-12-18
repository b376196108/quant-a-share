"""
Extreme Buy-Sell Strategy (极限买卖策略)

合并逻辑：
  - 极限抄底入场：BOLL 下轨/RSI/MFI/KDJ(J) 评分 + 触发条件（三选一）
  - 顶部卖出：空间极值(A) + 动量衰竭(B) + 形态/量能确认(C) 三元共振
  - 中轨生命线：close < BOLL mid 强制退出
  - 风控：结构止损、时间止损、卖出后冷却期

仅使用 BaoStock 日线 OHLCV（open/high/low/close/volume/amount），不依赖 pandas_ta。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..base_strategy import BaseStrategy, StrategyContext
from ..registry import StrategyMeta, register_strategy


# --------- 指标实现（纯 pandas/numpy）---------


def sma(series: pd.Series, n: int) -> pd.Series:
    n = int(n)
    if n <= 0:
        raise ValueError("SMA window must be positive")
    return series.rolling(window=n, min_periods=n).mean()


def boll(
    close: pd.Series,
    n: int = 20,
    k: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    n = int(n)
    k = float(k)
    mid = sma(close, n)
    std = close.rolling(window=n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    bw = (upper - lower) / mid.replace(0.0, np.nan)
    bw = bw.fillna(0.0)
    return mid, upper, lower, bw


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    period = int(period)
    if period <= 0:
        raise ValueError("RSI period must be positive")

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def bias(close: pd.Series, n: int) -> pd.Series:
    n = int(n)
    ma = sma(close, n)
    denom = ma.replace(0.0, np.nan)
    out = (close - ma) / denom * 100.0
    return out.fillna(0.0)


def compute_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    n = int(n)
    k_smooth = int(k_smooth)
    d_smooth = int(d_smooth)
    if n <= 0:
        raise ValueError("KDJ n must be positive")
    if k_smooth <= 0 or d_smooth <= 0:
        raise ValueError("KDJ smooth must be positive")

    llv = low.rolling(window=n).min()
    hhv = high.rolling(window=n).max()
    denom = hhv - llv

    rsv = ((close - llv) / denom) * 100.0
    rsv = rsv.where((denom != 0) & denom.notna(), 50.0).fillna(50.0)

    k_arr = np.full(len(close), 50.0, dtype=float)
    d_arr = np.full(len(close), 50.0, dtype=float)

    k_prev = 50.0
    d_prev = 50.0
    rsv_values = rsv.to_numpy(dtype=float, copy=False)

    for i in range(len(close)):
        k_now = (k_prev * (k_smooth - 1) + rsv_values[i]) / k_smooth
        d_now = (d_prev * (d_smooth - 1) + k_now) / d_smooth
        k_arr[i] = k_now
        d_arr[i] = d_now
        k_prev = k_now
        d_prev = d_now

    k_ser = pd.Series(k_arr, index=close.index).fillna(50.0)
    d_ser = pd.Series(d_arr, index=close.index).fillna(50.0)
    j_ser = (3.0 * k_ser - 2.0 * d_ser).fillna(50.0)
    return k_ser, d_ser, j_ser


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    n: int = 14,
) -> pd.Series:
    n = int(n)
    if n <= 0:
        raise ValueError("MFI n must be positive")

    tp = (high + low + close) / 3.0
    mf = tp * volume

    tp_prev = tp.shift(1)
    pmf = mf.where(tp > tp_prev, 0.0)
    nmf = mf.where(tp < tp_prev, 0.0)

    pmf_n = pmf.rolling(window=n, min_periods=n).sum()
    nmf_n = nmf.rolling(window=n, min_periods=n).sum()

    mfr = pmf_n / nmf_n.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + mfr)
    return mfi.fillna(50.0)


def shooting_star(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    rng = high - low
    body = (close - open_).abs()
    upper = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1) - low

    body_safe = body.replace(0.0, np.nan)
    cond = (
        (rng > 0)
        & ((body / rng) <= 0.3)
        & ((upper / body_safe) >= 2.0)
        & ((lower / body_safe) <= 0.5)
    )
    return cond.fillna(False)


def dark_cloud_cover(open_: pd.Series, close: pd.Series) -> pd.Series:
    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    prev_bull = prev_close > prev_open
    today_open_high = open_ > prev_close
    today_close_into = (close < (prev_open + prev_close) / 2.0) & (close > prev_open)

    cond = prev_bull & today_open_high & today_close_into
    return cond.fillna(False)


def bearish_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    prev_bull = prev_close > prev_open
    today_bear = close < open_
    engulf = (open_ >= prev_close) & (close <= prev_open)
    return (prev_bull & today_bear & engulf).fillna(False)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    n = int(n)
    if n <= 0:
        raise ValueError("ATR n must be positive")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=n, min_periods=n).mean()
    return atr.fillna(tr)


# --------- 策略元信息 ---------


META = StrategyMeta(
    id="extreme_buy_sell",
    name="极限买卖策略",
    category="reversal",
    description=(
        "极限抄底入场（BOLL/RSI/MFI/KDJ 评分 + 触发），"
        "顶部三元共振出场（空间极值+动量衰竭+形态/量能），"
        "中轨生命线强制退出，并带结构止损/时间止损/冷却期。"
    ),
    tags=["mean_reversion", "extreme", "rsi", "mfi", "kdj", "boll", "bias"],
    default_params={
        "boll_n": 20,
        "boll_k": 2.0,
        "boll_tol": 0.01,
        "rsi_n": 14,
        "rsi_buy_th": 30.0,
        "rsi_sell_th": 70.0,
        "mfi_n": 14,
        "mfi_buy_th": 20.0,
        "mfi_buffer": 5.0,
        "kdj_n": 9,
        "k_smooth": 3,
        "d_smooth": 3,
        "j_buy_th": 10.0,
        "j_extreme": 100.0,
        "j_buffer": 10.0,
        "use_trend_filter": True,
        "trend_n": 60,
        "score_min": 3,
        "score_min_up": 2,
        "entry_score_min": 2,
        "entry_score_min_up": 1,
        "entry_window": 3,
        "bias5_sell": 10.0,
        "bias10_extreme": 15.0,
        "resonance_window": 2,
        "sell_score_min": 2,
        "peak_window": 20,
        "peak_tol": 0.01,
        "kdj_overbought": 80.0,
        "atr_n": 14,
        "trail_start_profit": 0.05,
        "trail_atr_mult": 3.0,
        "vol_window": 20,
        "vol_spike": 2.0,
        "vol_shrink": 0.7,
        "life_line_grace_days": 5,
        "stop_lookback": 20,
        "stop_buffer": 0.0,
        "max_hold_days": 60,
        "cooldown_days": 3,
    },
    param_schema={
        "boll_n": {"type": "int", "min": 5, "max": 60, "step": 1},
        "boll_k": {"type": "float", "min": 1.0, "max": 4.0, "step": 0.1},
        "boll_tol": {"type": "float", "min": 0.0, "max": 0.05, "step": 0.001},
        "rsi_n": {"type": "int", "min": 2, "max": 50, "step": 1},
        "rsi_buy_th": {"type": "float", "min": 1.0, "max": 50.0, "step": 0.5},
        "rsi_sell_th": {"type": "float", "min": 50.0, "max": 99.0, "step": 0.5},
        "mfi_n": {"type": "int", "min": 2, "max": 50, "step": 1},
        "mfi_buy_th": {"type": "float", "min": 1.0, "max": 50.0, "step": 1.0},
        "mfi_buffer": {"type": "float", "min": 0.0, "max": 30.0, "step": 1.0},
        "kdj_n": {"type": "int", "min": 3, "max": 30, "step": 1},
        "k_smooth": {"type": "int", "min": 2, "max": 10, "step": 1},
        "d_smooth": {"type": "int", "min": 2, "max": 10, "step": 1},
        "j_buy_th": {"type": "float", "min": -20.0, "max": 50.0, "step": 1.0},
        "j_extreme": {"type": "float", "min": 70.0, "max": 150.0, "step": 1.0},
        "j_buffer": {"type": "float", "min": 0.0, "max": 50.0, "step": 1.0},
        "use_trend_filter": {"type": "bool"},
        "trend_n": {"type": "int", "min": 0, "max": 250, "step": 5},
        "score_min": {"type": "int", "min": 1, "max": 4, "step": 1},
        "score_min_up": {"type": "int", "min": 1, "max": 4, "step": 1},
        "entry_score_min": {"type": "int", "min": 0, "max": 4, "step": 1},
        "entry_score_min_up": {"type": "int", "min": 0, "max": 4, "step": 1},
        "entry_window": {"type": "int", "min": 1, "max": 10, "step": 1},
        "bias5_sell": {"type": "float", "min": 0.0, "max": 30.0, "step": 0.5},
        "bias10_extreme": {"type": "float", "min": 0.0, "max": 50.0, "step": 0.5},
        "resonance_window": {"type": "int", "min": 1, "max": 10, "step": 1},
        "sell_score_min": {"type": "int", "min": 1, "max": 3, "step": 1},
        "peak_window": {"type": "int", "min": 5, "max": 120, "step": 1},
        "peak_tol": {"type": "float", "min": 0.0, "max": 0.05, "step": 0.001},
        "kdj_overbought": {"type": "float", "min": 50.0, "max": 100.0, "step": 1.0},
        "atr_n": {"type": "int", "min": 2, "max": 50, "step": 1},
        "trail_start_profit": {"type": "float", "min": 0.0, "max": 0.5, "step": 0.01},
        "trail_atr_mult": {"type": "float", "min": 0.5, "max": 10.0, "step": 0.1},
        "vol_window": {"type": "int", "min": 5, "max": 60, "step": 1},
        "vol_spike": {"type": "float", "min": 1.0, "max": 5.0, "step": 0.1},
        "vol_shrink": {"type": "float", "min": 0.1, "max": 1.0, "step": 0.05},
        "life_line_grace_days": {"type": "int", "min": 0, "max": 30, "step": 1},
        "stop_lookback": {"type": "int", "min": 5, "max": 120, "step": 1},
        "stop_buffer": {"type": "float", "min": 0.0, "max": 0.2, "step": 0.005},
        "max_hold_days": {"type": "int", "min": 0, "max": 200, "step": 1},
        "cooldown_days": {"type": "int", "min": 0, "max": 30, "step": 1},
    },
)


@register_strategy(META)
class ExtremeBuySellStrategy(BaseStrategy):
    """
    极限买卖策略（Extreme Buy-Sell），只做多，输出 0/1 持仓序列（状态机）。

    约定：
      - 输入 df 至少包含 open/high/low/close/volume 列；
      - 指标在策略内部计算，不依赖外部指标引擎；
      - 输出 signal 值域 {-1,0,1}，本策略仅输出 0/1。
    """

    name = "extreme_buy_sell"

    def required_indicators(self) -> list[str]:
        return []

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Optional[StrategyContext] = None,
    ) -> pd.Series:
        params: Dict[str, Any] = {**META.default_params, **(self.params or {})}

        boll_n = int(params["boll_n"])
        boll_k = float(params["boll_k"])
        boll_tol = float(params["boll_tol"])

        rsi_n = int(params["rsi_n"])
        rsi_buy_th = float(params["rsi_buy_th"])
        rsi_sell_th = float(params["rsi_sell_th"])

        mfi_n = int(params["mfi_n"])
        mfi_buy_th = float(params["mfi_buy_th"])
        mfi_buffer = float(params["mfi_buffer"])

        kdj_n = int(params["kdj_n"])
        k_smooth = int(params["k_smooth"])
        d_smooth = int(params["d_smooth"])
        j_buy_th = float(params["j_buy_th"])
        j_extreme = float(params["j_extreme"])
        j_buffer = float(params["j_buffer"])

        use_trend_filter = bool(params["use_trend_filter"])
        trend_n = int(params["trend_n"])
        trend_n = max(0, trend_n)

        score_min = int(params["score_min"])
        score_min = int(min(4, max(1, score_min)))
        score_min_up = int(params["score_min_up"])
        score_min_up = int(min(4, max(1, score_min_up)))

        entry_score_min = int(params["entry_score_min"])
        entry_score_min = int(min(4, max(0, entry_score_min)))
        entry_score_min_up = int(params["entry_score_min_up"])
        entry_score_min_up = int(min(4, max(0, entry_score_min_up)))
        entry_window = max(1, int(params["entry_window"]))

        bias5_sell = float(params["bias5_sell"])
        bias10_extreme = float(params["bias10_extreme"])

        resonance_window = max(1, int(params["resonance_window"]))
        sell_score_min = int(params["sell_score_min"])
        sell_score_min = int(min(3, max(1, sell_score_min)))

        peak_window = int(params["peak_window"])
        peak_window = max(2, peak_window)
        peak_tol = float(params["peak_tol"])
        peak_tol = max(0.0, peak_tol)
        kdj_overbought = float(params["kdj_overbought"])

        atr_n = int(params["atr_n"])
        atr_n = max(1, atr_n)
        trail_start_profit = float(params["trail_start_profit"])
        trail_atr_mult = float(params["trail_atr_mult"])

        vol_window = int(params["vol_window"])
        vol_spike = float(params["vol_spike"])
        vol_shrink = float(params["vol_shrink"])

        life_line_grace_days = max(0, int(params["life_line_grace_days"]))
        stop_lookback = max(1, int(params["stop_lookback"]))
        stop_buffer = float(params["stop_buffer"])

        max_hold_days = int(params["max_hold_days"])
        cooldown_days = max(0, int(params["cooldown_days"]))

        df = df.sort_index().copy()
        open_ = pd.to_numeric(df["open"], errors="coerce").astype(float)
        high = pd.to_numeric(df["high"], errors="coerce").astype(float)
        low = pd.to_numeric(df["low"], errors="coerce").astype(float)
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)

        mid, upper, lower, _bw = boll(close, n=boll_n, k=boll_k)
        rsi = compute_rsi(close, period=rsi_n)
        bias5 = bias(close, 5)
        bias10 = bias(close, 10)
        k_val, d_val, j_val = compute_kdj(
            high=high, low=low, close=close, n=kdj_n, k_smooth=k_smooth, d_smooth=d_smooth
        )
        mfi = compute_mfi(high=high, low=low, close=close, volume=volume, n=mfi_n)

        # Trend regime (optional): relax entry thresholds in up-trends
        if use_trend_filter and trend_n > 0:
            trend_ma = sma(close, trend_n)
            up_trend = (close >= trend_ma).fillna(False)
            score_min_eff = pd.Series(
                np.where(up_trend.to_numpy(), score_min_up, score_min),
                index=df.index,
            )
            entry_score_min_eff = pd.Series(
                np.where(up_trend.to_numpy(), entry_score_min_up, entry_score_min),
                index=df.index,
            )
        else:
            up_trend = pd.Series(False, index=df.index)
            score_min_eff = pd.Series(score_min, index=df.index)
            entry_score_min_eff = pd.Series(entry_score_min, index=df.index)

        atr = compute_atr(high=high, low=low, close=close, n=atr_n)

        # --------- 入场（极限抄底）---------
        s1 = (low <= lower * (1.0 + boll_tol)) | (close <= lower * (1.0 + boll_tol))
        s2 = rsi <= rsi_buy_th
        s3 = mfi <= mfi_buy_th
        s4 = j_val <= j_buy_th

        oversold_score = (
            s1.fillna(False).astype(int)
            + s2.fillna(False).astype(int)
            + s3.fillna(False).astype(int)
            + s4.fillna(False).astype(int)
        )

        t1 = (close > open_) & (close >= lower)
        k_cross_up = (k_val > d_val) & (k_val.shift(1) <= d_val.shift(1))
        t2 = k_cross_up & (j_val <= (j_buy_th + j_buffer))
        t3 = (mfi > mfi.shift(1)) & (mfi <= (mfi_buy_th + mfi_buffer))

        oversold_hit = (oversold_score >= score_min_eff).fillna(False)
        oversold_recent = (
            oversold_hit.astype(int)
            .rolling(window=entry_window, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )
        entry_score_ok = (oversold_score >= entry_score_min_eff).fillna(False)
        entry_cond = oversold_recent & entry_score_ok & (t1 | t2 | t3)
        entry_cond = entry_cond.fillna(False)

        # --------- 出场（三元共振，忽略北向资金）---------
        a_boll_revert = (
            (close.shift(1) > upper.shift(1))
            & (close <= upper)
            & (upper.diff() < 0)
            & ((upper - lower).diff() < 0)
        )
        a_touch_upper = (high >= upper) | (close >= upper)
        rolling_high = high.rolling(window=peak_window, min_periods=peak_window).max()
        a_near_high = high >= (rolling_high * (1.0 - peak_tol))
        dim_a = (bias5 > bias5_sell) | (bias10 > bias10_extreme) | a_boll_revert | a_touch_upper | a_near_high.fillna(False)

        b_kdj_exhaust = (
            (j_val.shift(1) > j_extreme)
            & (k_val.shift(1) >= d_val.shift(1))
            & (k_val < d_val)
        )
        kdj_dead_cross = (k_val < d_val) & (k_val.shift(1) >= d_val.shift(1))
        kdj_prev_max = np.maximum(k_val.shift(1), d_val.shift(1))
        b_kdj_overbought = kdj_dead_cross & (kdj_prev_max >= kdj_overbought)
        b_rsi_exhaust = (rsi.shift(1) >= rsi_sell_th) & (rsi < rsi.shift(1))
        dim_b = b_kdj_exhaust | b_kdj_overbought.fillna(False) | b_rsi_exhaust

        pat_shooting_star = shooting_star(open_=open_, high=high, low=low, close=close)
        pat_dark_cloud = dark_cloud_cover(open_=open_, close=close)
        pat_bear_engulf = bearish_engulfing(open_=open_, close=close)

        vol_ma = volume.rolling(window=vol_window).mean()
        big_bear = (volume > vol_ma * vol_spike) & (close < open_)

        nh = close >= close.rolling(window=vol_window).max()
        shrink_new_high = (volume < vol_ma * vol_shrink) & nh

        dim_c = (
            pat_shooting_star
            | pat_dark_cloud
            | pat_bear_engulf
            | big_bear.fillna(False)
            | shrink_new_high.fillna(False)
        )

        a2 = dim_a.astype(int).rolling(window=resonance_window).max().fillna(0).astype(bool)
        b2 = dim_b.astype(int).rolling(window=resonance_window).max().fillna(0).astype(bool)
        c2 = dim_c.astype(int).rolling(window=resonance_window).max().fillna(0).astype(bool)
        sell_score = a2.astype(int) + b2.astype(int) + c2.astype(int)
        sell_resonance = (sell_score >= sell_score_min).fillna(False)

        # --------- 生命线（强制退出）---------
        life_line_raw = (close < mid).fillna(False)

        # --------- 状态机输出持仓序列（0/1）---------
        signal = pd.Series(0, index=df.index, dtype="int8")
        position = 0
        entry_i: Optional[int] = None
        entry_price = np.nan
        peak_close = np.nan
        stop_ref = np.nan
        cooldown_counter = 0
        mid_reclaimed = False

        for i in range(len(df)):
            if position == 0:
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                else:
                    if bool(entry_cond.iat[i]):
                        position = 1
                        entry_i = i
                        entry_price = float(close.iat[i])
                        peak_close = entry_price
                        lb = max(0, i - stop_lookback + 1)
                        stop_ref = float(low.iloc[lb : i + 1].min())
                        mid_i = mid.iat[i]
                        mid_reclaimed = bool(not np.isnan(mid_i) and entry_price >= float(mid_i))
            else:
                bars_since_entry = (i - entry_i) if entry_i is not None else 0
                hold_days = bars_since_entry + 1

                close_i = float(close.iat[i])
                if not np.isnan(close_i) and close_i > 0:
                    if np.isnan(peak_close) or close_i > peak_close:
                        peak_close = close_i

                mid_i = mid.iat[i]
                if not np.isnan(mid_i) and bool(close.iat[i] >= mid_i):
                    mid_reclaimed = True

                stop_loss = False
                if not np.isnan(stop_ref):
                    stop_loss = bool(close.iat[i] < stop_ref * (1.0 - stop_buffer))

                time_stop = bool(max_hold_days > 0 and hold_days >= max_hold_days)

                life_line_hit = False
                if not np.isnan(mid_i) and bool(life_line_raw.iat[i]):
                    if mid_reclaimed:
                        life_line_hit = True
                    elif life_line_grace_days > 0 and bars_since_entry >= life_line_grace_days:
                        life_line_hit = True

                trailing_exit = False
                if (
                    trail_start_profit > 0
                    and trail_atr_mult > 0
                    and not np.isnan(entry_price)
                    and entry_price > 0
                    and not np.isnan(peak_close)
                    and peak_close > 0
                ):
                    profit_peak = peak_close / entry_price - 1.0
                    if profit_peak >= trail_start_profit:
                        atr_i = float(atr.iat[i])
                        if not np.isnan(atr_i) and atr_i > 0:
                            trail_level = peak_close - trail_atr_mult * atr_i
                            if close_i <= trail_level:
                                trailing_exit = True

                exit_now = bool(sell_resonance.iat[i]) or life_line_hit or trailing_exit or stop_loss or time_stop
                if exit_now:
                    position = 0
                    entry_i = None
                    entry_price = np.nan
                    peak_close = np.nan
                    stop_ref = np.nan
                    cooldown_counter = cooldown_days

            signal.iat[i] = position

        signal.name = "signal"
        return signal
