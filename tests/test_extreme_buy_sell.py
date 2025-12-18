import unittest

import numpy as np
import pandas as pd

from quant_system.strategy.plugins.extreme_buy_sell import ExtremeBuySellStrategy


def make_ohlcv_df(
    close_values: list[float],
    open_values: list[float],
    volume_values: list[float] | None = None,
) -> pd.DataFrame:
    if len(close_values) != len(open_values):
        raise ValueError("close_values and open_values length mismatch")
    n = len(close_values)
    volume = volume_values if volume_values is not None else [1000.0] * n
    if len(volume) != n:
        raise ValueError("volume length mismatch")

    close = np.asarray(close_values, dtype=float)
    open_ = np.asarray(open_values, dtype=float)
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume_arr = np.asarray(volume, dtype=float)
    amount = close * volume_arr

    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume_arr,
            "amount": amount,
        },
        index=idx,
    )


class TestExtremeBuySellStrategy(unittest.TestCase):
    def test_entry_window_allows_lagged_trigger(self) -> None:
        # Construct a sharp downtrend (oversold_hit=True), then a rebound day where
        # RSI/J recover so oversold_score drops below score_min, but the trigger fires.
        down = list(range(100, 85, -1))  # 100..86
        close_values = [float(v) for v in down] + [88.0]  # rebound day
        open_values = [c + 0.5 for c in close_values[:-1]] + [86.0]
        df = make_ohlcv_df(close_values, open_values)

        base = {
            "boll_n": 5,
            "boll_tol": 0.05,
            "rsi_n": 3,
            "rsi_buy_th": 45.0,
            "mfi_n": 3,
            "mfi_buy_th": 45.0,
            "kdj_n": 3,
            "j_buy_th": 35.0,
            "score_min": 3,
            "max_hold_days": 0,
            "cooldown_days": 0,
        }

        sig_w1 = ExtremeBuySellStrategy(params={**base, "entry_window": 1}).generate_signals(df)
        sig_w2 = ExtremeBuySellStrategy(params={**base, "entry_window": 2}).generate_signals(df)

        self.assertEqual(int(sig_w1.iloc[-1]), 0, "entry_window=1 should not enter on rebound day")
        self.assertEqual(int(sig_w2.iloc[-1]), 1, "entry_window=2 should enter on rebound day")

    def test_entry_on_oversold_reclaim(self) -> None:
        down = list(range(100, 85, -1))  # 100..86
        close_values = [float(v) for v in down] + [88.0]  # reversal day
        open_values = [c + 0.5 for c in close_values[:-1]] + [86.0]
        df = make_ohlcv_df(close_values, open_values)

        strat = ExtremeBuySellStrategy(
            params={
                "boll_n": 5,
                "boll_tol": 0.05,
                "rsi_n": 3,
                "rsi_buy_th": 60.0,
                "mfi_n": 3,
                "mfi_buy_th": 45.0,
                "kdj_n": 3,
                "j_buy_th": 80.0,
                "score_min": 3,
                "max_hold_days": 0,
                "cooldown_days": 0,
            }
        )
        sig = strat.generate_signals(df)

        entries = (sig.eq(1) & sig.shift(1).fillna(0).eq(0))
        self.assertTrue(entries.any(), "expected at least one entry (0->1)")

        first_entry_pos = int(np.flatnonzero(entries.to_numpy())[0])
        self.assertEqual(int(sig.iloc[first_entry_pos]), 1)
        self.assertTrue((sig.iloc[:first_entry_pos] == 0).all())

    def test_exit_on_sell_resonance(self) -> None:
        down = list(range(100, 85, -1))  # 100..86
        reversal = [88.0]
        up = [90.0 + 2.0 * i for i in range(10)]  # 90..108
        peak = [110.0]
        big_bear = [108.0]
        tail = [107.0, 106.0, 105.0]
        close_values = [float(v) for v in down] + reversal + up + peak + big_bear + tail

        open_values: list[float] = []
        for i, c in enumerate(close_values):
            if i < len(down):
                open_values.append(c + 0.5)  # bearish downtrend
            elif i == len(down):  # reversal day
                open_values.append(86.0)
            elif i < len(down) + len(reversal) + len(up) + len(peak):
                open_values.append(c - 0.5)  # bullish run-up
            elif i == len(down) + len(reversal) + len(up) + len(peak):  # big bear
                open_values.append(112.0)
            else:
                open_values.append(c + 0.5)

        volume_values = [1000.0] * len(close_values)
        big_bear_pos = len(down) + len(reversal) + len(up) + len(peak)
        volume_values[big_bear_pos] = 5000.0

        df = make_ohlcv_df(close_values, open_values, volume_values)

        strat = ExtremeBuySellStrategy(
            params={
                "boll_n": 5,
                "rsi_n": 3,
                "rsi_buy_th": 60.0,
                "rsi_sell_th": 60.0,
                "mfi_n": 3,
                "mfi_buy_th": 45.0,
                "kdj_n": 3,
                "j_buy_th": 80.0,
                "j_extreme": 80.0,
                "score_min": 3,
                "bias5_sell": 1.0,
                "bias10_extreme": 1.5,
                "resonance_window": 2,
                "vol_window": 5,
                "vol_spike": 2.0,
                "max_hold_days": 0,
                "cooldown_days": 0,
            }
        )
        sig = strat.generate_signals(df)

        entries = (sig.eq(1) & sig.shift(1).fillna(0).eq(0))
        exits = (sig.eq(0) & sig.shift(1).fillna(0).eq(1))
        self.assertTrue(entries.any(), "expected an entry before exit test")
        self.assertTrue(exits.any(), "expected at least one exit (1->0)")

        first_entry_pos = int(np.flatnonzero(entries.to_numpy())[0])
        first_exit_pos = int(np.flatnonzero(exits.to_numpy())[0])
        self.assertGreater(first_exit_pos, first_entry_pos)

        self.assertEqual(int(sig.iloc[big_bear_pos]), 0, "expected exit on big bear resonance day")

    def test_trailing_exit_after_profit_drawdown(self) -> None:
        # Make sure ATR-based trailing exit can close the position after a strong rebound
        # and a mild pullback (without relying on sell_resonance / life_line).
        close_values = [100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 92.0, 95.0, 99.0, 105.0, 102.4, 104.0]
        open_values: list[float] = []
        for i, c in enumerate(close_values):
            if i <= 5:  # downtrend
                open_values.append(c + 0.5)
            elif i == 6:  # bullish reversal/entry day
                open_values.append(90.0)
            elif i == 10:  # pullback day (still above mid), to trigger trailing stop
                open_values.append(c + 1.0)
            else:
                open_values.append(c - 0.5)

        df = make_ohlcv_df(close_values, open_values)

        strat = ExtremeBuySellStrategy(
            params={
                "use_trend_filter": False,
                "boll_n": 3,
                "boll_tol": 0.05,
                "rsi_n": 3,
                "rsi_buy_th": 65.0,
                "rsi_sell_th": 100.0,
                "mfi_n": 3,
                "mfi_buy_th": 65.0,
                "kdj_n": 3,
                "j_buy_th": 65.0,
                "j_extreme": 200.0,
                "kdj_overbought": 200.0,
                "score_min": 3,
                "entry_score_min": 2,
                "entry_window": 2,
                "bias5_sell": 999.0,
                "bias10_extreme": 999.0,
                "sell_score_min": 3,
                "resonance_window": 1,
                "vol_window": 3,
                "vol_spike": 100.0,
                "vol_shrink": 0.0,
                "life_line_grace_days": 0,
                "max_hold_days": 0,
                "cooldown_days": 0,
                "atr_n": 3,
                "trail_start_profit": 0.02,
                "trail_atr_mult": 0.5,
                "peak_window": 3,
                "peak_tol": 0.0,
            }
        )
        sig = strat.generate_signals(df)

        entries = (sig.eq(1) & sig.shift(1).fillna(0).eq(0))
        exits = (sig.eq(0) & sig.shift(1).fillna(0).eq(1))
        self.assertTrue(entries.any(), "expected an entry for trailing-stop test")
        self.assertTrue(exits.any(), "expected an exit for trailing-stop test")

        # Exit should happen on the pullback day (index 10) due to trailing stop.
        self.assertEqual(int(sig.iloc[10]), 0)

    def test_cooldown_blocks_reentry(self) -> None:
        close_values = [
            100.0,
            98.0,
            96.0,
            94.0,
            92.0,
            90.0,
            92.0,  # entry (bullish)
            85.0,  # exit (life line)
            88.0,  # would re-enter if no cooldown (bullish)
            87.0,
            89.0,  # re-enter after cooldown
            90.0,
        ]
        open_values = [
            100.5,
            98.5,
            96.5,
            94.5,
            92.5,
            90.5,
            90.0,
            93.0,
            86.0,
            87.5,
            88.0,
            89.5,
        ]
        df = make_ohlcv_df(close_values, open_values)

        base_params = {
            "boll_n": 3,
            "boll_tol": 0.05,
            "rsi_n": 3,
            "rsi_buy_th": 65.0,
            "mfi_n": 3,
            "mfi_buy_th": 65.0,
            "kdj_n": 3,
            "j_buy_th": 65.0,
            "score_min": 3,
            "max_hold_days": 0,
        }

        sig_no_cd = ExtremeBuySellStrategy(params={**base_params, "cooldown_days": 0}).generate_signals(df)
        sig_cd = ExtremeBuySellStrategy(params={**base_params, "cooldown_days": 2}).generate_signals(df)

        # Ensure: first entry happens on day 6 (index position 6)
        self.assertEqual(int(sig_no_cd.iloc[6]), 1)
        self.assertEqual(int(sig_cd.iloc[6]), 1)

        # Ensure: exit happens on day 7 (life line)
        self.assertEqual(int(sig_no_cd.iloc[7]), 0)
        self.assertEqual(int(sig_cd.iloc[7]), 0)

        # Cooldown should block re-entry on day 8 and 9, while no-cooldown can re-enter on day 8.
        self.assertEqual(int(sig_no_cd.iloc[8]), 1)
        self.assertEqual(int(sig_cd.iloc[8]), 0)
        self.assertEqual(int(sig_cd.iloc[9]), 0)


if __name__ == "__main__":
    unittest.main()
