import unittest


from backend.tft_model.predict import baseline_price_forecast


class TestBaselinePriceForecast(unittest.TestCase):
    def test_business_days_and_nonflat_for_neutral(self) -> None:
        history = [
            {"time": "2025-12-01", "close": 60.00, "ma20": 60.20},
            {"time": "2025-12-02", "close": 60.30, "ma20": 60.25},
            {"time": "2025-12-03", "close": 60.10, "ma20": 60.24},
            {"time": "2025-12-04", "close": 60.50, "ma20": 60.30},
            {"time": "2025-12-05", "close": 60.40, "ma20": 60.33},
            {"time": "2025-12-08", "close": 60.80, "ma20": 60.40},
            {"time": "2025-12-09", "close": 61.00, "ma20": 60.45},
            {"time": "2025-12-10", "close": 60.90, "ma20": 60.50},
            {"time": "2025-12-11", "close": 61.20, "ma20": 60.60},
            {"time": "2025-12-12", "close": 62.10, "ma20": 60.70},
        ]

        points = baseline_price_forecast(
            last_date="2025-12-12",
            last_close=62.10,
            history=history,
            signal="中性",
            confidence=0.56,
            horizon=5,
        )

        self.assertEqual(len(points), 5)
        dates = [p["date"] for p in points]
        # 2025-12-12 is Friday, next business day is Monday.
        self.assertEqual(dates[0], "2025-12-15")
        self.assertEqual(dates[-1], "2025-12-19")

        closes = [round(p["predicted_close"], 4) for p in points]
        self.assertGreater(len(set(closes)), 1)

        band1 = points[0]["upper"] - points[0]["lower"]
        band5 = points[-1]["upper"] - points[-1]["lower"]
        self.assertGreaterEqual(band5, band1)

    def test_signal_tilt_moves_buy_vs_sell(self) -> None:
        history = [
            {"time": "2025-12-01", "close": 100.0, "ma20": 100.0},
            {"time": "2025-12-02", "close": 100.1, "ma20": 100.0},
            {"time": "2025-12-03", "close": 99.9, "ma20": 100.0},
            {"time": "2025-12-04", "close": 100.0, "ma20": 100.0},
            {"time": "2025-12-05", "close": 100.1, "ma20": 100.0},
            {"time": "2025-12-08", "close": 99.9, "ma20": 100.0},
            {"time": "2025-12-09", "close": 100.0, "ma20": 100.0},
            {"time": "2025-12-10", "close": 100.1, "ma20": 100.0},
            {"time": "2025-12-11", "close": 99.9, "ma20": 100.0},
            {"time": "2025-12-12", "close": 100.0, "ma20": 100.0},
        ]

        buy = baseline_price_forecast(
            last_date="2025-12-12",
            last_close=100.0,
            history=history,
            signal="买入",
            confidence=0.8,
            horizon=5,
        )
        sell = baseline_price_forecast(
            last_date="2025-12-12",
            last_close=100.0,
            history=history,
            signal="卖出",
            confidence=0.8,
            horizon=5,
        )

        self.assertGreater(buy[-1]["predicted_close"], sell[-1]["predicted_close"])


if __name__ == "__main__":
    unittest.main()

