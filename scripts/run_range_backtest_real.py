from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def normalize_symbol(code: str) -> str:
    """
    Convert 6-digit A-share code to baostock style:
      600xxx / 601xxx / 603xxx / 688xxx / 9xxxxx -> sh.XXXXXX
      others -> sz.XXXXXX
    If already like sh.XXXXXX / sz.XXXXXX, return as-is.
    """
    code = code.strip()
    if len(code) == 9 and "." in code:
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        return f"sz.{code}"
    return code


def _symbol_tag(symbol_full: str) -> str:
    sym = symbol_full.strip()
    if "." in sym:
        tail = sym.split(".")[-1]
        return tail if tail else sym.replace(".", "_")
    return sym


def _compute_win_rate(trades_df: pd.DataFrame) -> Tuple[int, float]:
    if trades_df is None or trades_df.empty:
        return 0, 0.0

    win_count = 0
    loss_count = 0
    last_buy_price = None
    last_buy_shares = 0
    last_buy_fee = 0.0

    for _, row in trades_df.iterrows():
        action = str(row.get("action") or "").lower()
        if action == "buy":
            last_buy_price = float(row.get("price", 0) or 0)
            last_buy_shares = int(row.get("shares", 0) or 0)
            last_buy_fee = float(row.get("fee", 0) or 0)
            continue

        if action == "sell" and last_buy_price is not None and last_buy_shares > 0:
            sell_price = float(row.get("price", 0) or 0)
            sell_fee = float(row.get("fee", 0) or 0)
            profit = (sell_price - float(last_buy_price)) * last_buy_shares - (
                float(last_buy_fee) + sell_fee
            )
            if profit >= 0:
                win_count += 1
            else:
                loss_count += 1
            last_buy_price = None
            last_buy_shares = 0
            last_buy_fee = 0.0

    trade_count = win_count + loss_count
    win_rate = win_count / trade_count if trade_count > 0 else 0.0
    return trade_count, float(win_rate)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Run real-data backtest for range_adaptive_bollinger_reversion (BaoStock daily data)."
    )
    parser.add_argument("--symbol", required=True, help="600519 / sh.600519 / sz.000001")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--cash", type=float, default=100000.0)
    parser.add_argument("--fee_bps", type=float, default=3.0)
    parser.add_argument("--slippage", type=float, default=0.01)
    parser.add_argument("--stamp_bps", type=float, default=5.0)
    parser.add_argument("--execution", choices=["open", "close"], default="open")
    parser.add_argument("--adjust", default="2", help="BaoStock adjustflag: 1/2/3 (default 2)")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from quant_system.backtest.engine import run_single_backtest  # noqa: WPS433
    from quant_system.strategy.plugins import load_all_plugins  # noqa: WPS433

    load_all_plugins()

    symbol_full = normalize_symbol(args.symbol)
    strategy_ids = ["range_adaptive_bollinger_reversion"]

    try:
        result = run_single_backtest(
            symbol=symbol_full,
            start_date=args.start,
            end_date=args.end,
            strategy_ids=strategy_ids,
            mode="OR",
            strategy_params=None,
            initial_cash=float(args.cash),
            fee_rate=float(args.fee_bps) / 10_000.0,
            slippage=float(args.slippage),
            stamp_duty_rate=float(args.stamp_bps) / 10_000.0,
            execution_price=str(args.execution).lower(),
            adjustflag=str(args.adjust),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] backtest failed: {exc}", file=sys.stderr)
        return 1

    out_dir = project_root / "outputs" / f"{_symbol_tag(symbol_full)}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_df = result.equity_curve.reset_index()
    equity_df.columns = ["date", "equity"]
    if pd.api.types.is_datetime64_any_dtype(equity_df["date"]):
        equity_df["date"] = equity_df["date"].dt.strftime("%Y-%m-%d")
    equity_df.to_csv(out_dir / "equity_curve.csv", index=False)

    trades_df = result.trades.copy() if result.trades is not None else pd.DataFrame()
    if not trades_df.empty and "date" in trades_df.columns:
        trades_df["date"] = pd.to_datetime(trades_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    trades_df.to_csv(out_dir / "trades.csv", index=False)

    trade_count, win_rate = _compute_win_rate(trades_df)
    stats: Dict[str, Any] = dict(result.stats or {})
    stats["trade_count"] = trade_count
    stats["win_rate"] = win_rate

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    total_return = float(stats.get("total_return", 0.0) or 0.0)
    annual_return = float(stats.get("annual_return", 0.0) or 0.0)
    max_dd = float(stats.get("max_drawdown", 0.0) or 0.0)
    sharpe = float(stats.get("sharpe", 0.0) or 0.0)

    print("=== Range Backtest (Real Data) ===")
    print(f"symbol: {symbol_full}  period: {args.start} ~ {args.end}")
    print(f"total_return: {total_return:.2%}")
    print(f"annual_return: {annual_return:.2%}")
    print(f"max_drawdown: {max_dd:.2%}")
    print(f"sharpe: {sharpe:.2f}")
    print(f"trades: {trade_count}  win_rate: {win_rate:.2%}")
    print(f"outputs: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
