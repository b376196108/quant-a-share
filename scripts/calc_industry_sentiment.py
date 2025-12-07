from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 确保可以 import quant_system 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.processing.industry_sentiment import (  # noqa: E402
    calc_industry_sentiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算指定交易日的行业情绪统计结果（中文输出）。"
    )
    parser.add_argument(
        "--date",
        "-d",
        dest="trade_date",
        type=str,
        default=None,
        help="交易日，格式 YYYY-MM-DD；若不指定则使用数据库中最新交易日。",
    )
    parser.add_argument(
        "--min-stock",
        "-m",
        dest="min_stock_count",
        type=int,
        default=5,
        help="行业内最少股票数量过滤门槛（默认 5）。",
    )
    return parser.parse_args()


def main() -> None:
    # 确保标准输出为 UTF-8，尽量避免中文乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = parse_args()

    df = calc_industry_sentiment(
        trade_date=args.trade_date,
        min_stock_count=args.min_stock_count,
    )

    if df.empty:
        print("未计算出任何行业统计结果，请检查数据或过滤条件。")
        return

    trade_date_str = args.trade_date or "最新交易日"
    print(f"=== 行业情绪统计：{trade_date_str} ===")
    print()

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.unicode.ambiguous_as_wide", True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
