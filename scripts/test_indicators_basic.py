from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能找到 quant_system 包（项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.data.fetcher import get_stock_data
from quant_system.indicators.engine import calculate_indicators, list_indicators


def main():
    print("当前已注册指标：", list_indicators())

    # 有数据的股票随便选一只
    code = "sh.600519"

    df = get_stock_data(
        code=code,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    print("原始数据列：", df.columns.tolist())

    # 1）测试基础指标
    df1 = calculate_indicators(df.copy(), indicators=["sma", "macd", "rsi"])
    print("\n追加基础指标后的部分列：")
    print([c for c in df1.columns if "SMA" in c or "MACD" in c.upper() or "RSI" in c.upper()])
    print(df1.tail(3))

    # 2）测试 ultimate_features（终极特征）
    df2 = calculate_indicators(df.copy(), indicators=["ultimate_features"])
    cols = [
        "SUPERTd",
        "SQZPRO_ON",
        "MACDh",
        "VWMACDh",
        "EWO",
        "RSI",
        "MA5_UP_MA20",
        "vol_adj_ratio",
    ]
    print("\nultimate_features 新增列是否存在：")
    for c in cols:
        print(f"{c:15}: ", "OK" if c in df2.columns else "缺失")

    # 只取真正存在的列，避免 KeyError
    present_cols = [c for c in cols if c in df2.columns]
    if present_cols:
        print("\nultimate_features 实际存在的列最后3行：")
        print(df2[present_cols].tail(3))
    else:
        print("\nultimate_features 暂时没有任何预期列，请检查插件实现。")



if __name__ == "__main__":
    main()
