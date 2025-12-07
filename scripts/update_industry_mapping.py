"""Update stock industry mapping (similar to ShenWan industry) into SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

import baostock as bs
import pandas as pd

# 把项目根目录加入 sys.path，方便脚本无论从哪里运行都能 import quant_system
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.data.storage import (  # noqa: E402
    init_db_schema,
    upsert_stock_industry,
)


def fetch_industry_from_baostock() -> pd.DataFrame:
    """从 BaoStock 拉取全市场股票行业信息，返回 DataFrame。"""
    rs = bs.query_stock_industry()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)
    return df


def main() -> None:
    print(">>> 初始化数据库表结构（若已存在则跳过）...")
    init_db_schema()

    print(">>> 登录 BaoStock ...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    try:
        print(">>> 开始拉取全市场行业映射数据 ...")
        df = fetch_industry_from_baostock()
        if df.empty:
            print(">>> 未获取到任何行业数据，请检查网络或 BaoStock 接口。")
            return

        print(f">>> 共获取 {len(df)} 条行业记录，写入 SQLite ...")
        upsert_stock_industry(df)
        print(">>> 行业映射表 stock_industry 更新完成。")

    finally:
        bs.logout()
        print(">>> BaoStock 已登出")


if __name__ == "__main__":
    main()
