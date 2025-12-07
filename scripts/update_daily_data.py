"""Incrementally update A-share stock & index daily data to today."""

from __future__ import annotations

import sys
from pathlib import Path

import baostock as bs

# 把项目根目录加入 sys.path，方便脚本无论从哪里运行都能 import quant_system
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.data.storage import init_db_schema  # noqa: E402
from quant_system.data.fetcher import (  # noqa: E402
    update_stock_daily_to_today,
    update_index_daily_to_today,
)


def main() -> None:
    """
    运行模式：
      - 无参数 / all  : 更新股票 + 指数
      - stock         : 只更新股票
      - index         : 只更新指数
    用法示例：
      python scripts/update_daily_data.py
      python scripts/update_daily_data.py stock
      python scripts/update_daily_data.py index
    """
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    print(">>> 运行模式：", mode)

    print(">>> 初始化数据库表结构（若已存在则跳过）...")
    init_db_schema()

    print(">>> 登录 BaoStock ...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    try:
        if mode in ("all", "stock"):
            print(">>> 开始更新【股票】日线数据到今天...")
            update_stock_daily_to_today()
            print(">>> 【股票】日线更新完成。")

        if mode in ("all", "index"):
            print(">>> 开始更新【指数】日线数据到今天...")
            update_index_daily_to_today()
            print(">>> 【指数】日线更新完成。")

    finally:
        bs.logout()
        print(">>> BaoStock 已登出")


if __name__ == "__main__":
    main()
