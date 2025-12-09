"""Incrementally update A-share stock & index daily data to today.

用法：
    # 默认：股票 + 指数 一起更新
    python scripts/update_daily_data.py

    # 只更新股票
    python scripts/update_daily_data.py stock

    # 只更新指数
    python scripts/update_daily_data.py index
"""

from __future__ import annotations

import sys
from pathlib import Path
import traceback

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
    """
    # ---------- 解析命令行参数 ----------
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    print("==================================================")
    print(">>> 日线增量更新脚本：update_daily_data.py")
    print(f">>> 项目根目录：{PROJECT_ROOT}")
    print(f">>> 运行模式：{mode}")
    print("==================================================")

    # ---------- 初始化数据库 ----------
    print(">>> 初始化数据库表结构（若已存在则跳过）...")
    try:
        init_db_schema()
        print(">>> 数据库表结构初始化完成。")
    except Exception as exc:
        print("!!! 初始化数据库表结构失败：", exc)
        traceback.print_exc()
        return

    # ---------- 登录 BaoStock ----------
    print(">>> 登录 BaoStock ...")
    lg = bs.login()
    if lg.error_code != "0":
        # 这里直接打印具体错误信息，方便排查（比如：服务暂停 / 账号问题等）
        print("!!! BaoStock 登录失败:")
        print("    error_code:", lg.error_code)
        print("    error_msg :", lg.error_msg)
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    print(">>> BaoStock 登录成功。")

    try:
        # ---------- 更新股票 ----------
        if mode in ("all", "stock"):
            print(">>> 开始更新【股票】日线数据到今天...")
            try:
                update_stock_daily_to_today()
                print(">>> 【股票】日线更新完成。")
            except Exception as exc:
                print("!!! 更新【股票】日线数据时发生异常：", exc)
                traceback.print_exc()
                # 如果你希望“股票失败也继续更新指数”，这里不要 return
                # 如果你希望失败就直接退出，可以改成：raise

        # ---------- 更新指数 ----------
        if mode in ("all", "index"):
            print(">>> 开始更新【指数】日线数据到今天...")
            try:
                update_index_daily_to_today()
                print(">>> 【指数】日线更新完成。")
            except Exception as exc:
                print("!!! 更新【指数】日线数据时发生异常：", exc)
                traceback.print_exc()

    finally:
        # ---------- 登出 BaoStock ----------
        bs.logout()
        print(">>> BaoStock 已登出")
        print(">>> 更新脚本结束。")
        print("==================================================")


if __name__ == "__main__":
    main()
