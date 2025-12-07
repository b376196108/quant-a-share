"""Download all A-share daily data since 2015 and store into SQLite."""

from __future__ import annotations

import datetime as dt
import time
from typing import List
# >>> 新增这段 <<<
import sys
from pathlib import Path

# 当前文件： .../quant-a-share/scripts/download_all_daily_since_2015.py
# 项目根目录就是它的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# <<< 新增这段 >>>
import baostock as bs
import pandas as pd

from quant_system.data.storage import (
    init_db_schema,
    upsert_stock_daily,
)

FIELDS = ",".join(
    [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "pctChg",
        "turn",
    ]
)


def get_all_a_share_codes(trade_date: str) -> List[str]:
    """
    使用 BaoStock(query_all_stock) 获取全 A 股列表，过滤出主板/创业板股票。
    """
    rs = bs.query_all_stock(trade_date)
    code_list: List[str] = []

    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        code = row[0]
        # A 股过滤逻辑：sh.6xxxx、sz.0xxxx、sz.3xxxx
        if code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3"):
            code_list.append(code)

    return code_list


def download_one_stock(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        code,
        FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",  # 前复权
    )

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)

    # 字段格式调整
    df.rename(columns={"date": "trade_date", "pctChg": "pct_chg"}, inplace=True)

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "pct_chg",
        "turn",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    # 历史起始日 + 结束日（今天）
    start_date = "2015-01-01"
    today_date = dt.date.today()
    today_str = today_date.strftime("%Y-%m-%d")

    print(">>> 初始化数据库表结构...")
    init_db_schema()

    print(">>> 登录 BaoStock ...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

    try:
        # ========= 1. 自动向前回溯，找到最近一个有股票列表的交易日 =========
        code_list_date = today_date
        codes: List[str] = []

        for _ in range(15):  # 最多往前找 15 天
            date_str = code_list_date.strftime("%Y-%m-%d")
            print(f">>> 尝试获取股票列表（日期：{date_str}）...")
            codes = get_all_a_share_codes(date_str)
            if codes:
                print(f">>> 找到 {len(codes)} 只 A 股，使用 {date_str} 作为股票列表日期。")
                break
            code_list_date -= dt.timedelta(days=1)

        if not codes:
            print(">>> 连续 15 天都没有获取到股票列表，可能是网络或接口问题，退出。")
            return

        # ========= 2. 循环下载每只股票的日线数据（2015-01-01 ~ today） =========
        total = len(codes)
        for i, code in enumerate(codes, start=1):
            prefix = f"[{i}/{total}] 下载 {code} ..."
            try:
                df = download_one_stock(code, start_date, today_str)
                if df.empty:
                    print(f"{prefix} 无数据，跳过。")
                    continue

                upsert_stock_daily(df)
                print(f"{prefix} 写入 {len(df)} 行。")
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix} 失败：{exc}")
                continue

            # 防止请求过快，稍稍歇一下
            time.sleep(0.1)

    finally:
        bs.logout()
        print(">>> BaoStock 已登出")

    print(">>> 全部完成。")


if __name__ == "__main__":
    main()
