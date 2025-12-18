"""A-share market data fetcher with BaoStock + local caching and SQLite sync."""

from __future__ import annotations

import datetime as _dt
from datetime import date, timedelta
from typing import Iterable, List, Optional
import traceback
import threading

import baostock as bs
import pandas as pd
import time as _time

from quant_system.data.storage import (
    init_db_schema,
    load_from_cache,
    save_to_cache,
    upsert_stock_daily,
    upsert_stock_info,
    upsert_index_daily,
    get_latest_trade_date,
)

# 默认 K 线字段（给 Notebook 用）
DEFAULT_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "preclose",
    "pctChg",
    "turn",
    "tradestatus",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
]

# 默认指数列表：上证、深成、创业板、沪深300
INDEX_CODES = {
    "sh.000001": "上证指数",
    "sz.399001": "深证成指",
    "sz.399006": "创业板指",
    "sh.000300": "沪深300",
}

_LOGGED_IN = False
_BS_LOCK = threading.RLock()

# BaoStock client is not thread-safe; treat common network/session issues as transient.
_TRANSIENT_BS_ERROR_CODES = {"10002007", "10002006", "10001001"}


def _reset_bs_session() -> None:
    """Best-effort reset BaoStock session (logout + mark as not logged in)."""
    global _LOGGED_IN
    try:
        bs.logout()
    except Exception:
        pass
    _LOGGED_IN = False


# -----------------------------------------------------------------------------
# BaoStock login / 基础工具
# -----------------------------------------------------------------------------


def _ensure_bs_login() -> bool:
    """Login to BaoStock once."""
    global _LOGGED_IN
    with _BS_LOCK:
        if _LOGGED_IN:
            return True
        resp = bs.login()
        if resp.error_code != "0":
            print(f"[BaoStock] 登录失败：{resp.error_code} {resp.error_msg}")
            return False
        _LOGGED_IN = True
        return True


def _validate_dates(start_date: str, end_date: str) -> bool:
    """Validate date strings in YYYY-MM-DD format and order."""
    try:
        start = _dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = _dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        print(f"[参数错误] 日期格式不正确：start={start_date}, end={end_date}")
        return False
    if start > end:
        print(f"[参数错误] 起始日期晚于结束日期：{start_date} > {end_date}")
        return False
    return True


def _coerce_numeric(series, col_name: str, context: str) -> pd.Series:
    """
    Convert a Series-like object to numeric, handling duplicated columns gracefully.
    """
    try:
        return pd.to_numeric(series, errors="coerce")
    except TypeError:
        if isinstance(series, pd.DataFrame):
            print(
                f"[warn] {context}: column '{col_name}' duplicated; using the first occurrence for numeric conversion"
            )
            return pd.to_numeric(series.iloc[:, 0], errors="coerce")

        print(
            f"[warn] {context}: failed to convert column '{col_name}' (type={type(series)}) to numeric; treating as NaN"
        )
        return pd.to_numeric(pd.Series(series), errors="coerce")


def _format_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw result strings to typed DataFrame with date index."""
    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = [c for c in df.columns if c not in {"date", "code"}]
    for col in numeric_cols:
        df[col] = _coerce_numeric(df[col], col, "format_history_df")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date").set_index("date")
    return df

def _format_seconds(sec: float) -> str:
    """把秒数格式化为“X小时Y分Z秒”的字符串，便于打印。"""
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}小时{m:02d}分{s:02d}秒"
    if m > 0:
        return f"{m}分{s:02d}秒"
    return f"{s}秒"


def _print_progress(prefix: str, current: int, total: int, start_ts: float) -> None:
    """
    终端进度条工具：
    - prefix：前缀说明文字，例如“股票日线更新进度”
    - current：当前完成数量
    - total：总数量
    - start_ts：开始时间戳（time.time()）
    """
    if total <= 0:
        return

    elapsed = _time.time() - start_ts
    rate = current / elapsed if elapsed > 0 else 0.0
    remain = (total - current) / rate if rate > 0 else 0.0

    bar_len = 30  # 进度条长度
    filled = int(bar_len * current / total)
    bar = "█" * filled + "-" * (bar_len - filled)

    msg = (
        f"\r{prefix} [{bar}] {current}/{total} "
        f"({current / total:.1%}) | 已用 {_format_seconds(elapsed)}"
        f" | 预计剩余 {_format_seconds(remain)}"
    )
    print(msg, end="", flush=True)
    if current >= total:
        print()  # 完成后换行


def _fetch_remote_history(
    code: str,
    start_date: str,
    end_date: str,
    freq: str,
    fields: List[str],
    adjust: str,
) -> pd.DataFrame:
    if freq != "d":
        print(f"[数据获取] 暂不支持 freq={freq}，当前仅支持日线（'d'）。")
        return pd.DataFrame()

    field_str = ",".join(fields)
    max_retries = 3
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        if not _ensure_bs_login():
            return pd.DataFrame()

        try:
            # BaoStock uses a global socket/session; it must be serialized across threads.
            with _BS_LOCK:
                rs = bs.query_history_k_data_plus(
                    code,
                    field_str,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=freq,
                    adjustflag=adjust,
                )

                if rs.error_code != "0":
                    if (rs.error_code in _TRANSIENT_BS_ERROR_CODES) or ("网络" in (rs.error_msg or "")):
                        raise RuntimeError(f"BaoStock transient error {rs.error_code}: {rs.error_msg}")

                    print(f"[数据获取] 拉取失败：{code}，错误 {rs.error_code} - {rs.error_msg}")
                    return pd.DataFrame()

                rows: List[List[str]] = []
                while rs.next():
                    rows.append(rs.get_row_data())

        except Exception as exc:  # noqa: BLE001
            last_err = exc
            with _BS_LOCK:
                _reset_bs_session()

            if attempt < max_retries:
                _time.sleep(0.5 * attempt)
                continue

            print(f"[数据获取] 拉取失败：{code}，异常：{last_err}")
            return pd.DataFrame()

        if not rows:
            print(f"[数据获取] 无数据：{code}，时间区间 {start_date} ~ {end_date}")
            return pd.DataFrame(columns=rs.fields)

        df = pd.DataFrame(rows, columns=rs.fields)
        return _format_history_df(df)

    print(f"[数据获取] 拉取失败：{code}，异常：{last_err}")
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 对 Notebook 友好的单票拉数接口（带 CSV 缓存）
# -----------------------------------------------------------------------------


def get_stock_data(
    code: str,
    start_date: str,
    end_date: str,
    freq: str = "d",
    fields: Optional[List[str]] = None,
    adjust: str = "2",
) -> pd.DataFrame:
    """
    Fetch historical K-line data for a single stock (daily by default) with local CSV cache.
    """
    if not _validate_dates(start_date, end_date):
        return pd.DataFrame()

    fields_to_use = fields or DEFAULT_FIELDS
    cache_key = f"stock_{code.replace('.', '-')}_{start_date}_{end_date}_{freq}.csv"
    cached = load_from_cache(cache_key)
    if cached is not None and not cached.empty:
        return cached

    df = _fetch_remote_history(code, start_date, end_date, freq=freq, fields=fields_to_use, adjust=adjust)
    if not df.empty:
        save_to_cache(cache_key, df)
    return df


def get_index_data(
    code: str,
    start_date: str,
    end_date: str,
    freq: str = "d",
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch index daily data (e.g., CSI 300) with the same interface style as get_stock_data.
    """
    if not _validate_dates(start_date, end_date):
        return pd.DataFrame()

    fields_to_use = fields or DEFAULT_FIELDS
    cache_key = f"index_{code.replace('.', '-')}_{start_date}_{end_date}_{freq}.csv"
    cached = load_from_cache(cache_key)
    if cached is not None and not cached.empty:
        return cached

    df = _fetch_remote_history(code, start_date, end_date, freq=freq, fields=fields_to_use, adjust="2")
    if not df.empty:
        save_to_cache(cache_key, df)
    return df


# -----------------------------------------------------------------------------
# 全市场批量拉数 + 写入 SQLite
# -----------------------------------------------------------------------------


def fetch_all_stock_info(day: str) -> pd.DataFrame:
    """
    Fetch all stock info on a given date, upsert into stock_info, and return DataFrame.
    """
    init_db_schema()
    if not _ensure_bs_login():
        return pd.DataFrame()

    rs = bs.query_all_stock(day)
    if rs.error_code != "0":
        print(f"[baostock] query_all_stock failed: {rs.error_code} {rs.error_msg}")
        return pd.DataFrame()

    rows: List[List[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        print(f"[baostock] query_all_stock returned empty for {day}")
        return pd.DataFrame(columns=rs.fields)

    df = pd.DataFrame(rows, columns=rs.fields)
    upsert_stock_info(df)
    return df


def _history_rows_to_df(rows: List[List[str]], fields: List[str]) -> pd.DataFrame:
    """
    将 BaoStock 返回的 rows + fields 转成统一字段的 DataFrame，方便写入日线表。
    同时适用于个股和指数（区别由 upsert_* 决定）。
    """
    df = pd.DataFrame(rows, columns=fields)
    # 去重字段，避免 pctChg -> pct_chg 后出现重复列导致行值变成 Series
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 先统一字段名
    if "pctChg" in df.columns:
        df = df.rename(columns={"pctChg": "pct_chg"})
    rename_map = {
        "peTTM": "pe_ttm",
        "pbMRQ": "pb_mrq",
        "psTTM": "ps_ttm",
        "pcfNcfTTM": "pcf_ncf_ttm",
        "isST": "is_st",
    }
    df = df.rename(columns=rename_map)

    numeric_cols = [c for c in df.columns if c not in {"date", "trade_date", "code"}]
    for col in numeric_cols:
        df[col] = _coerce_numeric(df[col], col, "_history_rows_to_df")

    # 确保 pct_chg 列存在
    if "pct_chg" not in df.columns:
        df["pct_chg"] = None
    keep_cols = [
        "code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "pct_chg",
        "turn",
        "tradestatus",
        "pe_ttm",
        "pb_mrq",
        "ps_ttm",
        "pcf_ncf_ttm",
        "is_st",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    return df[keep_cols]

def _get_stock_info_with_backoff(target_day: _dt.date, lookback_days: int = 10) -> tuple[str, pd.DataFrame]:
    """
    从 target_day 开始向前回溯，找到最近一个能成功拿到股票列表的交易日。

    返回: (实际使用的交易日字符串, 对应日的股票列表 DataFrame)
    若连续 lookback_days 天都失败，则返回 ("", 空 DataFrame)。
    """
    for i in range(lookback_days):
        day = target_day - timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        print(f"[fetcher] 尝试获取 {day_str} 的股票列表...")
        df = fetch_all_stock_info(day_str)
        if df is not None and not df.empty:
            print(f"[fetcher] 使用 {day_str} 的股票列表，共 {len(df)} 只股票。")
            return day_str, df

    print(f"[fetcher] 在过去 {lookback_days} 天都没获取到股票列表。")
    return "", pd.DataFrame()


def fetch_all_stock_daily(
    start_date: str,
    end_date: str,
) -> None:
    """
    拉取全市场 A 股在区间 [start_date, end_date] 的日线数据，并写入 stock_daily。
    可用于历史初始化或某个时间段的补数。
    """
    if not _validate_dates(start_date, end_date):
        return

    # 用 end_date（通常是“今天”）作为优先日期，
    # 如果当天 query_all_stock 暂时没有数据，就自动往前回溯几天。
    target_day = _dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    list_day_str, stock_info_df = _get_stock_info_with_backoff(target_day, lookback_days=10)

    if stock_info_df is None or stock_info_df.empty:
        print("[股票] 未能获取到股票列表，放弃本次日线更新。")
        return

    codes = stock_info_df["code"].tolist()
    fields = [
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
        "tradestatus",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "isST",
    ]
    total = len(codes)

    print(f"[股票] 使用股票列表日期：{list_day_str}，共 {total} 只股票。")
    print(f"[股票] 准备下载区间：{start_date} ~ {end_date}")

    start_ts = _time.time()
    success_cnt = 0
    empty_codes: List[str] = []
    failed_codes: List[str] = []

    for idx, code in enumerate(codes, start=1):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                ",".join(fields),
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                print(f"\n[\u80a1\u7968] \u62c9\u53d6 {code} \u5931\u8d25\uff1aBaoStock \u9519\u8bef {rs.error_code} {rs.error_msg}")
                failed_codes.append(code)
                continue

            rows: List[List[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                empty_codes.append(code)
                continue

            df = _history_rows_to_df(rows, fields)
            upsert_stock_daily(df)
            success_cnt += 1
        except Exception as exc:  # noqa: BLE001
            print(f"\n[\u80a1\u7968] \u62c9\u53d6 {code} \u5f02\u5e38\uff1a{exc}")
            traceback.print_exc()
            failed_codes.append(code)
            continue

        # \u6bcf\u53ea\u80a1\u7968\u66f4\u65b0\u4e00\u6b21\u8fdb\u5ea6\u6761
        _print_progress("\u80a1\u7968\u65e5\u7ebf\u66f4\u65b0\u8fdb\u5ea6", idx, total, start_ts)

    elapsed = _time.time() - start_ts
    print(
        f"[\u5b8c\u6210] \u80a1\u7968\u65e5\u7ebf\u66f4\u65b0\u5b8c\u6210\uff1a\u6210\u529f {success_cnt}/{total}\uff0c\u65e0\u6570\u636e {len(empty_codes)}\uff0c\u5931\u8d25 {len(failed_codes)}\uff0c\u533a\u95f4 {start_date} ~ {end_date}\uff0c\u8017\u65f6 {_format_seconds(elapsed)}"
    )
    if failed_codes:
        print(f"[\u5931\u8d25\u793a\u4f8b] {', '.join(failed_codes[:10])}")


def fetch_index_daily(
    start_date: str,
    end_date: str,
    codes: Optional[Iterable[str]] = None,
) -> None:
    """
    拉取指定指数在区间 [start_date, end_date] 的日线数据，并写入 index_daily。
    """
    if not _validate_dates(start_date, end_date):
        return
    init_db_schema()
    if not _ensure_bs_login():
        return

    idx_codes = list(codes) if codes is not None else list(INDEX_CODES.keys())
    fields = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]

    total = len(idx_codes)
    print(f"[指数] 需要更新 {total} 个指数，区间：{start_date} ~ {end_date}")
    start_ts = _time.time()
    success_cnt = 0
    empty_codes: List[str] = []
    failed_codes: List[str] = []

    for i, code in enumerate(idx_codes, start=1):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                ",".join(fields),
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                print(f"\n[\u6307\u6570] \u62c9\u53d6 {code} \u5931\u8d25\uff1aBaoStock \u9519\u8bef {rs.error_code} {rs.error_msg}")
                failed_codes.append(code)
                continue

            rows: List[List[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                empty_codes.append(code)
                continue

            df = _history_rows_to_df(rows, fields)
            upsert_index_daily(df)
            success_cnt += 1
        except Exception as exc:  # noqa: BLE001
            print(f"\n[\u6307\u6570] \u62c9\u53d6 {code} \u5f02\u5e38\uff1a{exc}")
            traceback.print_exc()
            failed_codes.append(code)
            continue

        _print_progress("\u6307\u6570\u65e5\u7ebf\u66f4\u65b0\u8fdb\u5ea6", i, total, start_ts)

    elapsed = _time.time() - start_ts
    print(
        f"[\u5b8c\u6210] \u6307\u6570\u65e5\u7ebf\u66f4\u65b0\u5b8c\u6210\uff1a\u6210\u529f {success_cnt}/{total}\uff0c\u65e0\u6570\u636e {len(empty_codes)}\uff0c\u5931\u8d25 {len(failed_codes)}\uff0c\u533a\u95f4 {start_date} ~ {end_date}\uff0c\u8017\u65f6 {_format_seconds(elapsed)}"
    )
    if failed_codes:
        print(f"[\u5931\u8d25\u793a\u4f8b] {', '.join(failed_codes[:10])}")





# -----------------------------------------------------------------------------
# 增量更新：把库里数据补到今天（股票 & 指数）
# -----------------------------------------------------------------------------


def _next_date_str(day_str: str) -> str:
    d = _dt.datetime.strptime(day_str, "%Y-%m-%d").date()
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def update_stock_daily_to_today(
    default_start: str = "2015-01-01",
) -> None:
    """
    将 stock_daily 表从当前最新交易日增量更新到今天。
    若表为空，则从 default_start 开始全量补数。
    """
    init_db_schema()
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    latest_str = get_latest_trade_date("stock_daily")
    if latest_str is None:
        start_date = default_start
        print(f"[更新-股票] 当前表为空，将从 {start_date} 开始补数到 {today_str}。")
    else:
        latest_date = _dt.datetime.strptime(latest_str, "%Y-%m-%d").date()
        if latest_date >= today:
            print("[更新-股票] stock_daily 已是最新，无需更新。")
            return
        start_date = _next_date_str(latest_str)
        print(f"[更新-股票] 最新交易日：{latest_str}，将补数 {start_date} -> {today_str}。")

    fetch_all_stock_daily(start_date, today_str)


def update_index_daily_to_today(
    default_start: str = "2015-01-01",
    codes: Optional[Iterable[str]] = None,
) -> None:
    """
    将 index_daily 表从当前最新交易日增量更新到今天。
    若表为空，则从 default_start 开始全量补数。
    """
    init_db_schema()
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    latest_str = get_latest_trade_date("index_daily")
    if latest_str is None:
        start_date = default_start
        print(f"[更新-指数] 当前表为空，将从 {start_date} 开始补数到 {today_str}。")
    else:
        latest_date = _dt.datetime.strptime(latest_str, "%Y-%m-%d").date()
        if latest_date >= today:
            print("[更新-指数] index_daily 已是最新，无需更新。")
            return
        start_date = _next_date_str(latest_str)
        print(f"[更新-指数] 最新交易日：{latest_str}，将补数 {start_date} -> {today_str}。")

    fetch_index_daily(start_date, today_str, codes=codes)



# 为了兼容你之前脚本里调用的名字，保留一个老接口别名
def update_daily_to_today() -> None:
    """
    兼容旧接口：仅更新股票日线到今天。
    """
    update_stock_daily_to_today()
