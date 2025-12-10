"""Local cache utilities for CSV and SQLite storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# CSV 还是放在 data_cache 目录
from pathlib import Path

# 当前文件路径：.../quant-a-share/quant_system/data/storage.py
# parents[0] = data
# parents[1] = quant_system
# parents[2] = 项目根目录 quant-a-share
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data_cache"             # csv 缓存目录
DB_PATH = DATA_DIR / "market_data.sqlite"      # ★ 就是你刚才检查的那份库

print("[storage] BASE_DIR =", BASE_DIR)
print("[storage] DB_PATH  =", DB_PATH)

# -----------------------------------------------------------------------------
# CSV cache helpers (kept for backward compatibility)
# -----------------------------------------------------------------------------


def _ensure_cache_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_from_cache(cache_key: str) -> Optional[pd.DataFrame]:
    """
    Load CSV by cache_key. Return None when missing.
    """
    path = DATA_DIR / cache_key
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=[0], index_col=0)
    except Exception as exc:
        print(f"[cache] failed to read {path}: {exc}")
        return None


def save_to_cache(cache_key: str, df: pd.DataFrame) -> None:
    """
    Save DataFrame to CSV for quick reuse.
    """
    _ensure_cache_dir()
    path = DATA_DIR / cache_key
    try:
        df.to_csv(path)
    except Exception as exc:
        print(f"[cache] failed to save {path}: {exc}")


# -----------------------------------------------------------------------------
# SQLite helpers
# -----------------------------------------------------------------------------


def get_db_connection(timeout: float = 30.0) -> sqlite3.Connection:
    """
    Return SQLite connection with sensible defaults for concurrent read/write.

    - timeout/busy_timeout: give writers time to wait for short-lived readers
    - WAL mode: allow reads while writing (needed when API is serving requests)
    - synchronous=NORMAL: balance durability and write speed for cache usage
    """
    _ensure_cache_dir()
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=timeout,
    )
    conn.row_factory = sqlite3.Row
    # Improve concurrent access characteristics
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db_schema() -> None:
    """Create stock_info, stock_daily, index_daily tables if not exist."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_info (
                code TEXT PRIMARY KEY,
                code_name TEXT,
                ipo_date TEXT,
                out_date TEXT,
                type TEXT,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_daily (
                code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                preclose REAL,
                volume REAL,
                amount REAL,
                pct_chg REAL,
                turn REAL,
                tradestatus INTEGER,
                pe_ttm REAL,
                pb_mrq REAL,
                ps_ttm REAL,
                pcf_ncf_ttm REAL,
                is_st INTEGER,
                PRIMARY KEY (code, trade_date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_daily (
                code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                preclose REAL,
                volume REAL,
                amount REAL,
                pct_chg REAL,
                PRIMARY KEY (code, trade_date)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_daily_trade_date
            ON stock_daily(trade_date)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_index_daily_trade_date
            ON index_daily(trade_date)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_industry (
                code TEXT PRIMARY KEY,
                code_name TEXT,
                industry TEXT,
                industry_classification TEXT,
                update_date TEXT
            )
            """
        )
        conn.commit()
        # 在建表后尝试迁移老库，确保新字段齐备
        migrate_stock_daily_add_new_fields(conn)
    finally:
        conn.close()


def migrate_stock_daily_add_new_fields(conn: sqlite3.Connection) -> None:
    """
    为 stock_daily 增加新增字段（若已存在则忽略），便于兼容旧库。
    """
    cursor = conn.cursor()
    columns = {
        "turn": "REAL",
        "tradestatus": "INTEGER",
        "pe_ttm": "REAL",
        "pb_mrq": "REAL",
        "ps_ttm": "REAL",
        "pcf_ncf_ttm": "REAL",
        "is_st": "INTEGER",
    }
    for name, col_type in columns.items():
        try:
            cursor.execute(f"ALTER TABLE stock_daily ADD COLUMN {name} {col_type}")
        except Exception:
            # 已存在时忽略
            pass
    conn.commit()


def _df_to_records(df: pd.DataFrame, columns: List[str]) -> List[tuple]:
    """Convert DataFrame to list of tuples with selected columns."""
    return [tuple(row[col] for col in columns) for _, row in df.iterrows()]


def _coerce_numeric(series, col_name: str, context: str) -> pd.Series:
    """
    Convert a Series-like object to numeric, handling duplicated columns gracefully.
    """
    try:
        return pd.to_numeric(series, errors="coerce")
    except TypeError:
        if isinstance(series, pd.DataFrame):
            print(
                f"[storage] {context}: column '{col_name}' duplicated; using the first occurrence for numeric conversion"
            )
            return pd.to_numeric(series.iloc[:, 0], errors="coerce")

        print(f"[storage] {context}: failed to convert column '{col_name}' (type={type(series)}) to numeric; treating as NaN")
        return pd.to_numeric(pd.Series(series), errors="coerce")


def upsert_stock_info(df: pd.DataFrame) -> None:
    """Upsert stock basic info by code."""
    if df is None or df.empty:
        return
    df = df.copy()
    # Normalize date columns if present
    if "ipoDate" in df.columns:
        df["ipo_date"] = pd.to_datetime(df["ipoDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "outDate" in df.columns:
        df["out_date"] = pd.to_datetime(df["outDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df.rename(columns={"ipoDate": "ipo_date", "outDate": "out_date"}, inplace=True)

    columns = ["code", "code_name", "ipo_date", "out_date", "type", "status"]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    records = _df_to_records(df[columns], columns)

    conn = get_db_connection()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_info (code, code_name, ipo_date, out_date, type, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    finally:
        conn.close()


def upsert_stock_industry(df: pd.DataFrame) -> None:
    """
    Upsert 股票行业信息（类似申万行业）到 stock_industry 表。

    预期 df 来自 BaoStock 的 query_stock_industry 接口，
    常见字段包括：
        code, code_name, industry, industryClassification, updateDate
    """
    if df is None or df.empty:
        return

    df = df.copy()

    # 标准化列名：industryClassification -> industry_classification
    if "industryClassification" in df.columns:
        df.rename(
            columns={"industryClassification": "industry_classification"},
            inplace=True,
        )

    # 标准化日期列：updateDate -> update_date，格式化为 YYYY-MM-DD
    if "updateDate" in df.columns and "update_date" not in df.columns:
        df.rename(columns={"updateDate": "update_date"}, inplace=True)
    if "update_date" in df.columns:
        df["update_date"] = pd.to_datetime(
            df["update_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    # 确保所有需要的列存在
    columns = ["code", "code_name", "industry", "industry_classification", "update_date"]
    for col in columns:
        if col not in df.columns:
            df[col] = None

    # 转换为 records 列表，沿用现有的 _df_to_records 工具函数
    records = _df_to_records(df[columns], columns)

    conn = get_db_connection()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_industry (
                code, code_name, industry, industry_classification, update_date
            ) VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            print("[storage] stock_industry upsert waited but database is still locked; is another process writing?")
        raise
    finally:
        conn.close()


def _prepare_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 去重字段，避免 pct_chg 重复导致行值变成 Series
    df = df.loc[:, ~df.columns.duplicated()]
    if "date" in df.columns:
        df.rename(columns={"date": "trade_date"}, inplace=True)
    rename_map = {
        "pctChg": "pct_chg",
        "peTTM": "pe_ttm",
        "pbMRQ": "pb_mrq",
        "psTTM": "ps_ttm",
        "pcfNcfTTM": "pcf_ncf_ttm",
        "isST": "is_st",
    }
    df.rename(columns=rename_map, inplace=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_cols = [c for c in df.columns if c not in {"trade_date", "code"}]
    for col in numeric_cols:
        df[col] = _coerce_numeric(df[col], col, "_prepare_daily_df")
    if "pct_chg" not in df.columns:
        df["pct_chg"] = None
    return df


def upsert_stock_daily(df: pd.DataFrame) -> None:
    """Upsert stock daily data by (code, trade_date)."""
    if df is None or df.empty:
        return
    df = _prepare_daily_df(df)
    columns = [
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
    for col in columns:
        if col not in df.columns:
            df[col] = None
    records = _df_to_records(df[columns], columns)

    conn = get_db_connection()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_daily (
                code, trade_date, open, high, low, close, preclose, volume, amount, pct_chg,
                turn, tradestatus, pe_ttm, pb_mrq, ps_ttm, pcf_ncf_ttm, is_st
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    finally:
        conn.close()


def upsert_index_daily(df: pd.DataFrame) -> None:
    """Upsert index daily data by (code, trade_date)."""
    if df is None or df.empty:
        return
    df = _prepare_daily_df(df)
    columns = [
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
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    records = _df_to_records(df[columns], columns)

    conn = get_db_connection()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO index_daily (
                code, trade_date, open, high, low, close, preclose, volume, amount, pct_chg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    finally:
        conn.close()


def _load_daily(
    table: str,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    codes = list(codes)
    if not codes:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in codes)
    sql = f"""
        SELECT * FROM {table}
        WHERE code IN ({placeholders})
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date, code
    """
    params: List[str] = [*codes, start_date, end_date]
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[col[0] for col in cur.description])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    numeric_cols = [c for c in df.columns if c not in {"trade_date", "code"}]
    for col in numeric_cols:
        df[col] = _coerce_numeric(df[col], col, "_load_daily")
    df = df.set_index(["trade_date", "code"])
    return df


def load_stock_daily(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load stock daily data with MultiIndex (trade_date, code).
    """
    return _load_daily("stock_daily", codes, start_date, end_date)


def load_stock_daily_with_industry(trade_date: str) -> pd.DataFrame:
    """
    读取某一交易日的全市场日线数据，并附带行业信息。

    返回值：
        一个 DataFrame，包含来自 stock_daily 表的行情字段，
        以及 stock_industry 表中的行业字段：
            - code
            - trade_date
            - open, high, low, close, preclose, volume, amount, ...
            - code_name
            - industry
            - industry_classification
            - update_date
    """
    conn = get_db_connection()
    # 只选取行业情绪计算需要的字段，减少 IO
    query = """
    SELECT
        d.code,
        d.trade_date,
        d.close,
        d.preclose,
        d.amount,
        i.industry
    FROM stock_daily AS d
    LEFT JOIN stock_industry AS i
        ON d.code = i.code
    WHERE d.trade_date = ?
    ORDER BY d.code
    """
    try:
        df = pd.read_sql_query(query, conn, params=(trade_date,))
    finally:
        conn.close()
    return df


def load_index_daily(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load index daily data with MultiIndex (trade_date, code)."""
    return _load_daily("index_daily", codes, start_date, end_date)


# -----------------------------------------------------------------------------
# Helpers for incremental update
# -----------------------------------------------------------------------------


def get_latest_trade_date(table: str = "stock_daily") -> Optional[str]:
    """
    获取指定表(stock_daily / index_daily)里的最新交易日字符串 YYYY-MM-DD。
    若表为空则返回 None。
    """
    conn = get_db_connection()
    try:
        cur = conn.execute(f"SELECT MAX(trade_date) FROM {table}")
        row = cur.fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] else None
