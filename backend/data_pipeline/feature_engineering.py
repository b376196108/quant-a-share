"""Feature engineering pipeline for 5-day stock trend prediction.

本模块负责：
1. 从现有 SQLite 库 (market_data.sqlite) 读取个股日线数据（stock_daily 表）。
2. 计算技术特征（收益、波动率、均线、RSI、MACD 等）。
3. 根据三路屏障法 (Triple Barrier Method) 生成未来 5 日趋势标签：
   - 1: 在 horizon 内先触及上轨（多头）
   - -1: 在 horizon 内先触及下轨（空头）
   - 0: 未触及任一屏障（中性）
4. 将特征 + 标签存入 backend/tft_model/data 目录下的 Parquet 文件，
   供训练脚本 backend/tft_model/train.py 使用。

注意：
- 仅依赖现有的日线级别数据，不重新调用 BaoStock。
- 作为独立脚本运行时，可以指定单只股票或全市场批量生成特征。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# quant_system 内已有 SQLite 连接与路径工具
try:
    from quant_system.data.storage import get_db_connection
except Exception as exc:  # pragma: no cover - 方便在未安装包时调试
    get_db_connection = None  # type: ignore
    print("[feature_engineering] Warning: cannot import quant_system.data.storage:", exc)


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

# 当前文件：backend/data_pipeline/feature_engineering.py
BACKEND_DIR = Path(__file__).resolve().parents[1]          # backend
PROJECT_ROOT = BACKEND_DIR.parent                          # 项目根目录
FEATURE_DIR = BACKEND_DIR / "tft_model" / "data"           # 特征输出目录
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 工具：代码规范化 / 技术指标 / 三路屏障
# ---------------------------------------------------------------------------


def normalize_stock_code(code: str) -> str:
    """将 6 位代码规范为 BaoStock / 数据库使用的形式，例如 600519 -> sh.600519。

    约定：
    - 以 6 开头：上证 sh.
    - 以 0 或 3 开头：深证 sz.
    - 已经包含 '.' 的视为完整代码，原样返回。
    """
    code = code.strip()
    if "." in code:
        return code.lower()

    if len(code) == 6:
        if code.startswith("6"):
            return f"sh.{code}"
        elif code.startswith(("0", "3")):
            return f"sz.{code}"

    # 兜底：直接返回
    return code


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """简单 RSI 实现，返回 0-100 区间的相对强弱指标。"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均 (EMA)。"""
    return series.ewm(span=span, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD 指标：返回 (macd, signal, hist)。"""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


@dataclass
class TripleBarrierConfig:
    horizon: int = 5          # 未来 N 个交易日观察窗口
    up_mult: float = 1.5      # 上轨系数 μ
    down_mult: float = 1.5    # 下轨系数 λ
    vol_window: int = 20      # 波动率估计窗口（日）


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """在原始日线数据基础上添加一系列技术特征。

    输入 df 要求至少包含：
        ["trade_date", "close", "open", "high", "low", "volume", "amount"]

    返回与输入等长的 DataFrame（保持 trade_date 顺序），新增列包括：
        - ret_1d, ret_3d, ret_5d, ret_10d, ret_20d
        - vol_5, vol_20
        - ma_5, ma_10, ma_20, ma_60
        - rsi_14
        - macd, macd_signal, macd_hist
    """
    df = df.copy()
    df = df.sort_values("trade_date")
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 使用收盘价计算简单收益与对数收益
    df["ret_1d"] = df["close"].pct_change()
    for lag in (3, 5, 10, 20):
        df[f"ret_{lag}d"] = df["close"].pct_change(periods=lag)

    # 波动率（基于 1 日收益）
    df["vol_5"] = df["ret_1d"].rolling(5).std()
    df["vol_20"] = df["ret_1d"].rolling(20).std()

    # 均线
    for win in (5, 10, 20, 60):
        df[f"ma_{win}"] = df["close"].rolling(win).mean()

    # 均线斜率（简单差分）
    df["ma_5_slope"] = df["ma_5"].diff()
    df["ma_20_slope"] = df["ma_20"].diff()

    # RSI & MACD
    df["rsi_14"] = _rsi(df["close"], window=14)
    macd, macd_signal, macd_hist = _macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # 成交量相关
    df["vol_ma_5"] = df["volume"].rolling(5).mean()
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_20"]

    # 处理无限值与异常
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def apply_triple_barrier(
    df: pd.DataFrame,
    cfg: TripleBarrierConfig = TripleBarrierConfig(),
) -> pd.Series:
    """基于三路屏障法生成标签列。

    三路屏障设定：
        Upper_Barrier = P_t * (1 + up_mult * sigma_t)
        Lower_Barrier = P_t * (1 - down_mult * sigma_t)
        时间窗口 = [t+1, t+horizon]

    其中 sigma_t 使用 vol_window（默认 20 日）收益标准差估计。
    返回 Series：index 与 df 对齐，值为 {1, -1, 0, NaN}
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # 波动率估计
    ret_1d = close.pct_change()
    sigma = ret_1d.rolling(cfg.vol_window).std()

    n = len(df)
    labels = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if i >= n - 1:
            break  # 最后一天之后没有未来数据了

        p0 = close.iloc[i]
        if not np.isfinite(p0):
            continue

        vol = sigma.iloc[i]
        if not np.isfinite(vol) or vol <= 0:
            # 如果此处波动率缺失，尝试用全局中位数兜底
            vol = np.nanmedian(sigma.values)
            if not np.isfinite(vol) or vol <= 0:
                continue

        up_price = p0 * (1 + cfg.up_mult * vol)
        down_price = p0 * (1 - cfg.down_mult * vol)

        up_hit: Optional[int] = None
        down_hit: Optional[int] = None

        # 观察未来 horizon 天内的价格路径
        max_j = min(cfg.horizon, n - 1 - i)
        for j in range(1, max_j + 1):
            hi = high.iloc[i + j]
            lo = low.iloc[i + j]
            if up_hit is None and np.isfinite(hi) and hi >= up_price:
                up_hit = j
            if down_hit is None and np.isfinite(lo) and lo <= down_price:
                down_hit = j

            # 两个屏障都已经在某日被触及，提前结束
            if up_hit is not None and down_hit is not None:
                break

        label = 0.0
        if up_hit is not None or down_hit is not None:
            if up_hit is not None and (down_hit is None or up_hit < down_hit):
                label = 1.0
            elif down_hit is not None and (up_hit is None or down_hit < up_hit):
                label = -1.0
            else:
                # 同一天同时触及（无法区分先后），保守设为 0
                label = 0.0

        labels[i] = label

    return pd.Series(labels, index=df.index, name="label")


def load_stock_daily_from_db(
    code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """从 SQLite 的 stock_daily 表加载指定个股的日线数据。"""
    if get_db_connection is None:
        raise RuntimeError("无法导入 quant_system.data.storage.get_db_connection，"
                           "请确认在项目根目录下运行，且 quant_system 可被 Python 导入。")

    norm_code = normalize_stock_code(code)
    conn = get_db_connection()
    try:
        sql = (
            "SELECT trade_date, code, open, high, low, close, preclose, volume, amount, pct_chg "
            "FROM stock_daily WHERE code = ?"
        )
        params: List[str] = [norm_code]
        if start_date:
            sql += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND trade_date <= ?"
            params.append(end_date)
        sql += " ORDER BY trade_date"

        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        print(f"[feature_engineering] stock_daily 中没有找到代码 {norm_code} 的数据。")
        return df

    return df


def build_features_for_stock(
    code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cfg: TripleBarrierConfig = TripleBarrierConfig(),
    save: bool = True,
) -> pd.DataFrame:
    """针对单只股票生成特征 + 标签，并（可选）写入 Parquet 文件。

    返回值：包含所有特征列 + 'label' 列的 DataFrame，按 trade_date 升序。
    """
    raw_df = load_stock_daily_from_db(code, start_date=start_date, end_date=end_date)
    if raw_df is None or raw_df.empty:
        return raw_df

    feat_df = compute_technical_features(raw_df)
    label_series = apply_triple_barrier(feat_df, cfg=cfg)
    feat_df["label"] = label_series

    # 丢弃近 horizon 天内无法完整打标签的样本（label 为 NaN）
    feat_df = feat_df[feat_df["label"].notna()].reset_index(drop=True)

    if save:
        norm_code = normalize_stock_code(code)
        out_path = FEATURE_DIR / f"features_{norm_code.replace('.', '_')}.parquet"
        try:
            import polars as pl  # type: ignore

            pl.from_pandas(feat_df).write_parquet(out_path)
            print(f"[feature_engineering] 已写入特征文件：{out_path}")
        except Exception as exc:  # pragma: no cover - 写入失败时降级为 CSV
            csv_path = out_path.with_suffix(".csv")
            feat_df.to_csv(csv_path, index=False)
            print(f"[feature_engineering] 写入 Parquet 失败，改为写入 CSV：{csv_path}，错误：{exc}")

    return feat_df


# ---------------------------------------------------------------------------
# CLI 入口：命令行 / 定时任务
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="生成个股五日走势预测特征数据。")
    parser.add_argument("--code", type=str, required=True, help="股票代码，如 600519 或 sh.600519")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD，默认使用数据库最早日期")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD，默认使用数据库最新日期")
    parser.add_argument("--horizon", type=int, default=5, help="三路屏障观察窗口，默认 5 日")
    parser.add_argument("--up-mult", type=float, default=1.5, help="上轨波动率倍数 μ，默认 1.5")
    parser.add_argument("--down-mult", type=float, default=1.5, help="下轨波动率倍数 λ，默认 1.5")
    parser.add_argument("--no-save", action="store_true", help="只生成 DataFrame 不落盘")

    args = parser.parse_args()

    cfg = TripleBarrierConfig(
        horizon=args.horizon,
        up_mult=args.up_mult,
        down_mult=args.down_mult,
    )

    build_features_for_stock(
        code=args.code,
        start_date=args.start,
        end_date=args.end,
        cfg=cfg,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
