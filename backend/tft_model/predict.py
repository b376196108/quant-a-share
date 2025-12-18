from __future__ import annotations

import datetime as dt
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    import baostock as bs
except ImportError:  # pragma: no cover
    bs = None
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from backend.data_pipeline.feature_engineering import build_features_for_stock

from .utils import load_feature_dataframe, normalize_stock_code

logger = logging.getLogger(__name__)
THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 5
REG_CKPT_PATH = (MODELS_DIR / "tft_reg_close_q10_50_90.ckpt").as_posix()


def _ensure_feature_dataframe(ts_code: str) -> pd.DataFrame:
    """
    尝试加载特征，缺失时自动运行特征工程生成。
    """
    norm_code = normalize_stock_code(ts_code)
    return load_feature_dataframe(norm_code, regenerate=True)


def _normalize_ts_code(ts_code: str) -> str:
    """
    将前端传入的股票代码统一转换为 baostock 使用的格式。

    支持：
    - "600519"               -> "sh.600519"
    - "000001"               -> "sz.000001"
    - "600519.SH"/"000001.SZ" -> "sh.600519"/"sz.000001"
    - 已经是 "sh.600519"/"sz.000001" 则原样返回。
    """
    code = ts_code.strip()
    if not code:
        raise ValueError("股票代码不能为空")

    code = code.lower()

    # 已经是 baostock 形式
    if code.startswith("sh.") or code.startswith("sz."):
        return code

    # 形如 600519.sh / 000001.sz
    if "." in code:
        body, suffix = code.split(".", 1)
        suffix = suffix.lower()
        if suffix in ("sh", "sz"):
            return f"{suffix}.{body}"
        # 其它后缀先直接返回原始字符串
        return code

    # 纯 6 位数字
    if len(code) == 6 and code.isdigit():
        if code.startswith("6"):
            return f"sh.{code}"
        else:
            return f"sz.{code}"

    # 兜底：直接返回
    return code


@lru_cache(maxsize=16)
def _fetch_recent_daily(
    bs_code: str,
    end_date: Optional[str] = None,
    lookback_days: int = 40,
) -> pd.DataFrame:
    """
    通过 baostock 获取最近一段时间的日线数据。

    当前版本只是作为 baseline 模型的数据来源：
    - 后续如果有本地特征仓库（Parquet/SQLite），只要改这里的实现即可。
    """
    if bs is None:
        raise ModuleNotFoundError("缺少依赖 baostock，请先安装：pip install baostock")

    if end_date is None:
        end = dt.date.today()
    else:
        end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

    start = end - dt.timedelta(days=lookback_days * 2)

    logger.info("fetching baostock data: code=%s, %s -> %s", bs_code, start, end)

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",  # 前复权
        )

        data_list = []
        while rs.error_code == "0" and rs.next():
            data_list.append(rs.get_row_data())
    finally:
        bs.logout()

    if not data_list:
        raise ValueError(f"未从 baostock 获取到 {bs_code} 的日线数据")

    df = pd.DataFrame(
        data_list,
        columns=["date", "code", "open", "high", "low", "close", "volume", "amount"],
    )

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)

    # 只保留最近 lookback_days 天，避免数据量太大
    if len(df) > lookback_days:
        df = df.iloc[-lookback_days:]

    return df


def _select_best_checkpoint() -> Optional[Path]:
    """Pick the best available TFT checkpoint by val_loss filename."""
    if not MODELS_DIR.exists():
        return None

    ckpts = sorted(MODELS_DIR.glob("tft-epoch=*-val_loss=*.ckpt"))
    if ckpts:
        return ckpts[0]

    # fallback: any ckpt file
    generic = sorted(MODELS_DIR.glob("*.ckpt"))
    return generic[0] if generic else None


@lru_cache(maxsize=1)
def _load_tft_model() -> Optional[TemporalFusionTransformer]:
    ckpt_path = _select_best_checkpoint()
    if ckpt_path is None:
        logger.warning("no TFT checkpoint found in %s", MODELS_DIR)
        return None

    try:
        logger.info("loading TFT checkpoint: %s", ckpt_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TemporalFusionTransformer.load_from_checkpoint(
            ckpt_path,
            map_location=device,
        )
        model.eval()
        return model
    except Exception:
        logger.exception("failed to load TFT checkpoint: %s", ckpt_path)
        return None


def _build_inference_dataset(
    ts_code: str,
    max_encoder_length: int = MAX_ENCODER_LENGTH,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
) -> Tuple[TimeSeriesDataSet, float]:
    """
    Prepare a minimal TimeSeriesDataSet for single-stock inference.
    Returns dataset and the latest close price for downstream forecast mapping.
    """
    df = _ensure_feature_dataframe(ts_code)
    if "trade_date" not in df.columns:
        raise ValueError("feature dataframe missing 'trade_date'")

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["time_idx"] = np.arange(len(df), dtype=int)

    # label_class mapping for available labels
    label_map = {-1.0: 0, -1: 0, 0.0: 1, 0: 1, 1.0: 2, 1: 2}
    if "label_class" not in df.columns:
        df["label_class"] = df["label"].map(label_map)
    else:
        df["label_class"] = df["label_class"].fillna(df["label"].map(label_map))

    # 丢弃缺失标签的行，避免预测时报 NA 错误
    df = df.dropna(subset=["label_class"])
    if df.empty:
        raise ValueError(f"{ts_code} 特征数据缺少有效 label_class，无法推理")

    last_close = float(df["close"].iloc[-1])

    # keep last encoder window
    if len(df) > max_encoder_length:
        df = df.iloc[-max_encoder_length:].copy()

    last_row = df.iloc[-1].copy()
    last_time_idx = int(last_row["time_idx"])
    last_trade_date = last_row["trade_date"]

    decoder_rows: List[pd.Series] = []
    for i in range(1, max_prediction_length + 1):
        new_row = last_row.copy()
        new_row["time_idx"] = last_time_idx + i
        new_row["trade_date"] = last_trade_date + pd.Timedelta(days=i)
        new_row["label"] = np.nan
        new_row["label_class"] = np.nan
        decoder_rows.append(new_row)

    if decoder_rows:
        df = pd.concat([df, pd.DataFrame(decoder_rows)], ignore_index=True)

    # clean NaN/Inf for features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["label", "label_class", "time_idx"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    time_varying_unknown_reals = numeric_cols
    time_varying_known_reals = ["time_idx"]

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="label_class",
        group_ids=["code"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["code"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        categorical_encoders={"code": NaNLabelEncoder().fit(df["code"])},
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

    return dataset, last_close


def _tft_predict_direction(ts_code: str) -> Optional[Tuple[str, float]]:
    """
    Run TFT to get direction classification (buy/neutral/sell) with confidence.
    Returns None if TFT is unavailable or fails.
    """
    # 确保特征文件存在（缺失/为空则动态生成）；失败则回退 baseline
    try:
        _ensure_feature_dataframe(ts_code)
    except Exception:
        logger.exception("failed to ensure feature dataframe for %s", ts_code)
        return None

    model = _load_tft_model()
    if model is None:
        return None

    try:
        dataset, _ = _build_inference_dataset(ts_code)
    except Exception:
        logger.exception("failed to prepare TFT inference dataset for %s", ts_code)
        return None

    loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    try:
        with torch.no_grad():
            raw = model.predict(loader, mode="raw")
    except Exception:
        logger.exception("TFT predict failed for %s", ts_code)
        return None

    # raw["prediction"] shape: [B, max_prediction_length, n_classes]
    preds = raw["prediction"] if isinstance(raw, dict) else raw
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    arr = np.asarray(preds)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        logits = arr[-1]  # use the last prediction step
    elif arr.ndim == 1:
        logits = arr
    else:
        logger.warning("unexpected TFT prediction shape %s", arr.shape)
        return None

    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_t, dim=-1).cpu().numpy()
    idx = int(np.argmax(probs))
    confidence = float(np.round(probs[idx], 4))

    label_map = {0: "卖出", 1: "中性", 2: "买入"}
    signal = label_map.get(idx, "中性")

    logger.info("TFT direction: code=%s, signal=%s, conf=%.4f", ts_code, signal, confidence)
    return signal, confidence


@lru_cache(maxsize=1)
def _load_tft_regression_model() -> TemporalFusionTransformer:
    """
    加载回归版 TFT 模型（预测收盘价分位数）。
    """
    if not os.path.exists(REG_CKPT_PATH):
        raise FileNotFoundError(f"回归模型 checkpoint 不存在: {REG_CKPT_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalFusionTransformer.load_from_checkpoint(
        REG_CKPT_PATH,
        map_location=device,
    )
    model.eval()
    return model


def _build_regression_dataset_for_code(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
) -> TimeSeriesDataSet:
    """
    基于单只股票的特征 df 构造 TimeSeriesDataSet，用于回归预测 close。
    """
    df = df.copy().sort_values(["code", "trade_date"])

    if "time_idx" not in df.columns:
        df["time_idx"] = (
            df.groupby("code")["trade_date"]
            .rank(method="first")
            .astype(int)
            - 1
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in ["label", "label_class"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="close",
        group_ids=["code"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=numeric_cols,
        time_varying_known_reals=["time_idx"],
        static_categoricals=["code"],
    )


def tft_price_forecast(
    ts_code: str,
    date: Optional[str] = None,
    horizon: int = 5,
) -> List[Dict[str, float]]:
    """
    使用回归版 TFT 模型，预测未来 horizon 天的收盘价路径（中位数 + 上/下分位）。
    """
    df_all = _ensure_feature_dataframe(ts_code)

    df_all = df_all.sort_values(["code", "trade_date"])

    if date is not None:
        cutoff = pd.to_datetime(date)
        df_all = df_all[df_all["trade_date"] <= cutoff]

    if len(df_all) < 80:
        raise ValueError("历史特征样本不足，无法进行回归预测")

    max_encoder_length = MAX_ENCODER_LENGTH
    max_prediction_length = horizon
    df_recent = df_all.iloc[-(max_encoder_length + max_prediction_length) :].copy()

    # 再次清理数值列的 NaN/Inf，避免推理报错
    df_recent.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df_recent.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df_recent = df_recent.dropna(subset=numeric_cols)
    if df_recent.empty:
        raise ValueError("回归预测：清理缺失后数据为空")

    # from_dataset 需要 time_idx 列；_build_regression_dataset_for_code 内部会生成 time_idx，
    # 但不会回写到 df_recent，因此这里提前补齐以避免 KeyError: 'time_idx'
    if "time_idx" not in df_recent.columns:
        df_recent = df_recent.sort_values(["code", "trade_date"]).copy()
        df_recent["time_idx"] = (
            df_recent.groupby("code")["trade_date"].rank(method="first").astype(int) - 1
        )

    training_ds = _build_regression_dataset_for_code(
        df_recent,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )

    pred_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df_recent,
        predict=True,
        stop_randomization=True,
    )

    pred_loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    model = _load_tft_regression_model()

    quantiles = [0.1, 0.5, 0.9]
    with torch.no_grad():
        preds = model.predict(pred_loader, mode="quantiles", quantiles=quantiles)

    if isinstance(preds, dict):
        preds = preds.get("prediction", preds)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    else:
        preds = np.asarray(preds)

    preds = np.asarray(preds)
    if preds.ndim == 3:
        preds = preds[0]
    elif preds.ndim != 2:
        raise ValueError(f"unexpected regression prediction shape: {preds.shape}")

    last_date = pd.to_datetime(df_recent["trade_date"].max()).date()
    future_dates = _next_business_days(last_date, horizon)
    results: List[Dict[str, float]] = []
    for i, d in enumerate(future_dates):
        p10, p50, p90 = preds[i].tolist()
        results.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "predicted_close": float(p50),
                "lower": float(p10),
                "upper": float(p90),
            }
        )

    return results


def get_latest_price(ts_code: str, date: Optional[str] = None) -> Tuple[str, float]:
    """
    获取指定股票最近一个交易日的日期和收盘价。

    返回:
        (last_date_str, last_close)
        last_date_str: 'YYYY-MM-DD'
        last_close: float
    """
    bs_code = _normalize_ts_code(ts_code)
    df = _fetch_recent_daily(bs_code, end_date=date)

    if df.empty:
        raise ValueError(f"未获取到 {ts_code} 的日线数据")

    last_row = df.iloc[-1]
    last_date = pd.to_datetime(last_row["date"]).date()
    last_close = float(last_row["close"])

    return last_date.strftime("%Y-%m-%d"), last_close


def _simple_5d_direction_baseline(
    ts_code: str,
    date: Optional[str] = None,
) -> Tuple[str, float]:
    """
    一个非常简单的 5 日方向 baseline，用来先打通前后端流程。

    逻辑（注意：这是一个研究/演示用 baseline，将来会被 TFT 模型替换）：
    - 使用 baostock 拉取最近一段时间日线；
    - 取最后 6 根收盘价，计算从 t-5 到 t 的 5 日收益；
    - 收益 > +2% 判定为「买入」，收益 < -2% 判定为「卖出」，其余为「中性」；
    - 将收益幅度映射为 [0.5, 0.95] 区间的置信度。
    """
    bs_code = _normalize_ts_code(ts_code)
    df = _fetch_recent_daily(bs_code, end_date=date)

    if len(df) < 6:
        raise ValueError("可用历史数据不足 6 个交易日，无法计算 5 日收益")

    closes = df["close"].to_numpy()
    # 最近 t 日和 t-5 日
    p_t = closes[-1]
    p_t5 = closes[-6]
    ret_5d = p_t / p_t5 - 1.0

    upper_th = 0.02   # +2%
    lower_th = -0.02  # -2%

    if ret_5d > upper_th:
        signal = "买入"
    elif ret_5d < lower_th:
        signal = "卖出"
    else:
        signal = "中性"

    # 将绝对收益映射到 [0.5, 0.95] 之间作为置信度
    max_ref = 0.10  # 10% 以上就视为“极强”
    strength = min(abs(ret_5d) / max_ref, 1.0)
    confidence = 0.5 + 0.45 * strength
    confidence = float(np.round(confidence, 4))

    logger.info(
        "baseline prediction: code=%s, ret_5d=%.4f, signal=%s, confidence=%.4f",
        ts_code,
        ret_5d,
        signal,
        confidence,
    )

    return signal, confidence


def get_recent_history(
    ts_code: str,
    end_date: Optional[str] = None,
    lookback_days: int = 90,
) -> List[dict]:
    """获取最近一段时间的历史 K 线数据，便于前端绘图。"""
    def _safe_float(val: object, default: float = 0.0) -> float:
        try:
            f = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        if not np.isfinite(f):
            return default
        return f

    norm_code = normalize_stock_code(ts_code)
    bs_code = _normalize_ts_code(norm_code)
    df = _fetch_recent_daily(bs_code, end_date=end_date, lookback_days=lookback_days)

    df = df.sort_values("date").reset_index(drop=True)
    df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()

    history: List[dict] = []
    for _, row in df.iterrows():
        history.append(
            {
                "time": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "open": _safe_float(row.get("open")),
                "high": _safe_float(row.get("high")),
                "low": _safe_float(row.get("low")),
                "close": _safe_float(row.get("close")),
                "volume": _safe_float(row.get("volume")),
                "ma5": _safe_float(row.get("ma5")),
                "ma20": _safe_float(row.get("ma20")),
            }
        )
    return history


def _next_business_days(start: dt.date, n: int) -> List[dt.date]:
    """
    Generate the next n business days after start (Mon-Fri).

    Note: this does not account for China A-share holidays.
    """
    if n <= 0:
        return []

    days: List[dt.date] = []
    cursor = start
    while len(days) < n:
        cursor += dt.timedelta(days=1)
        if cursor.weekday() < 5:
            days.append(cursor)
    return days


def baseline_price_forecast(
    last_date: str,
    last_close: float,
    history: List[dict] | None,
    signal: str,
    confidence: float,
    horizon: int = 5,
) -> List[Dict[str, float]]:
    """
    A deterministic baseline forecast for close price path.

    Uses recent return statistics + simple signal tilt to avoid a flat path
    (especially for neutral signals) when regression TFT is unavailable.
    """
    horizon = int(horizon) if horizon is not None else 0
    horizon = max(horizon, 1)

    try:
        base_date = dt.datetime.strptime(last_date, "%Y-%m-%d").date()
    except Exception:
        base_date = dt.date.today()

    confidence = float(confidence)
    if not np.isfinite(confidence):
        confidence = 0.5
    confidence = float(np.clip(confidence, 0.0, 1.0))

    closes: List[float] = []
    ma20: float | None = None
    if history:
        try:
            ordered = sorted(history, key=lambda x: x.get("time") or "")
        except Exception:
            ordered = list(history)
        for row in ordered:
            try:
                c = float(row.get("close"))
            except Exception:
                continue
            if np.isfinite(c):
                closes.append(c)
        if ordered:
            try:
                m = float(ordered[-1].get("ma20"))
                if np.isfinite(m) and m > 0:
                    ma20 = m
            except Exception:
                ma20 = None

    if not closes:
        closes = [float(last_close)]

    returns: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev and np.isfinite(prev) and np.isfinite(cur):
            returns.append(cur / prev - 1.0)

    lookback = min(len(returns), 20)
    recent_returns = returns[-lookback:] if lookback else []
    mu = float(np.mean(recent_returns)) if recent_returns else 0.0
    sigma = float(np.std(recent_returns, ddof=1)) if len(recent_returns) > 1 else float(abs(mu))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.012

    # Momentum proxy from the last 5 trading days (if possible)
    if len(closes) >= 6 and closes[-6] != 0:
        ret_5d = closes[-1] / closes[-6] - 1.0
        mom = ret_5d / 5.0
    else:
        mom = mu

    if ma20 is None:
        window = closes[-20:] if len(closes) >= 20 else closes
        ma20 = float(np.mean(window)) if window else float(last_close)

    reversion = 0.0
    if last_close and ma20 and np.isfinite(ma20) and ma20 > 0:
        reversion = (ma20 / float(last_close) - 1.0) / float(horizon)

    signal_tilt = 0.0
    if signal == "买入":
        signal_tilt = 0.0025 * (0.5 + confidence)
    elif signal == "卖出":
        signal_tilt = -0.0025 * (0.5 + confidence)

    # Trend: blend mean, momentum and mild mean-reversion; neutral keeps market drift.
    trend = 0.45 * mu + 0.35 * mom + 0.20 * reversion + signal_tilt
    trend = float(np.clip(trend, -0.03, 0.03))

    # Use recent return pattern as deterministic "wiggle"; lower confidence -> noisier.
    pattern_len = min(len(recent_returns), max(horizon, 5))
    pattern = recent_returns[-pattern_len:] if pattern_len else [0.0]
    noise_weight = 0.25 + (1.0 - confidence) * 0.5

    pred = float(last_close)
    dates = _next_business_days(base_date, horizon)
    results: List[Dict[str, float]] = []

    for i, d in enumerate(dates, start=1):
        base_r = float(pattern[(i - 1) % len(pattern)])
        step_ret = trend + noise_weight * (base_r - mu)
        step_ret = float(np.clip(step_ret, -0.05, 0.05))

        pred = float(pred * (1.0 + step_ret))

        band = max(0.01, float(sigma) * float(np.sqrt(i)) * 1.2)
        band *= 1.0 + (1.0 - confidence) * 1.0
        band = float(np.clip(band, 0.01, 0.20))

        lower = float(pred * (1.0 - band))
        upper = float(pred * (1.0 + band))

        results.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "predicted_close": pred,
                "lower": lower,
                "upper": upper,
            }
        )

    return results


def model_predict(ts_code: str, date: Optional[str] = None) -> Tuple[str, float]:
    """
    对外暴露的统一预测接口。

    当前版本：
    - 优先尝试 TFT 模型推理；
    - 若不可用则回退到简单的 5 日收益率 baseline。
    """
    norm_code = normalize_stock_code(ts_code)

    tft_res = _tft_predict_direction(norm_code)
    if tft_res is not None:
        return tft_res

    return _simple_5d_direction_baseline(norm_code, date)
