"""Utility functions for TFT-based 5-day stock trend prediction.

本模块主要提供：
1. 从 backend/tft_model/data 中加载特征文件（Parquet/CSV）。
2. 组装成 PyTorch Forecasting 所需的 TimeSeriesDataSet。
3. 构造训练 / 验证 DataLoader。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder


# ---------------------------------------------------------------------------
# 常量与路径
# ---------------------------------------------------------------------------

N_CLASSES: int = 3  # 标签取值 {-1,0,1} 映射为 {0,1,2}

THIS_DIR = Path(__file__).resolve().parent           # backend/tft_model
BACKEND_DIR = THIS_DIR.parent                        # backend
PROJECT_ROOT = BACKEND_DIR.parent                    # 项目根目录
FEATURE_DIR = THIS_DIR / "data"                      # 特征文件目录
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 工具函数：代码规范化 & 加载特征
# ---------------------------------------------------------------------------


def normalize_stock_code(code: str) -> str:
    """与 feature_engineering 中保持一致的代码规范函数。

    约定：
    - 已包含 '.' 的视为完整代码（sh.600519），转换为小写。
    - 6 开头视为上证：sh.XXXXXX
    - 0 或 3 开头视为深证：sz.XXXXXX
    其他情况原样返回。
    """
    code = code.strip()
    if "." in code:
        return code.lower()

    if len(code) == 6:
        if code.startswith("6"):
            return f"sh.{code}"
        elif code.startswith(("0", "3")):
            return f"sz.{code}"

    return code


def _feature_file_paths(code: str) -> Tuple[Path, Path]:
    """返回某只股票特征文件的 Parquet / CSV 路径（都可能存在）。"""
    norm_code = normalize_stock_code(code)
    base = f"features_{norm_code.replace('.', '_')}"
    parquet_path = FEATURE_DIR / f"{base}.parquet"
    csv_path = FEATURE_DIR / f"{base}.csv"
    return parquet_path, csv_path


def load_feature_dataframe(code: str, regenerate: bool = False) -> pd.DataFrame:
    """加载单只股票的特征 DataFrame。

    默认优先读取 Parquet，如不存在则尝试 CSV。
    如果均不存在且 regenerate=True，则调用特征工程脚本动态生成。
    """
    pq_path, csv_path = _feature_file_paths(code)

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        if not regenerate:
            raise FileNotFoundError(
                f"未找到股票 {code} 的特征文件，请先运行 "
                "backend/data_pipeline/feature_engineering.py 生成特征。"
            )
        # 动态生成特征
        from backend.data_pipeline.feature_engineering import build_features_for_stock

        df = build_features_for_stock(code)
        # build_features_for_stock 本身会写入 Parquet/CSV，这里直接返回

    if df.empty:
        raise ValueError(f"股票 {code} 特征数据为空，无法训练 / 预测。")

    # 确保 trade_date 为 datetime
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 统一增加 code 列（group id）
    df["code"] = normalize_stock_code(code)

    return df


def load_features_for_codes(codes: Iterable[str], regenerate: bool = False) -> pd.DataFrame:
    """加载多只股票的特征，并合并为一个 DataFrame。

    参数
    ----
    codes : 可迭代对象，包含若干股票代码（6 位或带前缀）
    regenerate : 如果特征文件缺失，是否自动调用特征工程生成

    返回
    ----
    df_all : 包含所有股票、按 trade_date 排序的 DataFrame
    """
    frames: List[pd.DataFrame] = []
    for code in codes:
        df = load_feature_dataframe(code, regenerate=regenerate)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values(["code", "trade_date"]).reset_index(drop=True)
    return df_all


# ---------------------------------------------------------------------------
# 构建 TimeSeriesDataSet & DataLoader
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    """打包训练/验证数据集与 DataLoader，方便后续使用。"""

    training: TimeSeriesDataSet
    validation: TimeSeriesDataSet
    train_dataloader: DataLoader
    val_dataloader: DataLoader


def prepare_tft_datasets(
    codes: Iterable[str],
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
    valid_ratio: float = 0.2,
) -> DatasetBundle:
    """针对给定的股票集合，构建 TFT 所需的 TimeSeriesDataSet 与 DataLoader。

    约定：
    - 目标列为 label，取值 {-1,0,1}，转换为 label_class∈{0,1,2} 作为分类目标。
    - group_ids 使用标准化后的 code 列。
    - time_idx 为每只股票内部从 0 递增的整数。
    """
    if valid_ratio <= 0 or valid_ratio >= 0.5:
        raise ValueError("valid_ratio 建议在 (0, 0.5) 区间内，例如 0.2")

    df = load_features_for_codes(codes, regenerate=False)

    # 丢弃缺 label 的样本
    if "label" not in df.columns:
        raise KeyError("特征数据中未找到 'label' 列，请确认特征工程脚本已正确生成标签。")

    df = df[df["label"].notna()].copy()

    # ---- NEW: 清理数值特征中的 NaN / Inf，避免 TimeSeriesDataSet 报错 ----
    # 将所有无限值先转为 NaN，然后对于数值型列，如果有 NaN 就直接丢弃对应样本。
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_all:
        df = df.dropna(subset=numeric_all)
    if df.empty:
        raise RuntimeError(
            "清理缺失值后数据为空，请检查特征工程是否生成了过多 NaN。"
        )
    # ---- NEW END ----

    # 生成时间索引：对每只股票单独排序并编号
    df = df.sort_values(["code", "trade_date"])
    df["time_idx"] = (
        df.groupby("code")["trade_date"]
        .rank(method="first")
        .astype(int)
        - 1
    )

    # 映射标签到 0/1/2
    label_map = {-1.0: 0, -1: 0, 0.0: 1, 0: 1, 1.0: 2, 1: 2}
    df["label_class"] = df["label"].map(label_map)
    if df["label_class"].isna().any():
        unknown = df.loc[df["label_class"].isna(), "label"].unique()
        raise ValueError(f"发现未知标签值 {unknown}，只支持 -1/0/1")

    df["label_class"] = df["label_class"].astype(int)

    # 选择数值型特征列（排除目标与 time_idx）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 移除 label, label_class, time_idx 本身
    for col in ["label", "label_class", "time_idx"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    time_varying_unknown_reals = numeric_cols
    time_varying_known_reals = ["time_idx"]  # time_idx 视为已知未来的时间特征

    # 训练 / 验证划分（基于 time_idx，全局切分）
    max_time_idx = df["time_idx"].max()
    training_cutoff = int(max_time_idx * (1.0 - valid_ratio))

    training_df = df[df["time_idx"] <= training_cutoff]
    validation_df = df[df["time_idx"] > training_cutoff]

    if len(training_df) == 0 or len(validation_df) == 0:
        raise RuntimeError(
            "训练/验证集划分结果为空，请检查样本数量是否足够，或调整 valid_ratio。"
        )

    # 构建 TimeSeriesDataSet
    training = TimeSeriesDataSet(
        training_df,
        time_idx="time_idx",
        target="label_class",
        group_ids=["code"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["code"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        categorical_encoders={"code": NaNLabelEncoder().fit(df["code"])},
        target_normalizer=None,  # 分类任务不需要归一化
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_df,
        stop_randomization=True,
    )

    # DataLoader
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    return DatasetBundle(
        training=training,
        validation=validation,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )


__all__ = [
    "N_CLASSES",
    "DatasetBundle",
    "prepare_tft_datasets",
    "load_feature_dataframe",
    "load_features_for_codes",
]
