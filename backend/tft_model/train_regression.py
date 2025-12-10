"""Train a TFT regression model to predict future 5-day closing prices (quantile regression)."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from .utils import FEATURE_DIR, load_features_for_codes, normalize_stock_code

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    codes: List[str]
    target: str = "close"
    max_encoder_length: int = 60
    max_prediction_length: int = 5
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 32
    attention_head_size: int = 4
    hidden_continuous_size: int = 16
    dropout: float = 0.1
    gradient_clip_val: float = 0.1
    seed: int = 42


def _parse_codes_arg(codes_str: str) -> List[str]:
    if codes_str.lower() == "all":
        codes: List[str] = []
        for path in FEATURE_DIR.glob("features_*.parquet"):
            code = path.stem.replace("features_", "").replace("_", ".")
            codes.append(code)
        if not codes:
            for path in FEATURE_DIR.glob("features_*.csv"):
                code = path.stem.replace("features_", "").replace("_", ".")
                codes.append(code)
        if not codes:
            raise FileNotFoundError("未找到任何特征文件，请先运行特征工程生成。")
        return codes

    codes = [normalize_stock_code(c.strip()) for c in codes_str.split(",") if c.strip()]
    if not codes:
        raise ValueError("参数 --codes 为空，请提供股票代码或使用 all。")
    return codes


def build_regression_datasets(df: pd.DataFrame, cfg: TrainConfig):
    """
    基于特征表 df 构造用于“预测 future 5 日 close”的 TimeSeriesDataSet。
    """
    df = df.copy()
    df = df.sort_values(["code", "trade_date"])

    df.replace([pd.NA, float("inf"), float("-inf")], pd.NA, inplace=True)
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    df = df.dropna(subset=[cfg.target] + numeric_cols)
    if df.empty:
        raise RuntimeError("回归训练：清理缺失后数据为空。")

    if "time_idx" not in df.columns:
        df["time_idx"] = (
            df.groupby("code")["trade_date"]
            .rank(method="first")
            .astype(int)
            - 1
        )

    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    for col in ["label", "label_class"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    min_len = cfg.max_encoder_length + cfg.max_prediction_length
    counts = df.groupby("code")["time_idx"].count()
    valid_codes = counts[counts >= min_len].index.tolist()
    df = df[df["code"].isin(valid_codes)]
    if df.empty:
        raise RuntimeError("回归训练：满足长度要求的样本为空，请检查特征数据或缩短窗口。")

    target = cfg.target

    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target,
        group_ids=["code"],
        max_encoder_length=cfg.max_encoder_length,
        max_prediction_length=cfg.max_prediction_length,
        static_categoricals=["code"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=numeric_cols,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=cfg.batch_size,
        num_workers=0,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    return training, train_dataloader, val_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="TFT 回归：预测未来 5 日收盘价（分位数回归）")
    parser.add_argument(
        "--codes",
        type=str,
        default="all",
        help="用于训练的股票代码列表，逗号分隔；默认 all 表示目录下全部特征文件",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("=== 回归版 TFT 训练：target=close，预测长度=5 ===")

    codes = _parse_codes_arg(args.codes)
    cfg = TrainConfig(codes=codes)

    seed_everything(cfg.seed, workers=True)

    df = load_features_for_codes(cfg.codes, regenerate=False)
    if df is None or df.empty:
        raise RuntimeError("回归训练：未加载到特征数据，请先运行特征工程。")

    logger.info("raw feature df shape = %s", df.shape)

    training, train_dataloader, val_dataloader = build_regression_datasets(df, cfg)

    quantiles = [0.1, 0.5, 0.9]
    loss = QuantileLoss(quantiles=quantiles)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        hidden_continuous_size=cfg.hidden_continuous_size,
        output_size=len(quantiles),
        loss=loss,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    logger.info("模型参数量: %d", sum(p.numel() for p in tft.parameters()))

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(MODEL_DIR),
        enable_model_summary=True,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    ckpt_name = "tft_reg_close_q10_50_90.ckpt"
    final_path = MODEL_DIR / ckpt_name
    trainer.save_checkpoint(str(final_path))
    logger.info("回归模型已保存到: %s", final_path)


if __name__ == "__main__":
    main()
