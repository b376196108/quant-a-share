"""Train script for 5-day stock trend prediction using Temporal Fusion Transformer (TFT).

功能概述
--------
1. 从 backend/tft_model/utils.py 加载特征数据，构建 TimeSeriesDataSet。
2. 使用 PyTorch Forecasting 的 TemporalFusionTransformer 训练分类模型：
   - 目标为 label_class ∈ {0,1,2}，对应 {空头(-1)、中性(0)、多头(1)}。
   - 损失函数使用 CrossEntropy（多分类交叉熵）。
3. 支持单只或多只股票联合训练（逗号分隔的代码列表）。
4. 自动检测是否有 GPU，有则使用 GPU；否则使用 CPU。
5. 训练完成后，将最优模型 checkpoint 保存到 backend/tft_model/models/ 目录。

使用方式
--------
在项目根目录下运行（示例）：

    python -m backend.tft_model.train --codes 600519,000001 --max-epochs 20

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import List

import torch

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy

# 兼容 lightning / pytorch_lightning 两套入口
try:  # 优先使用新版 lightning
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except Exception:  # 回退到 pytorch_lightning
    from pytorch_lightning import Trainer, seed_everything  # type: ignore
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

from .utils import (
    N_CLASSES,
    DatasetBundle,
    prepare_tft_datasets,
)


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 配置数据结构
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    codes: List[str]
    max_encoder_length: int = 60
    max_prediction_length: int = 5
    valid_ratio: float = 0.2
    max_epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 32
    attention_head_size: int = 4
    dropout: float = 0.1
    gradient_clip_val: float = 0.1
    seed: int = 42


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _parse_codes_arg(codes_str: str) -> List[str]:
    codes = [c.strip() for c in codes_str.split(",") if c.strip()]
    if not codes:
        raise ValueError("参数 --codes 解析后为空，请至少提供一个股票代码，例如 600519 或 600519,000001")
    return codes


def _build_trainer(cfg: TrainConfig) -> Trainer:
    """根据当前环境（是否有 GPU）与 Lightning 版本构建 Trainer。"""

    use_gpu = torch.cuda.is_available()
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=5,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=MODELS_DIR,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=callbacks,
        enable_model_summary=True,
        logger=False,  # 若后续需要可接入 TensorBoard / Neptune 等
    )

    # 兼容 lightning 2.x (accelerator/devices) 与旧版 pytorch_lightning (gpus)
    init_sig = signature(Trainer.__init__)
    if "accelerator" in init_sig.parameters:
        # 新版接口
        trainer_kwargs.update(
            accelerator="gpu" if use_gpu else "cpu",
            devices=1 if use_gpu else 1,
        )
    else:
        # 旧版接口
        trainer_kwargs.update(
            gpus=1 if use_gpu else 0,
        )

    trainer = Trainer(**trainer_kwargs)
    return trainer


def _build_model(bundle: DatasetBundle, cfg: TrainConfig) -> TemporalFusionTransformer:
    """基于训练数据集构建 TFT 模型实例。"""

    training = bundle.training

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        loss=CrossEntropy(),        # 多分类交叉熵
        output_size=N_CLASSES,      # 分类类别数：3
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    return tft


# ---------------------------------------------------------------------------
# 训练主流程
# ---------------------------------------------------------------------------


def train_tft(cfg: TrainConfig) -> Path:
    """根据配置训练 TFT 模型，并返回最优 checkpoint 路径。"""

    print(f"[train] 使用股票代码：{cfg.codes}")
    print(f"[train] max_encoder_length = {cfg.max_encoder_length}, "
          f"max_prediction_length = {cfg.max_prediction_length}, "
          f"valid_ratio = {cfg.valid_ratio}")
    print(f"[train] max_epochs = {cfg.max_epochs}, lr = {cfg.learning_rate}, "
          f"hidden_size = {cfg.hidden_size}, attention_head_size = {cfg.attention_head_size}")

    seed_everything(cfg.seed, workers=True)

    # 1. 准备数据集与 DataLoader
    bundle = prepare_tft_datasets(
        codes=cfg.codes,
        max_encoder_length=cfg.max_encoder_length,
        max_prediction_length=cfg.max_prediction_length,
        valid_ratio=cfg.valid_ratio,
    )
    print(f"[train] 训练样本数: {len(bundle.training)}, 验证样本数: {len(bundle.validation)}")

    # 2. 构建模型与 Trainer
    model = _build_model(bundle, cfg)
    trainer = _build_trainer(cfg)

    # 3. 训练
    trainer.fit(
        model,
        train_dataloaders=bundle.train_dataloader,
        val_dataloaders=bundle.val_dataloader,
    )

    # 4. 取得最佳 checkpoint 路径
    ckpt_callback = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_callback = cb
            break

    best_path: Path
    if ckpt_callback is not None and ckpt_callback.best_model_path:
        best_path = Path(ckpt_callback.best_model_path)
    else:
        # 若因某些原因未记录 best_model_path，则保存最后一次权重
        best_path = MODELS_DIR / "tft-last.ckpt"
        trainer.save_checkpoint(best_path)

    print(f"[train] 最优模型已保存到：{best_path}")

    # 可选：再额外保存一份简化版本（例如 state_dict），以后如需可再扩展
    # state_path = MODELS_DIR / "stock_tft_state_dict.pt"
    # torch.save(model.state_dict(), state_path)
    # print(f"[train] 额外保存 state_dict 到：{state_path}")

    return best_path


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="训练个股五日走势预测 TFT 模型。")
    parser.add_argument(
        "--codes",
        type=str,
        required=True,
        help="训练用股票代码列表，逗号分隔，例如：600519,000001 或 sh.600519,sz.000001",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="最大训练轮数，默认 20",
    )
    parser.add_argument(
        "--max-encoder-length",
        type=int,
        default=60,
        help="编码器长度（回看天数），默认 60",
    )
    parser.add_argument(
        "--max-prediction-length",
        type=int,
        default=5,
        help="预测长度（未来天数），默认 5，与标签 horizon 保持一致",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="验证集在 time_idx 维度上的比例，默认 0.2",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率，默认 1e-3",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="TFT 隐藏层维度，默认 32",
    )
    parser.add_argument(
        "--attention-head-size",
        type=int,
        default=4,
        help="注意力头数量，默认 4",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout 比例，默认 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )

    args = parser.parse_args()

    codes = _parse_codes_arg(args.codes)

    cfg = TrainConfig(
        codes=codes,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        valid_ratio=args.valid_ratio,
        max_epochs=args.max_epochs,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        seed=args.seed,
    )

    train_tft(cfg)


if __name__ == "__main__":
    main()
