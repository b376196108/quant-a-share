"""
简单测试 backend.tft_model.utils.prepare_tft_datasets 是否能正常导入和运行。
在项目根目录运行：
    python scripts/test_prepare_tft.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把“项目根目录”(quant-a-share) 手动加到 sys.path 里
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.tft_model.utils import prepare_tft_datasets


def main() -> None:
    # 用你已经生成过特征的那只股票代码，这里先用 600519 示范
    codes = ["600519"]

    print("开始构建 TFT 数据集，codes =", codes)
    bundle = prepare_tft_datasets(codes)

    print("=== 数据集长度检查 ===")
    print("training len:", len(bundle.training))
    print("validation len:", len(bundle.validation))

    # 尝试从训练 dataloader 里取一些结构信息（有就打印，没有就打印 ok）
    try:
        train_dl = bundle.train_dataloader
        ds = train_dl.dataset
        x_names = getattr(ds, "x_names", None)
        print("=== sample batch x keys ===")
        print(x_names if x_names is not None else "ok")
    except Exception as e:
        print("读取 dataloader 结构时出错（不一定是致命问题）：", repr(e))


if __name__ == "__main__":
    main()
