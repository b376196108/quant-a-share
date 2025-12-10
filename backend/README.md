# Backend 模块

用于个股五日 AI 走势预测（TFT + 三路屏障）。

- `data_pipeline/feature_engineering.py`：从日线行情生成特征与标签（包括三路屏障标签），写入 Parquet 或 SQLite。
- `tft_model/train.py`：使用 PyTorch Forecasting 的 Temporal Fusion Transformer 训练五日趋势预测模型。
- `tft_model/predict.py`：加载训练好的模型，对单只股票做未来五日“买入/卖出/中性”预测。
- `tft_model/api.py`：封装成 FastAPI 路由，提供 `/api/predict` 接口给前端调用。

后面我会在这些文件里逐步补充实现。
