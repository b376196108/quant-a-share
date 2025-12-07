QuantMind A股 — Web API（FastAPI）今日已搭建完成接口总结

本文档记录 2025-12-07 当天已经成功搭建并且前端已连接的 Web API 接口状态。
后端服务基于 FastAPI + Uvicorn，数据来源为你本地 SQLite（market_data.sqlite）。

1. 已实现并可正常返回数据的接口

前端 Dashboard 模块中，以下两个接口已经确认联通并正常返回 JSON 数据（从日志中看到 HTTP/1.1 200 OK）。

1.1 /api/overview — 市场总览接口
URL
GET /api/overview

功能

返回指定交易日（或最新交易日）的 A股市场整体情绪，包括：

涨跌家数

涨停 / 跌停

上涨占比

今日情绪得分（类似恐慌/贪婪值）

总成交额

平均涨跌幅、中位数涨跌幅

其他市场指标

该接口基于后端模块 quant_system.processing.market_view.calc_market_overview
（来源见 ✦【文件引用】）


前端使用位置

App.tsx —— Dashboard 页面首次加载时自动从后端拉取并覆盖 mock 数据。

1.2 /api/industry-sentiment — 行业情绪接口
URL
GET /api/industry-sentiment?limit=5

功能

返回最新交易日的 行业涨跌、行业情绪榜单：

行业名称

行业内股票数量

上涨/下跌家数

涨停/跌停家数

平均涨幅、中位数涨幅

成交额

情绪标签（高潮 / 普通 / 冰点）

计算逻辑来自 quant_system.processing.industry_sentiment.calc_industry_sentiment


前端使用位置

MarketStats.tsx —— 展示行业涨跌榜与行业情绪列表。

2. 已确认成功运行（来自 Uvicorn 日志）

你启动服务后日志出现：

INFO: 127.0.0.1:14961 - "GET /api/overview HTTP/1.1" 200 OK
INFO: 127.0.0.1:14961 - "GET /favicon.ico HTTP/1.1" 404 Not Found


说明：

/api/overview 已成功响应

favicon 不存在但不影响任何功能

这和前端 Dashboard 正常渲染一致。

3. 后端核心数据来源模块（自动支撑 Web API）

你的 Web API 已经成功读取了量化系统的数据层模块：

3.1 market_overview 计算函数（API 背后的数据来源）

来源：quant_system.processing.market_view.calc_market_overview
功能来自文件：


功能包括：

总股票数、涨跌家数、涨停跌停数

上涨占比

总成交额

情绪分级

3.2 行业情绪计算函数（API 背后的数据来源）

来源：quant_system.processing.industry_sentiment.calc_industry_sentiment


功能：

行业内涨跌统计

行业情绪标签

自动按情绪降序 + 平均涨幅排序

4. API 当前返回的数据格式（前端已验证）
/api/overview 映射到前端结构：

前端 map 函数 mapOverviewToMarketStats() 会把后端字段转成：

字段	含义
limitUp	涨停数
up	上涨家数
flat	平盘家数（如无则前端自动置 0）
down	下跌家数
limitDown	跌停数
sentimentScore	情绪得分

对应数据源来自 market overview。

/api/industry-sentiment 返回结构：
字段	含义
行业	行业名称
平均涨幅	当日行业涨幅
情绪	行业情绪（高 / 中 / 低）

前端 map 函数 mapIndustryList() 已做兼容处理。

5. 当前 API 已具备的能力范围

已具备：

自动读取 SQLite 数据库最新交易日数据

自动计算市场概览 + 行业情绪

结构化 JSON 输出，可直接用于前端 Dashboard

对无连接后端情况具备前端容错（fallback mock 数据）

这是一个完整的“量化系统 Web 总览 API”。

6. 预计下一步可扩展的接口（建议）

你下一步开发 Web 模块时，可新增：

6.1 指数 K 线接口（替换前端模拟 K 线）
GET /api/index-kline?symbol=sh.000001&start=2020-01-01


数据来源：
quant_system.data.fetcher.get_index_data

6.2 策略单标回测 API
POST /api/backtest/single


输入：策略参数 + 标的 + 区间
输出：净值曲线、交易日志、回撤等信息。