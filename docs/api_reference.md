# A股量化分析系统 API 说明文档

> 本文档由 `scripts/generate_api_docs.py` 自动生成（生成时间：2025-12-17 08:59:11）。
> 请勿手工修改本文件，如需更新请修改源码或脚本后重新生成。

---

## 模块 `quant_system.backtest.engine`

单标的日线回测引擎 + 多策略组合封装。

分两层：
1）BacktestEngine / BacktestConfig
    - 保持你原来的实现：输入已经带有 `signal` 列的 DataFrame，输出 BacktestResult。
2）run_single_backtest / combine_signals
    - 负责：
        * 从数据模块取日线行情（get_stock_data）
        * 通过策略注册表创建策略实例（create_strategy）
        * 生成每个策略的信号并按组合方式合成（AND / OR / Voting）
        * 调用 BacktestEngine.run() 完成真实回测


### 函数

#### combine_signals

- 签名：`combine_signals(signals: 'List[pd.Series]', mode: 'str' = 'OR') -> 'pd.Series'`

**说明：**

将多个策略信号按指定模式合成一个最终 signal 序列。

参数：
    signals : 若干个 pd.Series，每个取值一般在 {-1, 0, 1}
    mode    : "AND" / "OR" / "VOTING"

返回：
    combined : pd.Series，index 为日期，name="signal"

#### run_single_backtest

- 签名：`run_single_backtest(symbol: 'str', start_date: 'str', end_date: 'str', strategy_ids: 'List[str]', mode: 'str' = 'OR', strategy_params: 'Optional[Dict[str, Dict[str, Any]]]' = None, initial_cash: 'float' = 100000.0, fee_rate: 'float' = 0.0005, slippage: 'float' = 0.0) -> 'BacktestResult'`

**说明：**

高层封装：单只股票 + 多策略组合回测。

这就是后面 FastAPI /api/backtest/run 可以直接调用的核心函数。

参数：
    symbol          : 股票代码（如 "600519"）
    start_date      : 回测开始日期 "YYYY-MM-DD"
    end_date        : 回测结束日期 "YYYY-MM-DD"
    strategy_ids    : 参与组合的策略 id 列表（如 ["connors_rsi2"]）
    mode            : 组合方式，"AND" / "OR" / "VOTING"
    strategy_params : 每个策略的参数字典，key=策略 id，value=参数 dict，可为 None
    initial_cash    : 初始资金
    fee_rate        : 手续费率（万分之 2.5 就填 0.00025）
    slippage        : 单边滑点（元）

返回：
    BacktestResult（与 BacktestEngine.run 一致）


### 类

#### BacktestConfig

- 定义：`class BacktestConfig`

**说明：**

回测配置。

#### BacktestEngine

- 定义：`class BacktestEngine`

**说明：**

单标的、日线、仅多头的简单回测引擎。

**公开方法：**

- `run(self, df: 'pd.DataFrame') -> 'BacktestResult'`


## 模块 `quant_system.backtest.performance`

回测绩效指标与结果数据结构。


### 函数

#### calc_annual_return

- 签名：`calc_annual_return(returns: 'pd.Series', trading_days: 'int' = 252) -> 'float'`

**说明：**

根据日收益率估算年化收益率。
采用几何收益率：((1+总收益) ** (年度交易日/样本长度) - 1)。

#### calc_max_drawdown

- 签名：`calc_max_drawdown(equity: 'pd.Series') -> 'float'`

**说明：**

计算最大回撤，返回负数（如 -0.25）。

#### calc_sharpe

- 签名：`calc_sharpe(returns: 'pd.Series', risk_free: 'float' = 0.0, trading_days: 'int' = 252) -> 'float'`

**说明：**

计算夏普比率：((平均超额收益) / 标准差) * sqrt(年度交易日)。
risk_free 为年化无风险收益率。


### 类

#### BacktestResult

- 定义：`class BacktestResult`

**说明：**

回测结果数据类。

属性：
    equity_curve：账户总资产曲线。
    returns：日收益率序列。
    trades：逐笔交易记录。
    stats：汇总指标（total_return、annual_return、max_drawdown、sharpe 等）。


## 模块 `quant_system.data.fetcher`

A-share market data fetcher with BaoStock + local caching and SQLite sync.


### 函数

#### fetch_all_stock_daily

- 签名：`fetch_all_stock_daily(start_date: 'str', end_date: 'str') -> 'None'`

**说明：**

拉取全市场 A 股在区间 [start_date, end_date] 的日线数据，并写入 stock_daily。
可用于历史初始化或某个时间段的补数。

#### fetch_all_stock_info

- 签名：`fetch_all_stock_info(day: 'str') -> 'pd.DataFrame'`

**说明：**

Fetch all stock info on a given date, upsert into stock_info, and return DataFrame.

#### fetch_index_daily

- 签名：`fetch_index_daily(start_date: 'str', end_date: 'str', codes: 'Optional[Iterable[str]]' = None) -> 'None'`

**说明：**

拉取指定指数在区间 [start_date, end_date] 的日线数据，并写入 index_daily。

#### get_index_data

- 签名：`get_index_data(code: 'str', start_date: 'str', end_date: 'str', freq: 'str' = 'd', fields: 'Optional[List[str]]' = None) -> 'pd.DataFrame'`

**说明：**

Fetch index daily data (e.g., CSI 300) with the same interface style as get_stock_data.

#### get_stock_data

- 签名：`get_stock_data(code: 'str', start_date: 'str', end_date: 'str', freq: 'str' = 'd', fields: 'Optional[List[str]]' = None, adjust: 'str' = '2') -> 'pd.DataFrame'`

**说明：**

Fetch historical K-line data for a single stock (daily by default) with local CSV cache.

#### update_daily_to_today

- 签名：`update_daily_to_today() -> 'None'`

**说明：**

兼容旧接口：仅更新股票日线到今天。

#### update_index_daily_to_today

- 签名：`update_index_daily_to_today(default_start: 'str' = '2015-01-01', codes: 'Optional[Iterable[str]]' = None) -> 'None'`

**说明：**

将 index_daily 表从当前最新交易日增量更新到今天。
若表为空，则从 default_start 开始全量补数。

#### update_stock_daily_to_today

- 签名：`update_stock_daily_to_today(default_start: 'str' = '2015-01-01') -> 'None'`

**说明：**

将 stock_daily 表从当前最新交易日增量更新到今天。
若表为空，则从 default_start 开始全量补数。


## 模块 `quant_system.data.storage`

Local cache utilities for CSV and SQLite storage.


### 函数

#### get_db_connection

- 签名：`get_db_connection(timeout: 'float' = 30.0) -> 'sqlite3.Connection'`

**说明：**

Return SQLite connection with sensible defaults for concurrent read/write.

- timeout/busy_timeout: give writers time to wait for short-lived readers
- WAL mode: allow reads while writing (needed when API is serving requests)
- synchronous=NORMAL: balance durability and write speed for cache usage

#### get_latest_trade_date

- 签名：`get_latest_trade_date(table: 'str' = 'stock_daily') -> 'Optional[str]'`

**说明：**

获取指定表(stock_daily / index_daily)里的最新交易日字符串 YYYY-MM-DD。
若表为空则返回 None。

#### init_db_schema

- 签名：`init_db_schema() -> 'None'`

**说明：**

Create stock_info, stock_daily, index_daily tables if not exist.

#### load_from_cache

- 签名：`load_from_cache(cache_key: 'str') -> 'Optional[pd.DataFrame]'`

**说明：**

Load CSV by cache_key. Return None when missing.

#### load_index_daily

- 签名：`load_index_daily(codes: 'Iterable[str]', start_date: 'str', end_date: 'str') -> 'pd.DataFrame'`

**说明：**

Load index daily data with MultiIndex (trade_date, code).

#### load_stock_daily

- 签名：`load_stock_daily(codes: 'Iterable[str]', start_date: 'str', end_date: 'str') -> 'pd.DataFrame'`

**说明：**

Load stock daily data with MultiIndex (trade_date, code).

#### load_stock_daily_with_industry

- 签名：`load_stock_daily_with_industry(trade_date: 'str') -> 'pd.DataFrame'`

**说明：**

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

#### migrate_stock_daily_add_new_fields

- 签名：`migrate_stock_daily_add_new_fields(conn: 'sqlite3.Connection') -> 'None'`

**说明：**

为 stock_daily 增加新增字段（若已存在则忽略），便于兼容旧库。

#### save_to_cache

- 签名：`save_to_cache(cache_key: 'str', df: 'pd.DataFrame') -> 'None'`

**说明：**

Save DataFrame to CSV for quick reuse.

#### upsert_index_daily

- 签名：`upsert_index_daily(df: 'pd.DataFrame') -> 'None'`

**说明：**

Upsert index daily data by (code, trade_date).

#### upsert_stock_daily

- 签名：`upsert_stock_daily(df: 'pd.DataFrame') -> 'None'`

**说明：**

Upsert stock daily data by (code, trade_date).

#### upsert_stock_industry

- 签名：`upsert_stock_industry(df: 'pd.DataFrame') -> 'None'`

**说明：**

Upsert 股票行业信息（类似申万行业）到 stock_industry 表。

预期 df 来自 BaoStock 的 query_stock_industry 接口，
常见字段包括：
    code, code_name, industry, industryClassification, updateDate

#### upsert_stock_info

- 签名：`upsert_stock_info(df: 'pd.DataFrame') -> 'None'`

**说明：**

Upsert stock basic info by code.


## 模块 `quant_system.indicators.base_indicator`

技术指标基类，定义统一的计算接口。


### 类

#### BaseIndicator

- 定义：`class BaseIndicator`

**说明：**

技术指标抽象基类。

约定：
    - 入参 df 必须至少包含 open、high、low、close、volume 字段；
    - 索引建议为 DatetimeIndex，且按时间升序排列；
    - compute 在 df 上追加指标列后返回 DataFrame（可为原对象或拷贝）。

子类需设置类属性 name 作为唯一标识，并实现 compute。

**公开方法：**

- `compute(self, df: 'pd.DataFrame') -> 'pd.DataFrame'`
- `validate_input(df: 'pd.DataFrame') -> 'None'`


## 模块 `quant_system.indicators.engine`

技术指标引擎：注册表 + 统一计算入口。


### 函数

#### calculate_indicators

- 签名：`calculate_indicators(df: 'pd.DataFrame', indicators: 'Sequence[Union[str, BaseIndicator]] | None' = None, **indicator_kwargs: 'Any') -> 'pd.DataFrame'`

**说明：**

在日线行情 DataFrame 上按序追加指定指标列。

参数：
    df：要求包含 open/high/low/close/volume 等基础字段，索引为日期。
    indicators：
        - None：对全部已注册指标依次计算（数据量大时谨慎使用）；
        - 序列：元素可为指标名称（字符串）或已实例化的 BaseIndicator 对象。
    indicator_kwargs：
        - 预留给未来扩展，当前仅在实例化字符串指标时作为统一参数传入。

返回：
    附加指标列后的 DataFrame，按传入顺序链式计算。

说明：
    - 当前仅面向日线数据；
    - 只计算指标，不产生交易信号。

#### get_indicator

- 签名：`get_indicator(name: 'str') -> 'Type[BaseIndicator]'`

**说明：**

按名称获取指标类，不存在则抛出 ValueError。

#### list_indicators

- 签名：`list_indicators() -> 'List[str]'`

**说明：**

返回当前已注册的指标名称列表（按名称排序）。

#### register_indicator

- 签名：`register_indicator(cls: 'Type[BaseIndicator]') -> 'Type[BaseIndicator]'`

**说明：**

装饰器：将指标类注册到全局注册表。

- 以 cls.name（小写）为 key；
- 若重复注册则打印提示并覆盖旧值。


## 模块 `quant_system.indicators.plugins.macd`

MACD 指标插件。


### 类

#### MACDIndicator

- 定义：`class MACDIndicator`

**说明：**

MACD（指数平滑异同移动平均）插件。

入参要求：
    - df 包含 close 列与基础行情字段，索引为日期。
新增列：
    - MACD_{fast}_{slow}_{signal}
    - MACDh_{fast}_{slow}_{signal}（柱状图）
    - MACDs_{fast}_{slow}_{signal}（信号线）

**公开方法：**

- `compute(self, df: 'pd.DataFrame') -> 'pd.DataFrame'`


## 模块 `quant_system.indicators.plugins.moving_average`

简单移动平均线插件。


### 类

#### SimpleMovingAverage

- 定义：`class SimpleMovingAverage`

**说明：**

简单移动平均线（SMA）插件。

入参要求：
    - df 至少包含 price_col（默认 close）列，索引为日期。
新增列：
    - SMA_{window}：price_col 的 rolling(window).mean()。

**公开方法：**

- `compute(self, df: 'pd.DataFrame') -> 'pd.DataFrame'`


## 模块 `quant_system.indicators.plugins.rsi`

RSI 指标插件。


### 类

#### RSIIndicator

- 定义：`class RSIIndicator`

**说明：**

相对强弱指标（RSI）插件。

入参要求：
    - df 包含 close 列与基础行情字段，索引为日期。
新增列：
    - RSI_{length}

**公开方法：**

- `compute(self, df: 'pd.DataFrame') -> 'pd.DataFrame'`


## 模块 `quant_system.indicators.plugins.ultimate_features`

终极特征插件：聚合多项常用技术特征。


### 类

#### UltimateFeatures

- 定义：`class UltimateFeatures`

**说明：**

聚合 Supertrend、Squeeze Pro、MACD/VWMACD、EWO、RSI 等特征。

新增列（若计算成功）：
    - SUPERTd_{st_length}_{st_multiplier}：Supertrend 方向
    - SQZPRO_ON：Squeeze Pro 挤压状态
    - MACDh_{fast}_{slow}_{signal}：传统 MACD 柱状图
    - VWMACDh_{fast}_{slow}_{signal}：成交量加权 MACD 柱状图
    - EWO_{fast}_{slow}：Elliott Wave Oscillator
    - RSI_{rsi_length}：相对强弱指标
    - MA5_UP_MA20：5 日均线上穿 20 日均线（布尔转 float）
    - vol_adj_ratio：波动率标准化量比

**公开方法：**

- `compute(self, df: 'pd.DataFrame') -> 'pd.DataFrame'`


## 模块 `quant_system.processing.cleaner`

行情数据清洗与标准化工具。


### 函数

#### apply_adjustment

- 签名：`apply_adjustment(df: 'pd.DataFrame', method: 'str' = 'forward') -> 'pd.DataFrame'`

**说明：**

复权处理入口，预留扩展能力。
参数：
    df：已对齐的原始行情数据。
    method：复权方式占位，目前默认 "forward"。
返回：
    暂不做任何变换，直接返回传入的 DataFrame。

#### fill_missing_trading_days

- 签名：`fill_missing_trading_days(df: 'pd.DataFrame', calendar: 'Sequence[pd.Timestamp | str] | pd.Index', method: 'str' = 'ffill') -> 'pd.DataFrame'`

**说明：**

按给定交易日历补齐缺失日期，并按指定方式填充。
参数：
    df：已按日期索引的行情数据。
    calendar：交易日序列，可为列表、DatetimeIndex 或其它可迭代日期。
    method：填充方式，当前支持 "ffill"（向前填充）。
返回：
    索引对齐到交易日历后的 DataFrame。

#### prepare_for_analysis

- 签名：`prepare_for_analysis(df: 'pd.DataFrame', date_col: 'str' = 'trade_date', code_col: 'str' = 'code', rename_map: 'Mapping[str, str] | None' = None, calendar: 'Sequence[pd.Timestamp | str] | pd.Index | None' = None, adjust_method: 'str' = 'forward', fill_method: 'str' = 'ffill') -> 'pd.DataFrame'`

**说明：**

面向策略/回测的统一清洗入口：
1. standardize_ohlcv：统一列名、类型与索引；
2. apply_adjustment：预留复权处理；
3. 若缺失 pct_chg 列，则用前收盘计算涨跌幅；
4. 确保输出包含 open/high/low/close/preclose/pct_chg/volume/amount。
参数：
    df：原始行情数据。
    date_col：日期列名。
    code_col：代码列名。
    rename_map：可选的重命名映射。
    calendar：若提供则按照交易日历补齐缺口。
    adjust_method：复权方式占位。
    fill_method：补全交易日的填充方式，占位仅支持 ffill。
返回：
    列结构统一、索引为日期的 DataFrame。

#### standardize_ohlcv

- 签名：`standardize_ohlcv(df: 'pd.DataFrame', date_col: 'str', code_col: 'str', rename_map: 'Mapping[str, str] | None' = None) -> 'pd.DataFrame'`

**说明：**

将原始行情数据统一为标准格式：
- 日期列转为 datetime，并设置为索引；
- 根据 rename_map 重命名列（常用于把第三方字段映射到 open/close 等标准名）；
- 自动按日期排序，便于后续补全和复权；
- 除代码与日期列之外的字段全部转为浮点数。
参数：
    df：原始 DataFrame。
    date_col：日期字段名称（重命名前的列名）。
    code_col：证券代码字段名称（重命名前的列名）。
    rename_map：可选的重命名映射，键为原列名，值为目标列名。
返回：
    以日期为索引、数值列已统一为 float 的 DataFrame。


## 模块 `quant_system.processing.industry_sentiment`

行业情绪统计与标签计算。

Notebook 示例：
    from quant_system.processing.industry_sentiment import calc_industry_sentiment

    df_today = calc_industry_sentiment()  # 默认最新交易日
    df_today.head()

    df_20251205 = calc_industry_sentiment("2025-12-05", min_stock_count=5)
    df_20251205.head()


### 函数

#### calc_industry_sentiment

- 签名：`calc_industry_sentiment(trade_date: 'str | None' = None, min_stock_count: 'int' = 5) -> 'pd.DataFrame'`

**说明：**

计算指定交易日的行业情绪统计结果。

功能：
    - 从 SQLite 读取指定交易日的全市场日线 + 行业信息；
    - 计算每个行业的涨跌统计、平均涨幅、成交额等；
    - 打上行业情绪标签（高潮 / 普通 / 冰点）；
    - 按“情绪从高到低 + 平均涨幅从高到低”进行排序；
    - 列名全部使用中文，方便直接展示。

参数：
    trade_date: 交易日字符串，格式 "YYYY-MM-DD"。若为 None，则自动使用 stock_daily 表中的最新交易日。
    min_stock_count: 行业内最少股票数量过滤条件，小于此门槛的行业将被丢弃。

返回：
    一个按情绪排序的 DataFrame，主要字段包括：
        - 行业
        - 股票数
        - 上涨家数
        - 下跌家数
        - 涨停家数
        - 跌停家数
        - 平均涨幅(%)
        - 中位涨幅(%)
        - 总成交额(亿元)
        - 上涨占比
        - 情绪


## 模块 `quant_system.processing.market_view`

市场概览与情绪统计。


### 函数

#### calc_market_overview

- 签名：`calc_market_overview(trade_date: 'str | None' = None) -> 'pd.DataFrame'`

**说明：**

统计指定交易日的全市场情绪概览。
指标包含：
    总股票数、上涨/下跌/涨停/跌停家数、平均与中位数涨幅、总成交额(亿元)、上涨占比、市场情绪。
参数：
    trade_date：交易日，格式 YYYY-MM-DD；为空时自动取 stock_daily 中最新交易日。
返回：
    仅一行的 DataFrame，索引为交易日，列名全为中文。

#### calc_overview_between

- 签名：`calc_overview_between(start: 'str', end: 'str') -> 'pd.DataFrame'`

**说明：**

生成一段日期内的每日市场情绪序列。
参数：
    start：起始日期，格式 YYYY-MM-DD。
    end：结束日期，格式 YYYY-MM-DD。
返回：
    以日期为索引的 DataFrame，每行对应当日的市场概览（列结构与 calc_market_overview 保持一致）。


## 模块 `quant_system.strategy`

策略插件注册与加载。


### 函数

#### get_strategy

- 签名：`get_strategy(name: 'str') -> 'Type[BaseStrategy]'`

**说明：**

按名称获取策略类，若不存在则抛出 KeyError。

#### list_strategies

- 签名：`list_strategies() -> 'list[str]'`

**说明：**

返回当前已注册的策略名称列表。

#### register_strategy

- 签名：`register_strategy(cls: 'Type[BaseStrategy]') -> 'Type[BaseStrategy]'`

**说明：**

装饰器：注册策略，使用类属性 name 作为 key。


## 模块 `quant_system.strategy.base_strategy`

日线单标的策略基类与上下文定义。


### 类

#### BaseStrategy

- 定义：`class BaseStrategy`

**说明：**

日线单标的策略基类。

约定：
    - 输入：带有 open/high/low/close/volume 等基础行情列 + 指标列 的 DataFrame，
            行索引为 DatetimeIndex。
    - 输出：一个包含 'signal' 列的 DataFrame 或 Series：
        signal = 1  → 开多 / 持有多头
        signal = 0  → 空仓
        signal = -1 → 平多 or 做空（当前只实现多头，所以 -1 视为平仓）

**公开方法：**

- `generate_signals(self, df: 'pd.DataFrame', context: 'Optional[StrategyContext]' = None) -> 'pd.Series'`
- `required_indicators(self) -> 'list[str]'`

#### PluggableStrategy

- 定义：`class PluggableStrategy`

**说明：**

带有元信息的策略基类，所有“策略插件”都继承它。
注意：仍然沿用 BaseStrategy 的 generate_signals 接口（返回 signal 序列）。

**公开方法：**

- `get_meta() -> 'Dict[str, Any]'`

#### StrategyCategory

- 定义：`class StrategyCategory`

**说明：**

策略大类，对齐前端：趋势 / 反转 / 波动率 / 成交量。

#### StrategyContext

- 定义：`class StrategyContext`

**说明：**

策略运行上下文。

参数：
    code：标的代码。
    initial_cash：初始资金。
    fee_rate：手续费率。
    slippage：滑点（元）。
    extra：预留的额外上下文字段。

#### StrategyMeta

- 定义：`class StrategyMeta`

**说明：**

策略元信息，既给前端用，也给回测引擎用。

#### StrategyParamMeta

- 定义：`class StrategyParamMeta`

**说明：**

单个参数的元数据，用于前端渲染表单和后端校验。


## 模块 `quant_system.strategy.plugins`

策略插件包。

提供 `load_all_plugins()` 动态加载本目录下的所有策略模块，
确保它们的 @register_strategy 装饰器被执行。


### 函数

#### load_all_plugins

- 签名：`load_all_plugins() -> 'List[ModuleType]'`

**说明：**

动态导入本包下的所有 .py 策略文件（排除以下划线开头的模块）。

用法示例（在回测或 API 启动时调用一次即可）：
    from quant_system.strategy.plugins import load_all_plugins
    from quant_system.strategy.registry import list_strategies

    load_all_plugins()
    metas = [m.to_dict() for m in list_strategies()]


## 模块 `quant_system.strategy.plugins.connors_rsi2`

Connors RSI(2) 极限反转策略（V1：贴近期刊/书籍原始规则版本）

对应文档《策略一：Connors RSI(2) 极限反转策略》：

- 环境过滤：收盘价 > 200 日简单移动平均线 (SMA200)，只在长期上升趋势中做多；
- 入场信号：RSI(2) < 5（默认），部分激进可放宽到 < 10；
- 变体：CumRSI(2) = RSI(2) 当天 + 前一天，当 CumRSI(2) < 35 时买入（本版本中为可选增强，默认关闭）；
- 出场信号：收盘价 > 5 日简单移动平均线 (SMA5)，视为价格已回归短期均值；
- 止损：原版不设硬止损，依赖高胜率 + 分散持仓，本策略保持此设定，ATR 止损预留给回测引擎统一实现。


### 函数

#### compute_rsi

- 签名：`compute_rsi(close: 'pd.Series', period: 'int') -> 'pd.Series'`

**说明：**

简单 RSI 计算：
    RSI = 100 - 100 / (1 + RS)
    RS = 平均上涨幅度 / 平均下跌幅度

这里使用 rolling mean 近似 Wilder 的平滑方法，对 N=2 的短周期影响很小。


### 类

#### ConnorsRsi2Strategy

- 定义：`class ConnorsRsi2Strategy`

**说明：**

Connors RSI(2) 极限反转策略（V1 原始规则版，只做多）。

约定：
    - 输入 df 至少包含列：open/high/low/close/volume；
    - 本策略内部自行计算 RSI(2)、SMA5、SMA200，不依赖外部指标列；
    - 输出为 0/1 持仓序列（不做空），执行时机由回测引擎控制。

**公开方法：**

- `generate_signals(self, df: 'pd.DataFrame', context: 'Optional[StrategyContext]' = None) -> 'pd.Series'`
- `required_indicators(self) -> 'list[str]'`


## 模块 `quant_system.strategy.plugins.ma_rsi_long_only`

均线金叉 + RSI 过滤的多头示例策略。


### 类

#### MaRsiLongOnly

- 定义：`class MaRsiLongOnly`

**说明：**

简单多头策略：
    - 快速均线向上穿越慢速均线，且 RSI > rsi_lower 时开多/持有；
    - 快速均线下穿慢速均线，或 RSI > rsi_upper 时平仓；
    - 其余时间保持前一日信号。

主要参数：
    fast_ma：快线窗口
    slow_ma：慢线窗口
    rsi_lower：RSI 低阈值（低于该值才允许开仓）
    rsi_upper：RSI 高阈值（高于该值则平仓）

**公开方法：**

- `generate_signals(self, df: 'pd.DataFrame', context: 'StrategyContext | None' = None) -> 'pd.Series'`
- `required_indicators(self) -> 'list[str]'`


## 模块 `quant_system.strategy.registry`

策略注册表：统一管理所有可用的日线单标的策略。


### 函数

#### create_strategy

- 签名：`create_strategy(strategy_id: 'str', params: 'Optional[Dict[str, Any]]' = None) -> 'BaseStrategy'`

**说明：**

根据 id 创建策略实例，在回测模块里用。

#### get_strategy_meta

- 签名：`get_strategy_meta(strategy_id: 'str') -> 'StrategyMeta'`

**说明：**

(暂无说明，TODO)

#### list_strategies

- 签名：`list_strategies(category: 'Optional[str]' = None) -> 'List[StrategyMeta]'`

**说明：**

返回所有策略的元信息列表，可按类别过滤。

以后 /api/backtest/strategies 可以直接：
    from quant_system.strategy.registry import list_strategies
    ...

#### register_strategy

- 签名：`register_strategy(meta: 'StrategyMeta')`

**说明：**

用作类装饰器，将策略类注册到全局表中。

示例：
    @register_strategy(StrategyMeta(...))
    class MyStrategy(BaseStrategy):
        ...


### 类

#### StrategyMeta

- 定义：`class StrategyMeta`

**说明：**

策略元信息，用于前端展示和参数配置。

字段说明：
    id          : 唯一标识（英文短名，如 "connors_rsi2"）
    name        : 中文名，用于 UI 展示
    category    : 策略类别（如 "trend" / "reversal" / "volatility" / "volume"）
    description : 简短说明
    tags        : 若干标签（如 ["mean_reversion", "short_term"]）
    default_params : 默认参数（回测时若没传就用这里）
    param_schema   : 参数结构描述，给前端或配置中心用（可选）

**公开方法：**

- `to_dict(self) -> 'Dict[str, Any]'`

