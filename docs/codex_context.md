# A股量化分析系统 Codex 开发上下文指南
项目开发规范与模块说明详见：`docs/codex_context.md`，供 Codex 和协作者参考。
本文件旨在为 VS Code 中的 OpenAI Codex 插件提供长期参考的项目上下文。**Codex** 作为本项目的 AI 编码助手，应充分理解系统架构、模块职责、编码规范和开发流程，以便持续生成与架构一致、高质量的代码。本指南将介绍项目模块分割、插件机制、目录结构、接口设计、命名风格、Notebook 使用约定，以及开发优先级和扩展建议，确保 Codex 在协助开发时遵循既定规范，保持系统的模块化和可扩展性。

...

A股量化分析系统 Codex 开发上下文指南
本文件旨在为 VS Code 中的 OpenAI Codex 插件提供长期参考的项目上下文。Codex 作为本项目的 AI 编码助手，应充分理解系统架构、模块职责、编码规范和开发流程，以便持续生成与架构一致、高质量的代码。本指南将介绍项目模块划分、插件机制、目录结构、接口设计、命名风格、Notebook 使用约定，以及开发优先级和扩展建议，确保 Codex 在协助开发时遵循既定规范，保持系统的模块化和可扩展性。
Codex 在本项目中的角色和规范
Codex 的角色： 充当本量化分析系统的智能编码助手。在开发过程中，Codex 应根据本指南提供的架构和规范，协助编写代码、提供实现思路，确保新代码与现有结构保持一致。
行为规范：
•	遵循架构约定： Codex 在生成代码时须严格按照项目模块划分和职责边界放置功能。例如，数据相关功能应放在数据模块，指标计算应放在指标引擎模块等，避免将无关逻辑混入其他模块。
•	保持低耦合性： 遵从模块解耦原则，通过接口交互而非直接访问内部实现。Codex 应确保生成的代码在不同模块间仅通过清晰的接口通信，不引入环依赖或未经设计的直接耦合。
•	利用现有库与工具： 当前环境已安装了关键库（如 baostock、akshare、numpy、pandas、pandas-ta、matplotlib、mplfinance 等[1][2]）。Codex 生成代码时应尽量使用这些现有库实现功能，避免重复造轮子。例如，技术指标计算应使用 pandas_ta 提供的函数而非手动实现公式；数据获取优先调用 BaoStock 接口而非自行解析网页。
•	遵循代码风格： 采用 PEP8 编码规范，使用一致的命名风格。模块、文件和函数命名采用小写下划线风格，类命名采用大驼峰风格。为提高可读性，Codex 生成的代码应包含必要的注释和文档字符串，解释复杂逻辑或关键步骤。
•	稳健性与测试： Codex 在提议实现方案时应考虑异常处理和稳健性。例如，数据获取要考虑网络异常或数据缺失情况；回测执行要确保不会因单个错误中断。鼓励为关键函数生成单元测试（项目包含 tests/ 目录），Codex 可以根据上下文辅助编写测试用例，以验证模块功能和接口契合。
•	迭代开发思维： 遵循“小步快走，逐模块完善”的开发流程（见下文开发顺序）。Codex 应鼓励分阶段完成功能，每阶段在 Notebook 或测试环境下验证，再进行下一阶段开发。这有助于及时发现问题，确保各模块正确集成。
通过以上行为规范，Codex 将更有效地辅助开发者编写出结构清晰、易于维护的代码，保持整个项目的一致性和高质量。
模块划分与职责
本系统采用模块化架构，将不同功能划分到独立模块中，各模块高内聚、低耦合，协同构建完整的量化分析流程[3][4]。主要模块及其职责如下：
数据获取模块 (quant_system/data)
职责： 从外部数据源获取A股市场所需的各类数据，并进行本地存储或缓存[5]。包括： - 行情数据下载： 批量获取股票日线、周线历史行情（开盘价、收盘价、最高价、最低价、成交量等）以及指数行情数据。[5] - 基本面数据获取： 获取财务指标、行业分类等辅助数据（通过 BaoStock 财务数据或 AkShare 等接口）。 - 数据源接口封装： 封装 BaoStock 库的调用（如 query_history_k_data()）以及备用数据源 AkShare 调用，以提供统一的数据获取接口[6]。BaoStock 数据返回 Pandas DataFrame，便于后续处理[7]。 - 本地缓存与存储： 将获取的原始数据保存到本地（如 CSV 文件或轻量数据库 SQLite）以加速后续访问[5]。提供缓存管理，避免重复下载相同数据。
实现提示： 使用 quant_system/data/fetcher.py 实现数据下载逻辑，storage.py 处理本地文件读写。数据获取函数应设计简单明了，如 get_stock_data(code, start_date, end_date) 返回指定股票的历史 DataFrame。考虑实现基本缓存机制：优先检查本地是否已有数据，若无则调用远程接口获取并存储。
数据处理与市场概况模块 (quant_system/processing)
职责： 清洗整理原始数据，生成日常市场概况指标，为后续分析提供可靠基础[8]。包括： - 数据清洗与补全： 处理缺失值、异常值，根据需要进行前复权/后复权调整，确保时间序列数据连续合理[8]。实现例如 cleaner.py 中的函数，对下载的行情数据进行整理。 - 市场情绪统计： 计算每日整体市场指标，用于衡量市场情绪和概况[8]。例如： - 上涨家数与下跌家数，涨跌比率。 - 涨停板和跌停板股票数量[9]。 - 当日总成交量、平均换手率等。 这些指标帮助判断市场是否处于极端状态（情绪冰点或高潮），为策略提供参考[8]。 - 数据转换工具： 提供通用的时间序列处理函数（如计算滚动均值、涨跌幅等）供其他模块使用。
实现提示： quant_system/processing/cleaner.py 实现清洗函数，如 fill_missing_values(data)、adjust_for_splits(data) 等；market_view.py 实现市场概况计算，如 calc_market_overview(data) 返回包含涨跌家数、涨停数等指标的结果。注意确保处理后的数据格式统一，例如设置日期为索引的 DataFrame，列明晰。Codex 生成代码时，应确保这些工具函数通用且与 Pandas DataFrame 操作兼容。
板块与行业分析模块 (quant_system/sector)(计划中)
职责： 从行业和概念板块层面分析市场行情，识别热点板块和主题投资线索[10]。包括： - 行业分类获取： 获取每只股票所属行业/概念信息（可利用BaoStock财务数据或AkShare等获得行业映射）。 - 板块指标计算： 计算各行业板块内股票的涨跌幅平均值、中位数涨幅、板块内上涨下跌比等指标[10]。统计各板块的涨停占比（如板块内有多少股票涨停）等情绪指标，以衡量板块热度[10]。 - 领涨板块识别： 排序输出当日或周期内领涨和领跌的板块列表，辅助判断市场资金流向和板块轮动。
实现提示： 该模块目前规划中，未来可添加 quant_system/sector/ 包，实现 sector_analyzer.py 等脚本。Codex 在将来生成此模块代码时，应确保其与数据模块配合良好，例如从数据模块或处理模块获取已经整理好的行情数据，再按行业分类聚合计算。
技术指标引擎模块 (quant_system/indicators)
职责： 计算各种技术分析指标，并生成指标信号，提供给策略使用[11][12]。特点是支持插件式扩展技术指标。包括： - 常用技术指标计算： 内置计算移动平均线（MA）、指数平滑移动均线（EMA）、相对强弱指数（RSI）、移动平均趋同背离（MACD）、布林带等常见指标值。利用 Pandas 及 NumPy 高效计算滚动指标[11]；或直接使用 Pandas TA 库加速实现[13]。 - 指标信号生成： 根据计算的指标判断买卖信号。如金叉死叉信号、超买超卖信号等，并将这些信号标记在数据中供策略读取[11]。 - 用户自定义指标插件： 通过插件架构允许用户扩展新的指标计算方法而无需修改核心代码[11]。系统应扫描特定目录（如 indicators/plugins/）下的自定义指标脚本，并自动注册它们[12]。每个插件脚本可以定义一个如 compute_indicator(data) 的函数或定义一个派生自基础指标类的类，实现特定指标计算逻辑[12]。核心引擎通过统一接口调用这些插件。
实现提示： 在 quant_system/indicators/engine.py 中实现指标计算的入口函数，如 calculate_indicators(data, config)，根据配置确定计算哪些指标。可使用 pandas_ta 库直接添加指标列，例如：data["RSI"] = data["close"].ta.rsi(length=14) 等[13]。插件机制方面，Codex 编码时可实现一个扫描加载函数：启动时遍历 indicators/plugins 目录下的 .py 文件，使用 importlib 导入，每个插件应注册自身（例如通过在模块中添加到某全局指标列表或调用引擎注册函数）。确保 base_indicator.py 定义统一接口或基类（如 BaseIndicator），让插件继承，实现其 compute() 方法，从而引擎可以多态地调用。Codex 生成新指标插件代码时，应放入该目录并遵循既定接口，而非修改引擎核心代码，实现“开闭原则”下的功能扩展[12]。
策略与信号模块 (quant_system/strategy)
职责： 基于指标引擎输出的指标值和信号，定义具体的交易策略逻辑，并产生最终的买卖决策信号[14][15]。特点是策略与指标、回测解耦，每个策略独立插件化。包括： - 策略规则定义： 每个策略包含明确的买入、卖出条件规则。例如“多指标共振”（需满足多个指标条件）、均线交叉、动量反转等策略。[14] - 信号生成： 策略读取指标引擎计算出的原始信号和数据，在此基础上应用自身规则，输出交易信号序列（如 1 表示买入、-1 卖出、0 观望的数组）供回测模块执行[15]。策略可能进一步考虑止盈止损位等细节，生成更加完善的信号。 - 插件式策略扩展： 策略模块也采用插件架构，新策略可以通过新增文件方式接入[15]。在 strategy/plugins/ 目录下，每个文件实现一个策略类，继承自 BaseStrategy（定义于 base_strategy.py），并实现统一的接口方法（如 generate_signals(data, indicators)）[15]。系统初始化时扫描该目录，自动加载所有策略类，便于快速切换或组合策略。
实现提示： quant_system/strategy/base_strategy.py 定义 BaseStrategy 抽象类，规定如 initialize()（初始化参数）、generate_signals(data, indicators) 等方法签名。Codex 在生成新策略代码时，应创建新文件于 strategy/plugins/，定义类继承 BaseStrategy 并实现所需方法。策略逻辑内部应仅依赖传入的行情数据和指标结果，不直接调用数据获取或回测细节，从而保持策略的独立性。通过解耦，用户可以并行开发多种策略，而 Codex 提供代码应确保不同策略之间不会相互影响。
回测模块 (quant_system/backtest)
职责： 提供历史回测引擎，在给定历史数据和策略信号的情况下，模拟实际交易过程以评估策略表现[4][16]。包括： - 事件驱动模拟交易： 按交易日期迭代历史行情，每日根据策略信号决定是否买入或卖出[4]。可采用事件驱动（信号触发下单）或序贯循环的方式实现。回测引擎负责维护账户状态（现金余额、持仓）和交易执行。 - 交易规则约束： 在模拟下单时考虑实际交易限制，如当日涨跌停无法成交、交易手续费和滑点等，对策略信号进行现实约束。[4] - 绩效统计： 回测结束后输出策略表现报告，包括逐笔交易记录、每日资产净值曲线、累计收益率、年化收益、最大回撤、夏普比率等指标[4][16]。 - 风控与资金管理集成： 回测过程中，每次产生交易信号应调用风控模块检验、调用资金管理模块调整头寸，然后再执行交易[16]。这种通过接口回调嵌入风控/资金管理的设计，使回测引擎本身独立于具体风控策略，便于日后扩展。
实现提示： quant_system/backtest/engine.py 是回测主逻辑入口。Codex 生成代码时，可按时间序列迭代方式组织，例如 for date in dates: ... 依次处理每个交易日数据。在处理每个日期时，拿到策略模块给出的该日信号，然后通过调用风控检查（如 risk_manager.check_order(signal)) 和资金管理计算仓位 (position_sizer.allocate(capital, signal))[17][18]，最终决定下单执行。执行后更新账户余额和持仓状态。需要模拟扣除手续费、检查涨跌停板限制（若信号触发买入但涨停板，则无法买入等）。回测结束后汇总结果，可将交易日志存入列表或 DataFrame，计算净值曲线和各项评价指标。Codex 编写回测代码时，应保持回测引擎通用性，不捆绑特定策略或数据，以便测试不同策略组合。
风险控制模块 (quant_system/risk)(计划中)
职责： 集中管理交易风险，在策略给出交易信号后、执行交易前，过滤或调整违规信号，控制策略风险敞口[19][20]。包括： - 资金风险控制： 限制单笔交易资金占比或最大亏损额度。例如：单笔交易资金不得超过总资金的一定比例；设置整体回撤阈值，一旦净值回撤超限则停止交易[19]。 - 持仓风险控制： 控制持仓集中度。例如：单只股票的持仓市值不得超过组合的一定比例；禁止持有某些高风险板块超额头寸。 - 交易频率控制： 限制交易频率和连续亏损交易次数。例如：每日最多交易N次；若连续亏损超过M次则暂停交易一段时间[19]。 - 风控触发处理： 当策略信号违反风控规则时，风控模块可以采取措施：如直接拒绝该交易信号（不执行），或调整下单数量/金额至符合风控要求的范围[21]。风控模块应记录每次拦截或调整的事件，以供事后分析。
实现提示： 风控模块未来可作为独立包 quant_system/risk/，实现 risk_manager.py。其中 RiskManager 类提供方法 check_order(order) 或 validate_signal(signal, context) 等，用于在回测/实盘前审查交易信号[21]。Codex 在实现风控代码时，应使其规则配置化（如通过配置文件设定阈值参数）并尽量通用。例如，通过读取全局配置确定单笔交易最大比例、止损线等，在检测到超限时返回调整或拒绝标志。注意风控模块不参与策略逻辑，仅在信号到实际执行这一环节进行把关。
资金管理模块 (quant_system/money)(计划中)
职责： 决定每笔交易的资金或头寸规模，根据账户资金状况和风险偏好动态调整仓位，实现更科学的资金分配[22][23]。包括： - 仓位大小计算： 提供多种头寸计算策略，例如固定比例法（每次投入固定%资金）、金字塔加仓/减仓法（分批增减仓位）、凯利公式法（根据胜率和盈亏比计算最优投注比例）等[23]。 - 动态调整： 根据当前资金余额、持仓风险敞口，实时调整下单规模。例如在连续盈利后可增加仓位，连续亏损时减少仓位，以控制回撤。 - 接口集成： 资金管理通过标准接口供回测/实盘调用，如 allocate(capital, signal) 方法返回建议的下单股数或资金额[23]。这一接口可以被不同实现策略覆盖，方便快速切换资金管理方案。
实现提示： 未来在 quant_system/money/position_sizer.py 中实现 PositionSizer 类及其策略方法。Codex 编写资金管理代码时，应考虑简洁性和扩展性，例如使用策略模式根据配置选择不同算法。默认实现可简单按固定资金比例开仓，更复杂的可以通过参数控制启用。资金管理应独立于具体策略逻辑，通过在回测中调用其接口来决定下单量，不直接影响策略发出的信号，仅作用于执行层。
可视化与报告模块 (quant_system/visualization)
职责： 将分析结果和回测绩效进行直观展示，方便开发者和用户理解策略表现[24][25]。包括： - 图表绘制： 绘制股票价格走势K线图，叠加技术指标曲线以及买卖点标记；绘制回测期间的资产净值曲线和回撤曲线等[24]。利用 Matplotlib 及其金融绘图扩展 mplfinance 绘制专业金融图表[24]。 - 指标可视化： 将关键技术指标（如均线、MACD指标等）随行情数据一起绘制，以便观察指标与价格互动关系[26]。 - 绩效表格报告： 汇总回测结果指标，生成表格列出年化收益、最大回撤、夏普比率、胜率等绩效数据[4]。必要时将这些结果输出为 Markdown 或PDF报告，便于分享。 - 交互式展示（可选）： 在 Notebook 环境中，使用 Plotly 或 Bokeh 等库生成交互式图形，在浏览器中支持悬停查看详细数据、缩放时间轴等[27]。这对于深入分析策略细节、展示演示很有帮助。
实现提示： 在 quant_system/visualization/plotting.py 中实现绘图函数，例如 plot_candlestick_with_signals(data, signals) 绘制含买卖点的K线图，plot_equity_curve(equity) 绘制净值曲线等。Codex 生成绘图代码时应充分利用 mplfinance 简化K线绘制[24]，如使用 mpf.plot() 绘制 DataFrame（含 OHLC 数据和附加指标）的蜡烛图，并通过 addplot 参数添加交易信号标记[24]。注意图表美观易读，重要信息（买卖点、收益曲线转折点等）应有清晰标注。在报告部分（如 report.py），可汇总性能指标计算（可以调用回测模块提供的数据），输出为 DataFrame 或保存为文件。Codex 在此模块生成代码时，应确保图表在 Jupyter Notebook 中能够正确显示（例如调用 %matplotlib inline 或适配 VS Code 的 Notebook 渲染）。
配置与日志模块 (quant_system/config 与 quant_system/utils)
职责： 提供全局配置管理和日志记录工具，支撑各功能模块的可配置性和可监控性[28]。包括： - 配置管理： 将策略参数、指标参数、风控阈值等可变设置存储于配置文件（如 config/settings.yaml），系统启动时加载这些配置[29]。通过配置驱动，新增或调整参数不需要改动代码，只需修改配置文件，提升灵活性[29]。需提供解析器，如 config_loader.py，读取 YAML/JSON 配置并生成Python字典或对象。 - 日志记录： 通过统一的日志模块初始化日志器，在各模块关键步骤输出日志信息[28]。例如： - 数据获取成功或失败应记录； - 策略每次发出交易信号时记录信号详情； - 风控拒单或调整仓位时记录原因； - 回测完成后输出总结报告日志等。 使用Python内置 logging 库设置适当的日志级别和输出格式，将日志写入控制台和文件以便审计和调试[29]。 - 通用工具： 提供常用辅助函数模块 utils/helpers.py，实现如日期格式转换、文件路径处理、指标序列平滑等功能，供各模块重用。
实现提示： 当前项目已包含一个示例配置文件 config/settings.yaml 用于存放参数。Codex 编写代码时，可实现 quant_system/config/config_loader.py 以加载 YAML 配置为字典，并在需要处传入模块使用。日志方面，可在 quant_system/utils/logger.py 初始化一个模块级别的 logger（设置格式如 '%(asctime)s [%(levelname)s] %(message)s'）。Codex 应确保所有模块在关键操作前后调用日志，使开发者在 Notebook 或终端中能够跟踪程序流程[29]。这些工具模块虽然不直接承担量化业务逻辑，但对提高系统可维护性和可调试性至关重要。
项目目录结构
项目采用如下注释的目录结构组织代码，各模块代码文件清晰归类[30][31]。Codex 应熟悉此结构，在生成代码时遵循文件放置约定：



## 📁 Project Directory Structure <!-- CODEX_UPDATE_DIRECTORY_STRUCTURE -->

```text
quant-a-share/  # 项目根目录
├── backend/  # 后端模型与数据流水线
│   ├── __pycache__/  # 后端包缓存
│   │   └── __init__.cpython-310.pyc  # 包初始化缓存
│   ├── data_pipeline/  # 特征工程管道
│   │   ├── __pycache__/  # 数据管道缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   └── feature_engineering.cpython-310.pyc  # 特征工程缓存
│   │   ├── __init__.py  # 数据管道包初始化
│   │   └── feature_engineering.py  # 特征工程处理脚本
│   ├── tft_model/  # TFT 时间序列模型
│   │   ├── __pycache__/  # TFT 模块缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   ├── api.cpython-310.pyc  # API 缓存
│   │   │   ├── predict.cpython-310.pyc  # 预测脚本缓存
│   │   │   ├── train.cpython-310.pyc  # 训练脚本缓存
│   │   │   └── utils.cpython-310.pyc  # 工具函数缓存
│   │   ├── data/  # 模型特征数据
│   │   │   ├── features_sh_600519.csv  # 茅台特征CSV
│   │   │   └── features_sh_600519.parquet  # 茅台特征Parquet
│   │   ├── models/  # 模型权重
│   │   │   ├── tft-epoch=02-val_loss=1.0177.ckpt  # 训练检查点
│   │   │   └── tft-epoch=02-val_loss=1.0203.ckpt  # 训练检查点
│   │   ├── __init__.py  # 模型包初始化
│   │   ├── api.py  # 模型服务接口
│   │   ├── predict.py  # 推理脚本
│   │   ├── train.py  # 训练脚本
│   │   └── utils.py  # 模型工具函数
│   ├── __init__.py  # 后端包初始化
│   ├── README.md  # 后端说明文档
│   └── requirements.txt  # 后端依赖清单
├── config/  # 全局配置
│   └── settings.yaml  # 系统参数示例
├── data_cache/  # 行情数据缓存
│   ├── index_sh-000001_2024-12-08_2025-12-08_d.csv  # 上证指数日线缓存
│   ├── index_sh-000001_2024-12-09_2025-12-09_d.csv  # 上证指数日线缓存
│   ├── market_data.sqlite  # 行情SQLite缓存库
│   ├── stock_sh-600519_2015-01-01_2025-12-09_d.csv  # 贵州茅台日线缓存
│   ├── stock_sh-600519_2023-01-01_2024-12-31_d.csv  # 贵州茅台日线缓存
│   ├── stock_sh-600519_2024-01-01_2024-12-31_d.csv  # 贵州茅台日线缓存
│   ├── stock_sh-600519_2024-01-01_2025-12-07_d.csv  # 贵州茅台日线缓存
│   ├── stock_sh-600519_2024-01-01_2025-12-09_d.csv  # 贵州茅台日线缓存
│   ├── stock_sh-688017_2025-01-01_2025-12-09_d.csv  # 绿的谐波日线缓存
│   ├── stock_sh-688192_2024-01-01_2025-12-09_d.csv  # 迪哲医药日线缓存
│   ├── stock_sh-688192_2024-12-10_2025-12-10_d.csv  # 迪哲医药日线缓存
│   ├── stock_sh-688192_2025-01-01_2025-12-09_d.csv  # 迪哲医药日线缓存
│   └── stock_sz-002460_2025-01-01_2025-12-09_d.csv  # 赣锋锂业日线缓存
├── docs/  # 项目文档
│   ├── api_reference.md  # API 参考
│   └── codex_context.md  # Codex 上下文指南
├── google-ai-webui/  # Google AI Studio 前端样例
│   └── .env.local  # 本地环境变量示例
├── notebooks/  # Notebook 示例
│   ├── data_cache/  # Notebook 缓存数据
│   │   └── market_data.sqlite  # Notebook 本地数据库
│   ├── 01_data_and_market_overview.ipynb  # 数据与市场概览Notebook
│   └── 测试.ipynb  # 临时测试Notebook
├── quant_system/  # 量化系统核心代码
│   ├── __pycache__/  # 包缓存
│   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   └── __init__.cpython-313.pyc  # 包初始化缓存
│   ├── backtest/  # 回测引擎
│   │   ├── __pycache__/  # 回测模块缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   ├── __init__.cpython-313.pyc  # 包初始化缓存
│   │   │   ├── engine.cpython-310.pyc  # 引擎缓存
│   │   │   ├── engine.cpython-313.pyc  # 引擎缓存
│   │   │   ├── performance.cpython-310.pyc  # 绩效统计缓存
│   │   │   └── performance.cpython-313.pyc  # 绩效统计缓存
│   │   ├── __init__.py  # 回测包初始化
│   │   ├── engine.py  # 回测执行引擎
│   │   └── performance.py  # 回测绩效指标
│   ├── data/  # 数据获取与存储
│   │   ├── __pycache__/  # 数据模块缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   ├── __init__.cpython-313.pyc  # 包初始化缓存
│   │   │   ├── fetcher.cpython-310.pyc  # 抓取逻辑缓存
│   │   │   ├── fetcher.cpython-313.pyc  # 抓取逻辑缓存
│   │   │   ├── storage.cpython-310.pyc  # 存储逻辑缓存
│   │   │   └── storage.cpython-313.pyc  # 存储逻辑缓存
│   │   ├── __init__.py  # 数据包初始化
│   │   ├── fetcher.py  # 数据抓取封装
│   │   └── storage.py  # 本地存储封装
│   ├── indicators/  # 技术指标引擎
│   │   ├── __pycache__/  # 指标模块缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   ├── __init__.cpython-313.pyc  # 包初始化缓存
│   │   │   ├── base_indicator.cpython-310.pyc  # 指标基类缓存
│   │   │   ├── base_indicator.cpython-313.pyc  # 指标基类缓存
│   │   │   ├── engine.cpython-310.pyc  # 指标引擎缓存
│   │   │   └── engine.cpython-313.pyc  # 指标引擎缓存
│   │   ├── plugins/  # 指标插件
│   │   │   ├── __pycache__/  # 插件缓存
│   │   │   │   ├── __init__.cpython-310.pyc  # 插件初始化缓存
│   │   │   │   ├── __init__.cpython-313.pyc  # 插件初始化缓存
│   │   │   │   ├── macd.cpython-310.pyc  # MACD 缓存
│   │   │   │   ├── macd.cpython-313.pyc  # MACD 缓存
│   │   │   │   ├── moving_average.cpython-310.pyc  # 均线缓存
│   │   │   │   ├── moving_average.cpython-313.pyc  # 均线缓存
│   │   │   │   ├── rsi.cpython-310.pyc  # RSI 缓存
│   │   │   │   ├── rsi.cpython-313.pyc  # RSI 缓存
│   │   │   │   ├── ultimate_features.cpython-310.pyc  # 复合特征缓存
│   │   │   │   └── ultimate_features.cpython-313.pyc  # 复合特征缓存
│   │   │   ├── __init__.py  # 指标插件注册
│   │   │   ├── macd.py  # MACD 指标
│   │   │   ├── moving_average.py  # 均线指标
│   │   │   ├── rsi.py  # RSI 指标
│   │   │   └── ultimate_features.py  # 复合特征指标
│   │   ├── __init__.py  # 指标包初始化
│   │   ├── base_indicator.py  # 指标基类定义
│   │   └── engine.py  # 指标引擎调度
│   ├── processing/  # 数据清洗与市场视图
│   │   ├── __pycache__/  # 处理模块缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   ├── __init__.cpython-313.pyc  # 包初始化缓存
│   │   │   ├── cleaner.cpython-310.pyc  # 清洗逻辑缓存
│   │   │   ├── cleaner.cpython-313.pyc  # 清洗逻辑缓存
│   │   │   ├── industry_sentiment.cpython-310.pyc  # 行业情绪缓存
│   │   │   ├── industry_sentiment.cpython-313.pyc  # 行业情绪缓存
│   │   │   ├── market_view.cpython-310.pyc  # 市场概览缓存
│   │   │   └── market_view.cpython-313.pyc  # 市场概览缓存
│   │   ├── __init__.py  # 处理包初始化
│   │   ├── cleaner.py  # 数据清洗脚本
│   │   ├── industry_sentiment.py  # 行业情绪计算
│   │   └── market_view.py  # 市场概览计算
│   ├── strategy/  # 策略与插件
│   │   ├── __pycache__/  # 策略模块缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   ├── __init__.cpython-313.pyc  # 包初始化缓存
│   │   │   ├── base_strategy.cpython-310.pyc  # 策略基类缓存
│   │   │   ├── base_strategy.cpython-313.pyc  # 策略基类缓存
│   │   │   ├── registry.cpython-310.pyc  # 策略注册缓存
│   │   │   └── registry.cpython-313.pyc  # 策略注册缓存
│   │   ├── plugins/  # 策略插件
│   │   │   ├── __pycache__/  # 插件缓存
│   │   │   │   ├── __init__.cpython-310.pyc  # 插件初始化缓存
│   │   │   │   ├── __init__.cpython-313.pyc  # 插件初始化缓存
│   │   │   │   ├── connors_rsi2.cpython-310.pyc  # CRS 2 缓存
│   │   │   │   ├── connors_rsi2.cpython-313.pyc  # CRS 2 缓存
│   │   │   │   ├── ma_rsi_long_only.cpython-310.pyc  # MA+RSI 缓存
│   │   │   │   ├── ma_rsi_long_only.cpython-313.pyc  # MA+RSI 缓存
│   │   │   ├── __init__.py  # 策略插件注册
│   │   │   ├── connors_rsi2.py  # ConnorsRSI2 策略
│   │   │   └── ma_rsi_long_only.py  # MA+RSI 多头策略
│   │   ├── __init__.py  # 策略包初始化
│   │   ├── base_strategy.py  # 策略基类
│   │   └── registry.py  # 策略注册表
│   ├── visualization/  # 可视化组件
│   │   ├── __pycache__/  # 可视化缓存
│   │   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   │   ├── __init__.cpython-313.pyc  # 包初始化缓存
│   │   │   ├── plotting.cpython-310.pyc  # 绘图缓存
│   │   │   └── plotting.cpython-313.pyc  # 绘图缓存
│   │   ├── __init__.py  # 可视化包初始化
│   │   └── plotting.py  # 绘图函数占位
│   └── __init__.py  # 量化系统包初始化
├── scripts/  # 辅助脚本
│   ├── calc_industry_sentiment.py  # 行业情绪计算脚本
│   ├── download_all_daily_since_2015.py  # 全量日线下载脚本
│   ├── generate_api_docs.py  # 自动生成API文档
│   ├── run_backtest_demo.py  # 回测演示脚本
│   ├── test_indicators_basic.py  # 指标基础测试
│   ├── test_prepare_tft.py  # TFT数据准备测试
│   ├── update_codex_directory_structure.py  # Codex目录更新脚本
│   ├── update_daily_data.py  # 每日数据更新脚本
│   ├── update_directory_structure.py  # 目录结构生成脚本
│   └── update_industry_mapping.py  # 行业映射更新脚本
├── tests/  # 单元测试占位
│   └── __init__.py  # 测试包初始化
├── web_api/  # Python 后端接口
│   ├── __pycache__/  # Web API 缓存
│   │   ├── __init__.cpython-310.pyc  # 包初始化缓存
│   │   ├── __init__.cpython-313.pyc  # 包初始化缓存
│   │   ├── main.cpython-310.pyc  # 主入口缓存
│   │   └── main.cpython-313.pyc  # 主入口缓存
│   ├── __init__.py  # Web API 包初始化
│   ├── main.py  # Web API 入口
│   └── web_api.md  # Web API 文档
├── webui/  # 前端 React 项目
│   ├── node_modules/  # 前端依赖目录（未展开）
│   ├── public/  # 静态资源
│   │   └── vite.svg  # Vite 标识
│   ├── src/  # 前端源代码
│   │   ├── api/  # 前端API封装
│   │   │   └── backtest.ts  # 回测接口封装
│   │   ├── assets/  # 静态素材
│   │   │   └── react.svg  # React 标志
│   │   ├── components/  # 页面组件
│   │   │   ├── AIAnalyst.tsx  # AI 分析组件
│   │   │   ├── BacktestPanel.tsx  # 回测面板
│   │   │   ├── HoldingsTable.tsx  # 持仓表格
│   │   │   ├── IndustrySentimentTable.tsx  # 行业情绪表
│   │   │   ├── MarketChart.tsx  # 市场走势图
│   │   │   ├── MarketOverviewCard.tsx  # 市场概览卡片
│   │   │   ├── MarketStats.tsx  # 市场统计卡片
│   │   │   ├── NewsPage.tsx  # 新闻页面
│   │   │   ├── StockForecastPage.tsx  # 个股预测页
│   │   │   ├── StrategyBacktestPage.tsx  # 策略回测页
│   │   │   ├── StrategyPanel.tsx  # 策略设置面板
│   │   │   └── StrategySettingsPage.tsx  # 策略参数页
│   │   ├── services/  # 前端服务封装
│   │   │   └── geminiService.ts  # Gemini API 客户端
│   │   ├── App.css  # 应用样式
│   │   ├── App.tsx  # 应用入口组件
│   │   ├── index.css  # 全局样式
│   │   ├── main.tsx  # 前端入口
│   │   └── types.ts  # 类型定义
│   ├── .gitignore  # 前端忽略配置
│   ├── eslint.config.js  # ESLint 配置
│   ├── index.html  # 前端HTML模板
│   ├── package-lock.json  # 前端锁定依赖
│   ├── package.json  # 前端依赖声明
│   ├── postcss.config.js  # PostCSS 配置
│   ├── README.md  # 前端说明
│   ├── tailwind.config.js  # Tailwind 配置
│   ├── tsconfig.app.json  # TS 编译配置（应用）
│   ├── tsconfig.json  # TS 基础配置
│   ├── tsconfig.node.json  # TS Node 配置
│   └── vite.config.ts  # Vite 配置
├── .gitignore  # Git 忽略配置
├── env_before_fix.txt  # 环境修复前记录
├── git提交覆盖方法.txt  # Git 覆盖提交说明
├── requirements.txt  # 项目依赖占位
└── 每日运行脚本.txt  # 每日任务说明

```

<!-- CODEX_UPDATE_DIRECTORY_STRUCTURE -->

```text
quant-a-share/  # ?????
??? backend/  # ??????????
?   ??? __pycache__/  # ?????
?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ??? data_pipeline/  # ??????
?   ?   ??? __pycache__/  # ??????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? feature_engineering.cpython-310.pyc  # ??????
?   ?   ??? __init__.py  # ????????
?   ?   ??? feature_engineering.py  # ????????
?   ??? tft_model/  # TFT ??????
?   ?   ??? __pycache__/  # TFT ????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? api.cpython-310.pyc  # API ??
?   ?   ?   ??? predict.cpython-310.pyc  # ??????
?   ?   ?   ??? train.cpython-310.pyc  # ??????
?   ?   ?   ??? utils.cpython-310.pyc  # ??????
?   ?   ??? data/  # ??????
?   ?   ?   ??? features_sh_600519.csv  # ????CSV
?   ?   ?   ??? features_sh_600519.parquet  # ????Parquet
?   ?   ??? models/  # ????
?   ?   ?   ??? tft-epoch=02-val_loss=1.0177.ckpt  # ?????
?   ?   ?   ??? tft-epoch=02-val_loss=1.0203.ckpt  # ?????
?   ?   ??? __init__.py  # ??????
?   ?   ??? api.py  # ??????
?   ?   ??? predict.py  # ????
?   ?   ??? train.py  # ????
?   ?   ??? utils.py  # ??????
?   ??? __init__.py  # ??????
?   ??? README.md  # ??????
?   ??? requirements.txt  # ??????
??? config/  # ????
?   ??? settings.yaml  # ??????
??? data_cache/  # ??????
?   ??? index_sh-000001_2024-12-08_2025-12-08_d.csv  # ????????
?   ??? index_sh-000001_2024-12-09_2025-12-09_d.csv  # ????????
?   ??? market_data.sqlite  # ??SQLite???
?   ??? stock_sh-600519_2015-01-01_2025-12-09_d.csv  # ????????
?   ??? stock_sh-600519_2023-01-01_2024-12-31_d.csv  # ????????
?   ??? stock_sh-600519_2024-01-01_2024-12-31_d.csv  # ????????
?   ??? stock_sh-600519_2024-01-01_2025-12-07_d.csv  # ????????
?   ??? stock_sh-600519_2024-01-01_2025-12-09_d.csv  # ????????
?   ??? stock_sh-688017_2025-01-01_2025-12-09_d.csv  # ????????
?   ??? stock_sh-688192_2024-01-01_2025-12-09_d.csv  # ????????
?   ??? stock_sh-688192_2024-12-10_2025-12-10_d.csv  # ????????
?   ??? stock_sh-688192_2025-01-01_2025-12-09_d.csv  # ????????
?   ??? stock_sz-002460_2025-01-01_2025-12-09_d.csv  # ????????
??? docs/  # ????
?   ??? api_reference.md  # API ??
?   ??? codex_context.md  # Codex ?????
??? google-ai-webui/  # Google AI Studio ????
?   ??? .env.local  # ????????
??? notebooks/  # Notebook ??
?   ??? data_cache/  # Notebook ????
?   ?   ??? market_data.sqlite  # Notebook ?????
?   ??? 01_data_and_market_overview.ipynb  # ???????Notebook
?   ??? ??.ipynb  # ????Notebook
??? quant_system/  # ????????
?   ??? __pycache__/  # ???
?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ??? backtest/  # ????
?   ?   ??? __pycache__/  # ??????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ?   ?   ??? engine.cpython-310.pyc  # ????
?   ?   ?   ??? engine.cpython-313.pyc  # ????
?   ?   ?   ??? performance.cpython-310.pyc  # ??????
?   ?   ?   ??? performance.cpython-313.pyc  # ??????
?   ?   ??? __init__.py  # ??????
?   ?   ??? engine.py  # ??????
?   ?   ??? performance.py  # ??????
?   ??? data/  # ???????
?   ?   ??? __pycache__/  # ??????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ?   ?   ??? fetcher.cpython-310.pyc  # ??????
?   ?   ?   ??? fetcher.cpython-313.pyc  # ??????
?   ?   ?   ??? storage.cpython-310.pyc  # ??????
?   ?   ?   ??? storage.cpython-313.pyc  # ??????
?   ?   ??? __init__.py  # ??????
?   ?   ??? fetcher.py  # ??????
?   ?   ??? storage.py  # ??????
?   ??? indicators/  # ??????
?   ?   ??? __pycache__/  # ??????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ?   ?   ??? base_indicator.cpython-310.pyc  # ??????
?   ?   ?   ??? base_indicator.cpython-313.pyc  # ??????
?   ?   ?   ??? engine.cpython-310.pyc  # ??????
?   ?   ?   ??? engine.cpython-313.pyc  # ??????
?   ?   ??? plugins/  # ????
?   ?   ?   ??? __pycache__/  # ????
?   ?   ?   ?   ??? __init__.cpython-310.pyc  # ???????
?   ?   ?   ?   ??? __init__.cpython-313.pyc  # ???????
?   ?   ?   ?   ??? macd.cpython-310.pyc  # MACD ??
?   ?   ?   ?   ??? macd.cpython-313.pyc  # MACD ??
?   ?   ?   ?   ??? moving_average.cpython-310.pyc  # ????
?   ?   ?   ?   ??? moving_average.cpython-313.pyc  # ????
?   ?   ?   ?   ??? rsi.cpython-310.pyc  # RSI ??
?   ?   ?   ?   ??? rsi.cpython-313.pyc  # RSI ??
?   ?   ?   ?   ??? ultimate_features.cpython-310.pyc  # ??????
?   ?   ?   ?   ??? ultimate_features.cpython-313.pyc  # ??????
?   ?   ?   ??? __init__.py  # ??????
?   ?   ?   ??? macd.py  # MACD ??
?   ?   ?   ??? moving_average.py  # ????
?   ?   ?   ??? rsi.py  # RSI ??
?   ?   ?   ??? ultimate_features.py  # ??????
?   ?   ??? __init__.py  # ??????
?   ?   ??? base_indicator.py  # ??????
?   ?   ??? engine.py  # ??????
?   ??? processing/  # ?????????
?   ?   ??? __pycache__/  # ??????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ?   ?   ??? cleaner.cpython-310.pyc  # ??????
?   ?   ?   ??? cleaner.cpython-313.pyc  # ??????
?   ?   ?   ??? industry_sentiment.cpython-310.pyc  # ??????
?   ?   ?   ??? industry_sentiment.cpython-313.pyc  # ??????
?   ?   ?   ??? market_view.cpython-310.pyc  # ??????
?   ?   ?   ??? market_view.cpython-313.pyc  # ??????
?   ?   ??? __init__.py  # ??????
?   ?   ??? cleaner.py  # ??????
?   ?   ??? industry_sentiment.py  # ??????
?   ?   ??? market_view.py  # ??????
?   ??? strategy/  # ?????
?   ?   ??? __pycache__/  # ??????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ?   ?   ??? base_strategy.cpython-310.pyc  # ??????
?   ?   ?   ??? base_strategy.cpython-313.pyc  # ??????
?   ?   ?   ??? registry.cpython-310.pyc  # ??????
?   ?   ?   ??? registry.cpython-313.pyc  # ??????
?   ?   ??? plugins/  # ????
?   ?   ?   ??? __pycache__/  # ????
?   ?   ?   ?   ??? __init__.cpython-310.pyc  # ???????
?   ?   ?   ?   ??? __init__.cpython-313.pyc  # ???????
?   ?   ?   ?   ??? connors_rsi2.cpython-310.pyc  # CRS 2 ??
?   ?   ?   ?   ??? connors_rsi2.cpython-313.pyc  # CRS 2 ??
?   ?   ?   ?   ??? ma_rsi_long_only.cpython-310.pyc  # MA+RSI ??
?   ?   ?   ?   ??? ma_rsi_long_only.cpython-313.pyc  # MA+RSI ??
?   ?   ?   ??? __init__.py  # ??????
?   ?   ?   ??? connors_rsi2.py  # ConnorsRSI2 ??
?   ?   ?   ??? ma_rsi_long_only.py  # MA+RSI ????
?   ?   ??? __init__.py  # ??????
?   ?   ??? base_strategy.py  # ????
?   ?   ??? registry.py  # ?????
?   ??? visualization/  # ?????
?   ?   ??? __pycache__/  # ?????
?   ?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ?   ?   ??? plotting.cpython-310.pyc  # ????
?   ?   ?   ??? plotting.cpython-313.pyc  # ????
?   ?   ??? __init__.py  # ???????
?   ?   ??? plotting.py  # ??????
?   ??? __init__.py  # ????????
??? scripts/  # ????
?   ??? calc_industry_sentiment.py  # ????????
?   ??? download_all_daily_since_2015.py  # ????????
?   ??? generate_api_docs.py  # ????API??
?   ??? run_backtest_demo.py  # ??????
?   ??? test_indicators_basic.py  # ??????
?   ??? test_prepare_tft.py  # TFT??????
?   ??? update_codex_directory_structure.py  # Codex??????
?   ??? update_daily_data.py  # ????????
?   ??? update_directory_structure.py  # ????????
?   ??? update_industry_mapping.py  # ????????
??? tests/  # ??????
?   ??? __init__.py  # ??????
??? web_api/  # Python ????
?   ??? __pycache__/  # Web API ??
?   ?   ??? __init__.cpython-310.pyc  # ??????
?   ?   ??? __init__.cpython-313.pyc  # ??????
?   ?   ??? main.cpython-310.pyc  # ?????
?   ?   ??? main.cpython-313.pyc  # ?????
?   ??? __init__.py  # Web API ????
?   ??? main.py  # Web API ??
?   ??? web_api.md  # Web API ??
??? webui/  # ?? React ??
?   ??? node_modules/  # ???????????
?   ??? public/  # ????
?   ?   ??? vite.svg  # Vite ??
?   ??? src/  # ?????
?   ?   ??? api/  # ??API??
?   ?   ?   ??? backtest.ts  # ??????
?   ?   ??? assets/  # ????
?   ?   ?   ??? react.svg  # React ??
?   ?   ??? components/  # ????
?   ?   ?   ??? AIAnalyst.tsx  # AI ????
?   ?   ?   ??? BacktestPanel.tsx  # ????
?   ?   ?   ??? HoldingsTable.tsx  # ????
?   ?   ?   ??? IndustrySentimentTable.tsx  # ?????
?   ?   ?   ??? MarketChart.tsx  # ?????
?   ?   ?   ??? MarketOverviewCard.tsx  # ??????
?   ?   ?   ??? MarketStats.tsx  # ??????
?   ?   ?   ??? NewsPage.tsx  # ????
?   ?   ?   ??? StockForecastPage.tsx  # ?????
?   ?   ?   ??? StrategyBacktestPage.tsx  # ?????
?   ?   ?   ??? StrategyPanel.tsx  # ??????
?   ?   ?   ??? StrategySettingsPage.tsx  # ?????
?   ?   ??? services/  # ??????
?   ?   ?   ??? geminiService.ts  # Gemini API ???
?   ?   ??? App.css  # ????
?   ?   ??? App.tsx  # ??????
?   ?   ??? index.css  # ????
?   ?   ??? main.tsx  # ????
?   ?   ??? types.ts  # ????
?   ??? .gitignore  # ??????
?   ??? eslint.config.js  # ESLint ??
?   ??? index.html  # ??HTML??
?   ??? package-lock.json  # ??????
?   ??? package.json  # ??????
?   ??? postcss.config.js  # PostCSS ??
?   ??? README.md  # ????
?   ??? tailwind.config.js  # Tailwind ??
?   ??? tsconfig.app.json  # TS ????????
?   ??? tsconfig.json  # TS ????
?   ??? tsconfig.node.json  # TS Node ??
?   ??? vite.config.ts  # Vite ??
??? .gitignore  # Git ????
??? env_before_fix.txt  # ???????
??? git??????.txt  # Git ??????
??? requirements.txt  # ??????
??? ??????.txt  # ??????
```

<!-- CODEX_UPDATE_DIRECTORY_STRUCTURE -->

```text
quant-a-share/  # 项目根目录
├── .gitignore  # Git 忽略配置
├── config/  # 全局配置
│   └── settings.yaml  # 系统参数示例
├── data_cache/  # 本地行情缓存与数据库
│   ├── index_sh-000001_2024-12-08_2025-12-08_d.csv  # 上证指数日线缓存
│   ├── market_data.sqlite  # 行情 SQLite 缓存库
│   ├── stock_sh-600519_2023-01-01_2024-12-31_d.csv  # 贵州茅台日线缓存（23-24）
│   └── stock_sh-600519_2024-01-01_2024-12-31_d.csv  # 贵州茅台日线缓存（2024）
├── docs/  # 项目文档
│   ├── api_reference.md  # API 参考
│   └── codex_context.md  # Codex 上下文说明
├── google-ai-webui/  # Google AI Studio 生成的前端样例
│   ├── components/  # 页面组件库
│   │   ├── AIAnalyst.tsx  # AI 分析组件
│   │   ├── BacktestPanel.tsx  # 回测控制面板
│   │   ├── HoldingsTable.tsx  # 持仓表格
│   │   ├── MarketChart.tsx  # 行情图表
│   │   ├── MarketStats.tsx  # 市场统计卡片
│   │   ├── NewsPage.tsx  # 新闻页面
│   │   ├── StockForecastPage.tsx  # 个股预测页
│   │   ├── StrategyBacktestPage.tsx  # 策略回测页
│   │   ├── StrategyPanel.tsx  # 策略选择面板
│   │   └── StrategySettingsPage.tsx  # 策略参数配置
│   ├── services/  # 前端服务封装
│   │   └── geminiService.ts  # Gemini API 调用
│   ├── src/  # 精简版前端源码
│   │   ├── components/  # 组件子集
│   │   │   ├── IndustrySentimentTable.tsx  # 行业情绪表
│   │   │   ├── MarketOverviewCard.tsx  # 市场概览卡片
│   │   │   ├── StockForecastPage.tsx  # 个股预测页（精简版）
│   │   │   └── StrategySettingsPage.tsx  # 策略配置页（精简版）
│   │   ├── App.css  # 样式文件
│   │   ├── App.tsx  # 入口页面
│   │   └── types.ts  # 类型定义（src）
│   ├── .gitignore  # 前端忽略配置
│   ├── App.tsx  # 入口组件（根目录版）
│   ├── index.html  # 静态模板
│   ├── index.tsx  # 入口挂载
│   ├── metadata.json  # 元数据配置
│   ├── package.json  # 前端依赖声明
│   ├── README.md  # 说明文档
│   ├── tsconfig.json  # TypeScript 配置
│   ├── types.ts  # 类型定义（根目录）
│   └── vite.config.ts  # Vite 配置
├── notebooks/  # Notebook 示例与实验
│   ├── data_cache/  # Notebook 专用缓存（当前为空）
│   ├── 01_data_and_market_overview.ipynb  # 数据与市场概览示例
│   ├── market_data.sqlite  # Notebook 占位数据库
│   └── 测试.ipynb  # 临时测试笔记
├── quant_system/  # 核心量化系统代码
│   ├── __pycache__/  # 顶层包编译缓存
│   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   └── __init__.cpython-313.pyc  # 缓存
│   ├── backtest/  # 回测引擎
│   │   ├── __pycache__/  # 回测模块编译缓存
│   │   │   ├── engine.cpython-310.pyc  # 缓存
│   │   │   ├── engine.cpython-313.pyc  # 缓存
│   │   │   ├── performance.cpython-310.pyc  # 缓存
│   │   │   └── performance.cpython-313.pyc  # 缓存
│   │   ├── __init__.py  # 回测包初始化
│   │   ├── engine.py  # 日线回测引擎
│   │   └── performance.py  # 绩效统计
│   ├── data/  # 数据获取与存储
│   │   ├── __pycache__/  # 数据模块编译缓存
│   │   │   ├── fetcher.cpython-310.pyc  # 缓存
│   │   │   ├── fetcher.cpython-313.pyc  # 缓存
│   │   │   ├── storage.cpython-310.pyc  # 缓存
│   │   │   ├── storage.cpython-313.pyc  # 缓存
│   │   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   │   └── __init__.cpython-313.pyc  # 缓存
│   │   ├── __init__.py  # 数据包初始化
│   │   ├── fetcher.py  # 数据抓取与缓存
│   │   └── storage.py  # CSV/SQLite 存储封装
│   ├── indicators/  # 技术指标引擎
│   │   ├── plugins/  # 指标插件
│   │   │   ├── __pycache__/  # 指标插件编译缓存
│   │   │   │   ├── macd.cpython-310.pyc  # 缓存
│   │   │   │   ├── macd.cpython-313.pyc  # 缓存
│   │   │   │   ├── moving_average.cpython-310.pyc  # 缓存
│   │   │   │   ├── moving_average.cpython-313.pyc  # 缓存
│   │   │   │   ├── rsi.cpython-310.pyc  # 缓存
│   │   │   │   ├── rsi.cpython-313.pyc  # 缓存
│   │   │   │   ├── ultimate_features.cpython-310.pyc  # 缓存
│   │   │   │   ├── ultimate_features.cpython-313.pyc  # 缓存
│   │   │   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   │   │   └── __init__.cpython-313.pyc  # 缓存
│   │   │   ├── __init__.py  # 插件注册
│   │   │   ├── macd.py  # MACD 指标
│   │   │   ├── moving_average.py  # 均线指标
│   │   │   ├── rsi.py  # RSI 指标
│   │   │   └── ultimate_features.py  # 终极特征集合
│   │   ├── __pycache__/  # 指标模块编译缓存
│   │   │   ├── base_indicator.cpython-310.pyc  # 缓存
│   │   │   ├── base_indicator.cpython-313.pyc  # 缓存
│   │   │   ├── engine.cpython-310.pyc  # 缓存
│   │   │   ├── engine.cpython-313.pyc  # 缓存
│   │   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   │   └── __init__.cpython-313.pyc  # 缓存
│   │   ├── __init__.py  # 指标包初始化
│   │   ├── base_indicator.py  # 指标基类
│   │   └── engine.py  # 指标注册与计算入口
│   ├── processing/  # 数据清洗与市场概况
│   │   ├── __pycache__/  # 处理模块编译缓存
│   │   │   ├── cleaner.cpython-310.pyc  # 缓存
│   │   │   ├── cleaner.cpython-313.pyc  # 缓存
│   │   │   ├── industry_sentiment.cpython-310.pyc  # 缓存
│   │   │   ├── industry_sentiment.cpython-313.pyc  # 缓存
│   │   │   ├── market_view.cpython-310.pyc  # 缓存
│   │   │   ├── market_view.cpython-313.pyc  # 缓存
│   │   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   │   └── __init__.cpython-313.pyc  # 缓存
│   │   ├── __init__.py  # 处理包初始化
│   │   ├── cleaner.py  # 行情清洗与填补
│   │   ├── industry_sentiment.py  # 行业情绪计算
│   │   └── market_view.py  # 市场情绪概览
│   ├── strategy/  # 策略框架
│   │   ├── plugins/  # 策略插件
│   │   │   ├── __pycache__/  # 策略插件编译缓存
│   │   │   │   ├── ma_rsi_long_only.cpython-310.pyc  # 缓存
│   │   │   │   ├── ma_rsi_long_only.cpython-313.pyc  # 缓存
│   │   │   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   │   │   └── __init__.cpython-313.pyc  # 缓存
│   │   │   ├── __init__.py  # 策略插件注册
│   │   │   └── ma_rsi_long_only.py  # 均线+RSI 多头策略
│   │   ├── __pycache__/  # 策略模块编译缓存
│   │   │   ├── base_strategy.cpython-310.pyc  # 缓存
│   │   │   ├── base_strategy.cpython-313.pyc  # 缓存
│   │   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   │   └── __init__.cpython-313.pyc  # 缓存
│   │   ├── __init__.py  # 策略包初始化
│   │   └── base_strategy.py  # 策略基类
│   ├── visualization/  # 可视化模块
│   │   ├── __pycache__/  # 可视化编译缓存
│   │   │   ├── plotting.cpython-310.pyc  # 缓存
│   │   │   ├── plotting.cpython-313.pyc  # 缓存
│   │   │   ├── __init__.cpython-310.pyc  # 缓存
│   │   │   └── __init__.cpython-313.pyc  # 缓存
│   │   ├── __init__.py  # 可视化包初始化
│   │   └── plotting.py  # 绘图入口
│   └── __init__.py  # 量化系统包初始化
├── scripts/  # 工具与批处理脚本
│   ├── calc_industry_sentiment.py  # 行业情绪计算
│   ├── download_all_daily_since_2015.py  # 全量日线下载
│   ├── generate_api_docs.py  # 生成 API 文档
│   ├── run_backtest_demo.py  # 回测示例
│   ├── test_indicators_basic.py  # 指标基础测试
│   ├── update_codex_directory_structure.py  # Codex 目录更新脚本
│   ├── update_daily_data.py  # 更新日线数据
│   ├── update_directory_structure.py  # 自动生成目录树
│   └── update_industry_mapping.py  # 更新行业映射
├── tests/  # 测试占位
│   └── __init__.py  # 测试包初始化
├── web_api/  # Web API 服务
│   ├── __pycache__/  # 接口模块编译缓存
│   │   ├── main.cpython-310.pyc  # 缓存
│   │   └── __init__.cpython-310.pyc  # 缓存
│   ├── main.py  # FastAPI 服务入口
│   ├── web_api.md  # Web API 说明
│   └── __init__.py  # 包初始化
├── webui/  # 前端 React+TS+Vite 项目
│   ├── node_modules/  # 前端依赖（已安装，未展开）
│   ├── public/  # 静态资源
│   │   └── vite.svg  # Vite 图标
│   ├── src/  # 前端源码
│   │   ├── assets/  # 静态资产
│   │   │   └── react.svg  # React 标识
│   │   ├── components/  # 页面组件
│   │   │   ├── AIAnalyst.tsx  # AI 分析组件
│   │   │   ├── BacktestPanel.tsx  # 回测面板
│   │   │   ├── HoldingsTable.tsx  # 持仓表格
│   │   │   ├── IndustrySentimentTable.tsx  # 行业情绪表
│   │   │   ├── MarketChart.tsx  # 行情图表
│   │   │   ├── MarketOverviewCard.tsx  # 市场概览卡片
│   │   │   ├── MarketStats.tsx  # 市场统计卡片
│   │   │   ├── NewsPage.tsx  # 新闻页面
│   │   │   ├── StockForecastPage.tsx  # 个股预测页
│   │   │   ├── StrategyBacktestPage.tsx  # 策略回测页
│   │   │   ├── StrategyPanel.tsx  # 策略面板
│   │   │   └── StrategySettingsPage.tsx  # 策略参数配置
│   │   ├── services/  # 前端服务封装
│   │   │   └── geminiService.ts  # Gemini 请求封装
│   │   ├── types.ts  # 类型定义
│   │   ├── App.css  # 全局样式
│   │   ├── App.tsx  # 前端入口组件
│   │   ├── index.css  # 全局样式入口
│   │   └── main.tsx  # 前端入口挂载
│   ├── .gitignore  # 前端忽略配置
│   ├── eslint.config.js  # ESLint 配置
│   ├── index.html  # Vite 模板
│   ├── package-lock.json  # 依赖锁定
│   ├── package.json  # 依赖声明
│   ├── postcss.config.js  # PostCSS 配置
│   ├── README.md  # 前端说明
│   ├── tailwind.config.js  # Tailwind 配置
│   ├── tsconfig.app.json  # TS 应用配置
│   ├── tsconfig.json  # TS 基础配置
│   ├── tsconfig.node.json  # TS Node 配置
│   └── vite.config.ts  # Vite 配置
├── requirements.txt  # Python 依赖清单
└── 每日运行脚本.txt  # 每日运行说明
```

<!-- CODEX_UPDATE_DIRECTORY_STRUCTURE -->
说明： 上述结构中标注“（计划）”的模块可能尚未实现，但预留了扩展空间。Codex 在未来生成相关代码时，应将文件置于相应目录下。目录组织体现了各模块的边界：Codex 切勿将不属于该模块的代码杂糅进去。例如，不要在策略模块直接写数据下载代码，而应通过数据模块接口获取数据。
接口设计与编码风格
为确保模块解耦和交互顺畅，各模块通过明确的接口通信[32]。Codex 在编写代码时应遵循以下接口设计原则和命名风格：
•	函数接口命名： 函数名清晰表达功能，采用动宾短语。如数据获取 get_stock_data(code, start_date, end_date)，市场指标计算 calc_market_overview(data)，指标引擎 calculate_indicators(data)，风险检查 check_order(order)，仓位计算 allocate(capital, signal) 等[32]。命名采用全小写，单词间以下划线分隔。
•	类接口设计： 各模块核心类以名词或名词短语命名，采用大写驼峰风格。如 RiskManager，PositionSizer，MACDIndicator，MovingAverageCrossoverStrategy 等。基础接口类命名以 Base 前缀起头，如 BaseStrategy，BaseIndicator，在其中定义通用方法签名。派生类应覆盖父类方法实现各自逻辑。
•	参数与返回值： 接口定义尽量简单、稳定。参数使用内置类型或标准数据结构（如 Pandas DataFrame/Numpy 数组），返回值清晰明了。举例来说，get_stock_data 返回 DataFrame，索引为日期，列包含行情字段；generate_signals 返回与输入数据等长的信号序列（list或np.array）。Codex 需确保为每个公开接口编写文档字符串，说明参数意义、返回格式和异常情况，方便将来参考和维护。
•	模块内部实现隐藏： 其他模块只调用接口而不关心内部实现。例如策略获取数据应通过数据模块提供的函数，不应直接依赖 BaoStock API；回测调用风控只通过 RiskManager.check_order 接口，不需要了解风控如何判断[32]。Codex 生成代码时应坚持这一思想，将实现细节封装在模块内部，暴露必要的接口给外部使用。
•	命名一致性： 避免不同模块出现含义相同但命名不一致的情况。Codex 应熟悉已有命名，例如已经存在 market_view.py 计算市场概况，那么不要再创建 market_analyzer.py 功能重复的模块；若添加类似功能，直接扩充或复用已有模块函数。统一的命名风格和术语有助于保持代码上下文一致，方便团队协作和 AI 理解。
此外，应严格遵循 PEP8 格式：缩进4空格，合理换行，避免过长函数。使用类型注解提升代码自描述性（如 def get_stock_data(code: str, start: str, end: str) -> pd.DataFrame:）。Codex 补全代码时也应维护此风格，使生成的代码无论由AI或人工编写，都保持风格统一。
Notebook 使用方式
本项目偏向使用 Jupyter Notebook 进行研究、可视化和结果展示。VS Code 中已安装 Jupyter 扩展，允许在编辑器直接运行 Notebook[33][34]。Codex 需了解 Notebook 在本项目中的作用和使用习惯：
•	交互式开发与调试： Notebook 被用于分步骤开发和调试策略。例如 notebooks/01_data_and_market_overview.ipynb 演示了如何获取数据并计算市场指标。Notebook 环境支持逐段运行代码，Codex 可以帮助将模块函数应用于示例数据，立即展示输出。这种即时反馈有助于验证模块功能是否正常[34]。Codex 在生成 Notebook 示例代码时，应确保每个单元独立可运行，并输出预期结果，方便开发者理解模块用法。
•	可视化集成： Notebook 便于将图表嵌入输出中[34]。开发者常在 Notebook 中调用可视化模块绘制K线图或净值曲线，以观察策略表现。Codex 生成绘图代码时，要适配 Notebook，例如使用 %matplotlib inline 或确保 plt.show() 被调用，使图像正确显示。对于交互式图表（如Plotly），Notebook 能直接渲染输出供用户交互浏览[27]。
•	记录实验过程： Notebook 通过 Markdown 单元可以夹叙夹议地记录策略思路、参数说明和结论[34]。开发者会在 Notebook 中写下分析笔记和结果解读，形成完整的实验报告。Codex 可以在需要时协助生成 Markdown 文本（如解释某段代码作用、结果含义），帮助撰写清晰的研究记录。
•	与模块代码协同： Notebook 主要用于调用已经实现的模块接口执行任务，而不会在其中编写大量业务逻辑代码。Codex 应鼓励将核心逻辑写入模块，通过 Notebook 调用模块接口实现流程。例如，在 Notebook 里调用 data.fetcher.get_stock_data 获取数据，再调用 indicators.engine.calculate_indicators 计算指标，然后调用策略模块生成信号，最后用回测模块评估。这样 Notebook 保持简洁，如同脚本串联各模块，逻辑清晰且易于复用。
•	运行环境约定： Notebook 默认在 quant_system 路径可导入项目模块（VS Code 配置了该环境）。Codex 需要确保导入路径正确，如 from quant_system.data import fetcher。另外，Notebook 运行涉及较大数据时，应关注性能和内存，Codex 可提示使用高效的方法（如向量化操作替代Python循环）提高 Notebook 执行效率[35]。
总之，Notebook 是本项目的重要开发和沟通工具。Codex 应协助生成既函数化（模块提供接口）又交互友好（Notebook 使用体验佳）的代码，使用户能通过 Notebook 顺畅地完成从数据->指标->策略->回测->可视化的全流程实验。
开发流程与优先级建议
为有序构建完整系统，建议按照由基础到高级的顺序分阶段开发[36]。Codex 在建议实现时，应考虑当前开发阶段的重点，遵循以下顺序逐步完善模块：
1.	数据获取与处理阶段： (优先级最高) 首先实现数据获取模块，然后完成数据处理和市场概况计算[37]。确保能够成功下载历史行情并进行基本清洗，验证数据准确性（例如用 Notebook 绘制价格曲线或输出涨跌家数检验)。此阶段建立可靠的数据基础，之后的指标和策略都依赖于此。[37]
2.	指标引擎与基础策略阶段： 在数据可用后，开发技术指标引擎，封装常用指标计算接口[38]。同时设计并实现指标/策略插件架构（创建插件目录、加载机制）[38]。随后可编写一两个简单策略（如均线交叉策略）作为示例，利用指标引擎输出信号来产生交易信号，为回测做好准备。[38]Codex 此阶段应专注于指标计算正确性和策略框架搭建，而非策略复杂度，验证解耦思路即可。
3.	回测框架阶段： 开发回测模块，实现基本的历史模拟交易流程[17]。一开始可以不引入复杂风控、资金管理，先做到能根据策略信号执行交易、跟踪持仓和资金变化，计算初步绩效指标。通过简单回测检验策略雏形效果。Codex 在此阶段生成代码时，应注意回测逻辑的正确性（如持仓更新、手续费计算）以及与策略接口的衔接。
4.	风控与资金管理阶段： 在基本回测跑通后，引入风险控制模块和资金管理模块，提高策略稳健性[39]。先实现关键风控规则（如单笔交易限额、最大回撤停止），以及简单仓位分配策略，然后将它们接入回测流程[17]。通过多次回测调优这些规则参数，找到策略在风险可控前提下的较优表现[39]。Codex 应帮助编写可配置的风控/资金管理代码，并确保它们通过接口与回测集成，而非硬编码在回测内部[40]。
5.	可视化与报告阶段： 最后，完善结果展示模块[41]。绘制出交易信号标注的K线图、回测净值曲线等，将策略表现可视化，方便分析和展示[41]。生成绩效报告，包括统计指标表格和策略评价。在此阶段，Codex 可专注于图形正确标绘（检查买卖点是否正确对齐行情）、报告数据准确计算，以及Notebook中展示的美观性。
每完成一个阶段，都应进行单元测试或使用 Notebook 验证模块功能是否满足预期[42]。这种循序渐进的方法确保“先搭地基再盖高楼”：先有数据，其次指标和策略，继而回测验证，最后完善风控和展示[43]。Codex 应根据当前开发重点调整建议，例如在早期更多关注数据准确性，中期确保架构连贯，后期提升用户体验。始终牢记各阶段成果都应集成到整体架构中，保持模块契合和风格一致。
扩展性与未来开发建议
为了保证本系统能够随时间演进、方便添加新功能，架构设计从一开始就强调高可扩展性和插件化[44]。Codex 在协助未来开发时，应遵循这些原则，指导新功能的实现方式：
•	新增指标插件： 当用户需要引入新的技术指标（例如某特定算法的信号），应新建一个脚本文件于 indicators/plugins/ 中，实现既定接口（如定义 compute_indicator(data) 或新建类继承 BaseIndicator）。主指标引擎通过扫描机制将其自动注册，无需修改引擎代码[44]。Codex 应确保提供的新指标代码符合接口要求，计算结果与现有DataFrame整合良好。例如，若新指标需要参数，可通过配置文件或在插件内定义默认参数，避免对引擎硬编码配置。
•	新增策略插件： 同理，引入新交易策略时，在 strategy/plugins/ 下创建新策略类，继承 BaseStrategy 并实现 generate_signals 方法[44]。策略所需的指标若未包含在指标引擎中，可同时编写对应指标插件，从而策略通过引擎获取所需数据。Codex 生成新策略代码时，应提醒开发者在配置文件中登记启用该策略（如果有策略选择配置），以及为策略编写基本说明文档，方便日后维护。
•	扩展数据源或市场： 如果未来需要支持新的数据源（如实时行情、其他市场品种）或增加数据频率（如分钟线），应在数据模块内扩展而不是改写现有代码。例如，新数据源可实现一个新的Fetcher类封装API，并根据需要在配置中选择使用哪个源。Codex 可以建议采用工厂模式或策略模式，根据配置提供不同的数据获取实现，但对外接口仍保持 get_stock_data 这类方法不变，确保其他模块不受影响。
•	添加风险控制/资金管理策略： 风控和资金管理模块本身也可进一步插件化。例如，可将不同风控规则拆成独立类或配置项，方便组合启用。新增风控规则时，可在 risk_manager.py 内以方法形式实现，并通过配置决定是否激活。资金管理亦可设计支持多策略：Codex 可建议实现如 PositionSizer 基类，不同算法继承并实现 allocate()，然后根据设置选择实例化哪一种。关键是新增规则不改变原有接口，使回测引擎调用风控/资金管理的代码无需修改。[45]
•	优化性能或引入并行： 随着数据量和策略复杂度提升，可能需要优化性能。Codex 可建议引入诸如 Numba 加速关键计算，或利用 multiprocessing 并行回测不同策略参数。在引入这些优化时，也应与架构契合，例如封装在工具模块，不要让并行逻辑散落各处。始终保证新增的优化层对外透明，以不修改主流程为前提提升效率。
•	借鉴成熟框架： 持续关注开源量化交易框架的新特性，将有用的概念融入本系统。例如 Backtrader 的 Observer/Analyzer 机制、VN.py 的事件总线与插件体系等[46]。Codex 可以在用户有此需求时，帮助评估如何将这些框架的优点整合进现有架构（如通过插件形式接入新的分析器，而非替换整个框架）。
•	保持文档更新： 每当新增模块或重要功能时，务必更新项目文档（包括本上下文指南和 README 等），描述新模块的目的、接口和用法。Codex 可以协助生成相应的文档草稿。完善的文档确保即使过一段时间再次阅读或他人加入，也能迅速理解系统设计思想和最新功能。
总体而言，Codex 在未来协助开发时，应始终贯彻开闭原则：对修改关闭、对扩展开放[47]。通过插件化、配置驱动和清晰接口，任何新模块的增加都不应破坏原有模块。每次扩展都像给系统“插入”一块新积木，而非推倒重来。只有这样，项目才能在保持稳定的同时不断演化。Codex 扮演的正是帮助搭建和衔接这些“新积木”的角色——在遵循架构蓝图的基础上，加速实现开发者的创意想法，同时确保架构大厦稳固长青。

**前端环境框架说明（React + TypeScript + Vite）**

一、使用的前端框架
前端采用 React + TypeScript 技术栈，支持 TSX 代码格式，适配 Google AI Studio 自动生成的组件代码。

二、构建工具
使用 Vite 作为构建与开发服务器工具，具有启动快、热更新快、适合现代前端工程开发等优点。

三、语言与文件格式
前端统一使用 TypeScript（.ts / .tsx）进行开发，所有组件采用 TSX 结构组织。

四、项目初始化方式（Vite）
创建项目命令：
npm create vite@latest webui

交互选项：
Framework: React
Variant: TypeScript + SWC
Use rolldown-vite?: No
Install with npm?: Yes

五、运行开发环境
进入 webui 目录，执行：
npm install
npm run dev
默认运行地址：
http://localhost:5173/

六、组件与代码结构兼容性
前端可直接兼容 Google AI Studio 生成的以下内容：
- App.tsx
- 各类组件（components/*.tsx）
- 服务类文件（services/*.ts）
- 类型定义文件（types.ts）
这些文件可直接放入 src/ 目录中运行。

七、可选扩展（按需使用）
- 样式框架：Tailwind CSS 或其他 CSS-in-JS
- UI 组件库：shadcn/ui、Material UI、Ant Design（任选其一）
- 图表库（如需要 K 线/市场可视化）：Recharts 或 ECharts for React
- 网络请求库：axios 或 Fetch API

八、与后端的集成方式
前端通过 HTTP API （例如 axios 或 fetch）连接 Python 后端（FastAPI / Flask）。
示例：
fetch("http://localhost:8000/api/xxx")

九、支持的使用场景
- 量化系统多页面控制台
- 大屏式数据可视化
- 策略面板与交互模块
- 股票分析功能界面
- Google AI Studio 自动生成的界面快速接入

十、核心定位
本前端环境用于构建现代化量化交易系统可视化界面，将 Google AI Studio 的智能生成 UI 与本地量化后端无缝整合。


总结
本指南为 A股量化分析系统项目的架构设计和开发规范提供了全面描述，旨在作为 OpenAI Codex 在 VS Code 中的长期参考上下文。通过严格遵循模块边界、插件机制、接口规范和编码风格，Codex 将能够持续产出高质量、与系统设计契合的代码，加速项目开发进程。开发者与 Codex 共同协作时，应反复参考此文档确保方向正确：从模块职责到目录布局，从函数命名到接口设计，每个细节的一致性累积起来就是项目的整体可靠性和可维护性。
凭借明确的架构和阶段化的开发路线，本系统将逐步完善各项功能，实现数据驱动的策略研究与回测。未来，无论新增何种策略、指标或功能，皆可在既有框架上无缝扩展。希望 Codex 在了解本上下文后，发挥智能补全和建议优势，成为开发者值得信赖的“AI拍档”，共同打造一个模块清晰、运行稳定、易于扩展的量化分析平台。
