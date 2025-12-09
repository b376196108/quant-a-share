// webui/src/api/backtest.ts
// 单标的多策略回测相关的前端 API 封装

// 组合方式，与后端 BacktestEngine 中的定义保持一致
export type CombinationMode = 'AND' | 'OR' | 'VOTING';

// ------- 策略元信息（用于左侧策略列表）-------

export interface StrategyMeta {
  id: string;                         // "connors_rsi2"
  name: string;                       // "Connors RSI(2) 极限反转"
  category: string;                   // "trend" | "reversal" | "volatility" | "volume" | ...
  description?: string;
  tags?: string[];
  default_params?: Record<string, any>;
  param_schema?: Record<string, any>;
}

// ------- 回测请求/响应的数据结构 -------

export interface BacktestStrategyIn {
  id: string;                         // 策略 id
  params: Record<string, any>;        // 对该策略的参数覆盖（不填则用 default_params）
}

export interface BacktestRequest {
  symbol: string;                     // 股票代码：建议 6 位数字，例如 "600519"
  start_date: string;                 // "YYYY-MM-DD"
  end_date: string;                   // "YYYY-MM-DD"
  strategies: BacktestStrategyIn[];   // 勾选的策略列表
  mode: CombinationMode;              // "AND" | "OR" | "VOTING"
  initial_capital: number;            // 初始资金
  fee_rate_bps: number;               // 手续费（万分比），例如 2.5 = 万分之 2.5
  slippage: number;                   // 每股滑点（元）
}

export interface BacktestStats {
  total_return: number;
  annual_return: number;
  max_drawdown: number;
  sharpe: number;
  // 预留：后端如果再加别的指标（胜率、交易次数等），这里也能接住
  [k: string]: number;
}

export interface EquityPoint {
  date: string;                       // "YYYY-MM-DD"
  equity: number;                     // 账户权益
}

export interface TradeRecord {
  date: string | null;                // "YYYY-MM-DD" 或 null
  action: 'buy' | 'sell' | string;    // 买 / 卖
  price: number;
  shares: number;
  fee: number;
  cash_after: number;
  position_after: number;
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TradeStats {
  win_rate?: number;       // 0~1
  trade_count?: number;
}

export interface BacktestResponse {
  symbol_input: string;               // 前端原始输入，例如 "600519"
  symbol: string;                     // 后端规范化后的代码，例如 "sh.600519"
  start_date: string;
  end_date: string;
  mode: CombinationMode;
  strategy_ids: string[];
  stats: BacktestStats;
  equity_curve: EquityPoint[];
  trades: TradeRecord[];
  price_series?: PricePoint[];
  trade_stats?: TradeStats;
}

// ------- API 基础配置 -------

// 本地开发默认后端地址；如果你后面有全局配置，可以改成从环境变量读取
const API_BASE = 'http://localhost:8000';

// ------- 对外暴露的服务函数 -------

/**
 * 获取当前所有可用的策略元信息，用于左侧策略选择面板。
 * 对应后端：GET /api/backtest/strategies
 */
export async function fetchStrategies(): Promise<StrategyMeta[]> {
  const resp = await fetch(`${API_BASE}/api/backtest/strategies`);

  if (!resp.ok) {
    throw new Error(`fetchStrategies failed: HTTP ${resp.status}`);
  }

  const data = (await resp.json()) as StrategyMeta[];
  return data;
}

/**
 * 运行单标的多策略组合回测。
 * 对应后端：POST /api/backtest/run
 */
export async function runBacktest(
  payload: BacktestRequest,
): Promise<BacktestResponse> {
  const resp = await fetch(`${API_BASE}/api/backtest/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    const msg = text || `runBacktest failed: HTTP ${resp.status}`;
    throw new Error(msg);
  }

  const data = (await resp.json()) as BacktestResponse;
  return data;
}
