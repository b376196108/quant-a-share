
export enum MarketTrend {
  BULLISH = 'Bullish',
  BEARISH = 'Bearish',
  NEUTRAL = 'Neutral'
}

export interface OHLCData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ma5: number;
  ma20: number;
}

// Deprecate simple StockData for OHLC
export type StockData = OHLCData;

export interface Position {
  symbol: string;
  name: string;
  amount: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
}

export interface Strategy {
  id: string;
  name: string;
  status: 'active' | 'paused' | 'stopped';
  returnRate: number;
  drawdown: number;
  sharpeRatio: number;
  description: string;
}

export interface AIChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
  isThinking?: boolean;
}

export interface IndustryData {
  name: string;
  change: number;
  sentiment: 'High' | 'Medium' | 'Low';
}

export interface MarketStatsData {
  limitUp: number; // 涨停
  up: number;
  flat: number;
  down: number;
  limitDown: number; // 跌停
  sentimentScore: number; // 0-100
}

export interface TradeRecord {
  id: string;
  date: string;
  type: 'BUY' | 'SELL';
  price: number;
  amount: number;
  profit?: number;
  profitPercent?: number;
}

export interface BacktestResult {
  symbol: string;
  startDate: string;
  endDate: string;
  totalReturn: number;
  totalProfit: number; // Added
  maxDrawdown: number;
  sharpeRatio: number; // Added
  winRate: number;
  trades: TradeRecord[];
  data: StockData[];
}

// --- New Types for Forecast & Settings ---

export interface StockForecastPoint {
  date: string;
  predicted_close: number;
  change_pct: number;
  lower?: number;
  upper?: number;
}

export interface StockForecastResponse {
  symbol: string;
  name?: string;
  last_date: string;
  last_close: number;
  method: string;
  forecast: StockForecastPoint[];
}

export interface StrategyParamSchema {
  key: string;
  label: string;
  type: "int" | "float" | "select" | "string";
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string | number; label: string }[];
}

export interface StrategyConfig {
  id: string;
  name: string;
  enabled: boolean;
  description?: string;
  params: Record<string, number | string>;
  schema: StrategyParamSchema[];
}

export interface MarketOverview {
  "交易日": string;
  "市场情绪": string;
  "上涨家数": number;
  "下跌家数": number;
  "上涨占比": number;
  "总成交额(亿元)": number;
  "总股票数": number;
  "涨停家数": number;
  "跌停家数": number;
  "平均涨幅(%)": number;
  "中位数涨幅(%)": number;
}

export interface IndustrySentiment {
  "行业": string;
  "情绪": string;
  "平均涨幅(%)": number;
  "中位涨幅(%)": number;
  "上涨占比": number;
  "总成交额(亿元)": number;
  "上涨家数": number;
  "下跌家数": number;
  "涨停家数": number;
  "跌停家数": number;
}

export interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  time: string;
  sentiment: 'Bullish' | 'Bearish' | 'Neutral';
  tags: string[];
}
