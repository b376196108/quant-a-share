export type MarketTrend = 'Bullish' | 'Bearish' | 'Neutral';

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

export type IndustrySentiment = Record<string, string | number>;

export type MarketOverview = Record<string, string | number>;
