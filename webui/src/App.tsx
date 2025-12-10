import React, { useState, useEffect } from 'react';
import { LayoutDashboard, LineChart, Settings, Bell, Search, TrendingUp, Newspaper } from 'lucide-react';
import MarketChart from './components/MarketChart';
import StrategyPanel from './components/StrategyPanel';
import AIAnalyst from './components/AIAnalyst';
import MarketStats from './components/MarketStats';
import StrategyBacktestPage from './components/StrategyBacktestPage'; // Use new page component
import StockForecastPage from './components/StockForecastPage';
import StrategySettingsPage from './components/StrategySettingsPage';
import NewsPage from './components/NewsPage';
import type { StockData, Position, Strategy, IndustryData, MarketStatsData } from './types';

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ||
  'http://localhost:8000';

// ---------------------- 工具函数：后端字段兼容处理 ----------------------

/**
 * 将后端 /api/overview 返回的一行 JSON 映射为前端的 MarketStatsData
 * 做了多种字段名的兜底，避免列名稍有不同就直接报错。
 */
const mapOverviewToMarketStats = (
  overview: Record<string, unknown> | null | undefined,
  fallback: MarketStatsData,
): MarketStatsData => {
  if (!overview || typeof overview !== 'object') {
    return fallback;
  }

  const toNumber = (value: unknown, fallbackValue = 0): number => {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallbackValue;
  };

  const total = toNumber(
    overview['总股票数'] ?? overview['total_stocks'] ?? overview['stock_count'],
    fallback.up + fallback.down + fallback.flat,
  );

  const up = toNumber(
    overview['上涨家数'] ?? overview['up_count'] ?? overview['up_stocks'],
    fallback.up,
  );

  const down = toNumber(
    overview['下跌家数'] ?? overview['down_count'] ?? overview['down_stocks'],
    fallback.down,
  );

  const limitUp = toNumber(
    overview['涨停家数'] ?? overview['limit_up'] ?? overview['limit_up_count'],
    fallback.limitUp,
  );

  const limitDown = toNumber(
    overview['跌停家数'] ?? overview['limit_down'] ?? overview['limit_down_count'],
    fallback.limitDown,
  );

  const upRatioRaw =
    overview['上涨占比'] ??
    overview['up_ratio'] ??
    overview['up_pct'] ??
    null;

  // 上涨占比：有的返回 0~1，有的返回 0~100，这里做个兼容
  let sentimentScore = fallback.sentimentScore;
  if (typeof upRatioRaw === 'number') {
    sentimentScore = upRatioRaw <= 1 ? Math.round(upRatioRaw * 100) : Math.round(upRatioRaw);
  }

  const flat = Math.max(total - up - down, 0);

  return {
    limitUp,
    up,
    flat,
    down,
    limitDown,
    sentimentScore,
  };
};

/**
 * 将 /api/industry-sentiment 返回的数组映射为 IndustryData[]
 */
const mapIndustryList = (
  list: Array<Record<string, unknown>> | null | undefined,
  fallback: IndustryData[],
): IndustryData[] => {
  if (!Array.isArray(list) || list.length === 0) {
    return fallback;
  }

  const mapSentimentLabel = (label: unknown, change: number): 'High' | 'Medium' | 'Low' => {
    if (typeof label === 'string') {
      if (label.includes('高潮') || label.includes('高涨') || label.includes('高')) return 'High';
      if (label.includes('冰点') || label.includes('低迷') || label.includes('低')) return 'Low';
      if (label.includes('普通') || label.includes('中')) return 'Medium';
    }
    if (change >= 1) return 'High';
    if (change <= -1) return 'Low';
    return 'Medium';
  };

  try {
    const mapped = list.map((item) => {
      const name =
        item['行业'] ??
        item['行业名称'] ??
        item['name'] ??
        item['industry_name'] ??
        '未知行业';

      const change =
        item['平均涨幅(%)'] ??
        item['平均涨跌幅(%)'] ??
        item['avg_change_pct'] ??
        item['change'] ??
        0;

      const sentimentRaw =
        item['情绪'] ??
        item['sentiment'] ??
        item['情绪标签'] ??
        '';

      const changeNum = Number(change) || 0;
      const sentiment = mapSentimentLabel(sentimentRaw, changeNum);

      return {
        name: String(name),
        change: changeNum,
        sentiment,
      } as IndustryData;
    });

    const sentimentWeight = { High: 2, Medium: 1, Low: 0 } as const;
    return mapped.sort((a, b) => {
      const sa = sentimentWeight[a.sentiment] ?? 0;
      const sb = sentimentWeight[b.sentiment] ?? 0;
      if (sa !== sb) return sb - sa;
      return (b.change || 0) - (a.change || 0);
    });
  } catch {
    return fallback;
  }
};

// ---------------------- K线模拟数据 ----------------------

const generateMarketData = (points: number): StockData[] => {
  let close = 3000;
  const data: StockData[] = [];
  const now = new Date();

  for (let i = 0; i < points; i++) {
    const time = new Date(now.getTime() - (points - i) * 24 * 60 * 60 * 1000);
    const volatility = 30;

    // Generate OHLC
    const open = close + (Math.random() - 0.5) * volatility;
    const move = (Math.random() - 0.5) * volatility * 2;
    close = open + move;
    const high = Math.max(open, close) + Math.random() * 10;
    const low = Math.min(open, close) - Math.random() * 10;

    data.push({
      time: time.toISOString().split('T')[0], // YYYY-MM-DD
      open,
      high,
      low,
      close,
      volume: Math.floor(Math.random() * 50000000) + 10000000,
      ma5: close + (Math.random() - 0.5) * 50,
      ma20: close + (Math.random() - 0.5) * 150,
    });
  }
  return data;
};

const mapIndexKlineToStockData = (rows: unknown[]): StockData[] => {
  if (!Array.isArray(rows)) return [];

  return rows
    .map((row) => {
      if (typeof row !== 'object' || row === null) {
        return null;
      }

      const record = row as Record<string, unknown>;
      const time =
        (record.time as string | undefined) ??
        (record.date as string | undefined) ??
        (record.trade_date as string | undefined) ??
        '';

      const toNumber = (v: unknown, fallback = 0): number => {
        const n = Number(v);
        return Number.isFinite(n) ? n : fallback;
      };

      const close = toNumber(record.close);

      return {
        time,
        open: toNumber(record.open),
        high: toNumber(record.high),
        low: toNumber(record.low),
        close,
        volume: toNumber(record.volume),
        ma5: toNumber(record.ma5, close),
        ma20: toNumber(record.ma20, close),
      };
    })
    .filter((d): d is StockData => !!d && !!d.time);
};

// ---------------------- 默认 mock 数据（作为兜底） ----------------------

const defaultIndustryData: IndustryData[] = [
  { name: '白酒', change: 2.35, sentiment: 'High' },
  { name: '新能源车', change: -1.2, sentiment: 'Low' },
  { name: '半导体', change: 0.85, sentiment: 'Medium' },
  { name: '银行', change: 0.45, sentiment: 'Medium' },
  { name: '医药生物', change: -0.65, sentiment: 'Low' },
];

const defaultMarketStats: MarketStatsData = {
  limitUp: 45,
  up: 2800,
  flat: 300,
  down: 1800,
  limitDown: 12,
  sentimentScore: 68,
};

// ---------------------- 初始持仓 / 策略 ----------------------

const initialPositions: Position[] = [
  {
    symbol: '600519',
    name: '贵州茅台',
    amount: 200,
    avgPrice: 1680.5,
    currentPrice: 1750.2,
    pnl: 13940.0,
    pnlPercent: 4.15,
  },
  {
    symbol: '300750',
    name: '宁德时代',
    amount: 500,
    avgPrice: 195.0,
    currentPrice: 188.5,
    pnl: -3250.0,
    pnlPercent: -3.33,
  },
];

const initialStrategies: Strategy[] = [
  {
    id: '1',
    name: '双均线趋势 (MA5/20)',
    status: 'active',
    returnRate: 15.5,
    drawdown: 5.2,
    sharpeRatio: 1.6,
    description: '日线MA5上穿MA20买入，下穿卖出。',
  },
  {
    id: '2',
    name: 'RSI 超卖反转',
    status: 'paused',
    returnRate: -2.1,
    drawdown: 6.8,
    sharpeRatio: 0.5,
    description: 'RSI < 30 时分批建仓，RSI > 70 止盈。',
  },
  {
    id: '3',
    name: '布林带突破',
    status: 'stopped',
    returnRate: 6.4,
    drawdown: 1.5,
    sharpeRatio: 2.1,
    description: '股价突破布林带上轨追涨。',
  },
];

// ---------------------- 主组件 ----------------------

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [marketData, setMarketData] = useState<StockData[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>(initialStrategies);
  const [positions] = useState<Position[]>(initialPositions);

  // 从后端获取上证指数 K 线数据，失败时回退到本地 mock
  useEffect(() => {
    const fetchIndexKline = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/index-kline?symbol=sh.000001&limit=90`);

        if (!res.ok) {
          console.warn('index-kline 接口返回非 200：', res.status);
          setMarketData(generateMarketData(60));
          return;
        }

        const json = await res.json();
        const series = mapIndexKlineToStockData(json);

        if (series.length > 0) {
          setMarketData(series);
        } else {
          console.warn('index-kline 返回数据为空，使用 mock 数据兜底');
          setMarketData(generateMarketData(60));
        }
      } catch (err) {
        console.error('获取指数 K 线失败：', err);
        setMarketData(generateMarketData(60));
      }
    };

    fetchIndexKline();
  }, []);

  // 行业 & 市场统计：默认用 mock，接口成功后覆盖
  const [industryData, setIndustryData] = useState<IndustryData[]>(defaultIndustryData);
  const [marketStats, setMarketStats] = useState<MarketStatsData>(defaultMarketStats);

  // 从后端 FastAPI 拉取市场总览 & 行业情绪
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [overviewRes, industryRes] = await Promise.all([
          fetch(`${API_BASE}/api/overview`),
          fetch(`${API_BASE}/api/industry-sentiment?limit=5`),
        ]);

        if (overviewRes.ok) {
          const overviewJson = await overviewRes.json();
          const mapped = mapOverviewToMarketStats(overviewJson, defaultMarketStats);
          setMarketStats(mapped);
        } else {
          console.warn('overview 接口返回非 200：', overviewRes.status);
        }

        if (industryRes.ok) {
          const industryJson = await industryRes.json();
          const mappedIndustries = mapIndustryList(industryJson, defaultIndustryData);
          setIndustryData(mappedIndustries);
        } else {
          console.warn('industry-sentiment 接口返回非 200：', industryRes.status);
        }
      } catch (err) {
        console.error('获取市场统计/行业情绪失败：', err);
        // 失败时保持默认 mock 数据，不影响页面展示
      }
    };

    fetchStats();
  }, []);

  const toggleStrategy = (id: string) => {
    setStrategies((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, status: s.status === 'active' ? 'paused' : 'active' } : s,
      ),
    );
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 flex font-sans">
      {/* Sidebar */}
      <aside className="w-20 lg:w-64 bg-slate-950 border-r border-slate-800 flex flex-col fixed h-full z-10 transition-all duration-300">
        <div className="h-16 flex items-center justify-center lg:justify-start lg:px-6 border-b border-slate-800">
          <div className="w-8 h-8 bg-red-600 rounded-lg flex items-center justify-center font-bold text-white shadow-lg shadow-red-500/30">
            Q
          </div>
          <span className="ml-3 font-bold text-xl text-white hidden lg:block tracking-tight">
            QuantMind A股
          </span>
        </div>

        <nav className="flex-1 py-6 space-y-2 px-2 lg:px-4">
          {[
            { id: 'dashboard', icon: LayoutDashboard, label: '大盘总览 Dashboard' },
            { id: 'backtest', icon: LineChart, label: '策略回测 Backtest' },
            { id: 'forecast', icon: TrendingUp, label: '走势预测 Forecast' },
            { id: 'news', icon: Newspaper, label: '财经新闻 News' },
            { id: 'settings', icon: Settings, label: '系统设置 System' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center p-3 rounded-lg transition-all duration-200 group ${
                activeTab === item.id
                  ? 'bg-red-600 text-white shadow-lg shadow-red-900/50'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <item.icon size={22} className={activeTab === item.id ? 'animate-pulse' : ''} />
              <span className="ml-3 font-medium hidden lg:block">{item.label}</span>
              {activeTab === item.id && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-white hidden lg:block" />
              )}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 ml-20 lg:ml-64 p-4 lg:p-8 overflow-x-hidden">
        {/* Header */}
        <header className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white">
              {activeTab === 'dashboard' && 'A股市场全景 (A-Share Overview)'}
              {activeTab === 'backtest' && '策略回测 (Strategy Backtesting)'}
              {activeTab === 'forecast' && '走势预测 (AI Market Forecast)'}
              {activeTab === 'news' && '财经新闻快讯 (Financial News)'}
              {activeTab === 'settings' && '系统设置 (System)'}
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              {activeTab === 'dashboard' ? (
                <>
                  上证指数:{' '}
                  <span className="text-red-400 font-bold">3,052.14 (+1.2%)</span> | 成交量:{' '}
                  <span className="text-slate-200">8,500亿</span>
                </>
              ) : (
                '专业量化回测引擎 | 支持日线级别多策略并发测试'
              )}
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative hidden md:block">
              <Search
                className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                size={16}
              />
              <input
                type="text"
                placeholder="全局搜索..."
                className="bg-slate-800 border border-slate-700 rounded-full pl-10 pr-4 py-2 text-sm focus:outline-none focus:border-red-500 w-64 text-white placeholder-slate-500 transition-all"
              />
            </div>
            <button className="relative p-2 text-slate-400 hover:text-white bg-slate-800 rounded-full border border-slate-700 hover:bg-slate-700 transition-colors">
              <Bell size={20} />
              <span className="absolute top-0 right-0 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-slate-900"></span>
            </button>
          </div>
        </header>

        {/* Content Switching Logic */}
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <div className="xl:col-span-2 space-y-6">
              <div className="h-[450px]">
                <MarketChart data={marketData} symbol="上证指数 (000001.SH)" />
              </div>
              <MarketStats industryData={industryData} marketStats={marketStats} />
            </div>
            <div className="xl:col-span-1 space-y-6">
              <StrategyPanel strategies={strategies} onToggleStrategy={toggleStrategy} />
              <AIAnalyst positions={positions} strategies={strategies} />
            </div>
          </div>
        )}

        {activeTab === 'backtest' && <StrategyBacktestPage />}

        {activeTab === 'forecast' && <StockForecastPage />}

        {activeTab === 'news' && <NewsPage />}

        {activeTab === 'settings' && <StrategySettingsPage />}
      </main>
    </div>
  );
};

export default App;
