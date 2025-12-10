// webui/src/pages/StrategyBacktestPage.tsx
import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
  Search,
  Play,
  Settings,
  Info,
  ChevronDown,
  ChevronRight,
  Layers,
  Activity,
  TrendingUp,
  BarChart2,
  Wallet,
  Percent,
  Hash,
  ArrowUpRight,
  ArrowDownRight,
  Loader2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceDot,
  Legend,
} from 'recharts';
import {
  fetchStrategies,
  runBacktest,
  type BacktestRequest,
  type BacktestResponse,
  type CombinationMode,
  type StrategyMeta,
  type PricePoint,
  type TradeRecord as ApiTradeRecord,
} from '../api/backtest';

// --- Types & Interfaces ---

interface BacktestParams {
  symbol: string;
  initialCapital: number;
  startDate: string;
  endDate: string;
  feeRate: number;      // 万分比
  slippage: number;     // 每股滑点（元）
  tradeSizeMode: 'FixedShares' | 'FixedPercent';
  tradeSizeValue: number;
}

interface TradeRecord {
  id: string;
  date: string;
  side: 'BUY' | 'SELL';
  price: number;
  qty: number;
  profitPct: number;
  cumProfitPct: number;
  positionDirection: 'LONG' | 'SHORT' | 'FLAT';
}

interface ChartDataPoint {
  date: string;
  price: number;
  equity: number;
  rsi: number;
}

interface BacktestSignal {
  date: string;
  type: 'BUY' | 'SELL';
  price: number;
}

interface BacktestResult {
  returnPct: number;
  annualReturnPct: number;
  maxDrawdownPct: number;
  sharpe: number;
  winRatePct: number;
  calmar: number;
  priceSeries: ChartDataPoint[];
  equityCurve: ChartDataPoint[];
  indicatorSeries: ChartDataPoint[];
  tradeSignals: BacktestSignal[];
  trades: TradeRecord[];
}

interface StockOption {
  code: string;
  name: string;
}

const formatDate = (date: Date) => {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, '0');
  const day = `${date.getDate()}`.padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const getDefaultDateRange = () => {
  const end = new Date();
  const start = new Date(end);
  start.setFullYear(end.getFullYear() - 1);
  return {
    start: formatDate(start),
    end: formatDate(end),
  };
};

// --- Mock Constants ---

const STOCK_OPTIONS: StockOption[] = [
  { code: '688192', name: '迪哲医药' },
  { code: '600519', name: '贵州茅台' },
  { code: '300750', name: '宁德时代' },
  { code: '600036', name: '招商银行' },
  { code: '000001', name: '平安银行' },
  { code: '601318', name: '中国平安' },
];

const FALLBACK_STRATEGIES: StrategyMeta[] = [
  {
    id: 'dual_ma',
    name: '双均线趋势 (MA5/20)',
    category: 'trend',
    description: '短期均线向上突破长期均线买入，反向卖出',
  },
  {
    id: 'ma_bull',
    name: '均线多头排列',
    category: 'trend',
    description: 'MA5 > MA10 > MA20 时买入',
  },
  {
    id: 'rsi_rev',
    name: 'RSI 超卖反转',
    category: 'reversal',
    description: 'RSI < 30 买入，RSI > 70 卖出',
  },
  {
    id: 'boll_rev',
    name: '布林带反向突破',
    category: 'reversal',
    description: '价格跌破下轨反弹买入',
  },
  {
    id: 'boll_break',
    name: '布林带突破',
    category: 'volatility',
    description: '价格突破上轨买入',
  },
  {
    id: 'vol_break',
    name: '成交量放大突破',
    category: 'volume',
    description: '量比 > 2 且价格上涨',
  },
];

// --- Mock Simulation Logic ---

const runMockBacktest = (
  params: BacktestParams,
  strategies: string[],
  mode: CombinationMode
): Promise<BacktestResult> => {
  return new Promise(resolve => {
    setTimeout(() => {
      const start = new Date(params.startDate);
      const end = new Date(params.endDate);
      const totalDays = Math.max(
        60,
        Math.min(
          400,
          Math.ceil((end.getTime() - start.getTime()) / (1000 * 3600 * 24))
        )
      );

      let price = 100 + Math.random() * 20;
      let capital = params.initialCapital;
      let position = 0;
      let entryPrice = 0;
      let peakEquity = capital;
      let maxDrawdown = 0;

      const trades: TradeRecord[] = [];
      const dataPoints: ChartDataPoint[] = [];
      const signals: BacktestSignal[] = [];

      const tradeModeFactor = mode === 'VOTING' ? 1.2 : mode === 'OR' ? 1 : 0.8;
      const strategyFactor = Math.max(1, strategies.length);

      let cumProfitPct = 0;
      let winCount = 0;
      let tradeCount = 0;

      for (let i = 0; i < totalDays; i++) {
        const date = new Date(start);
        date.setDate(start.getDate() + i);
        const dateStr = date.toISOString().split('T')[0];

        // 简单随机游走 + 略向上偏移
        const change = (Math.random() - 0.48) * 3;
        price += change;
        if (price < 5) price = 5;

        // 伪 RSI：在 10~90 之间震荡
        const rsi =
          30 +
          Math.random() * 40 +
          Math.sin(i / 10) * 20 +
          (strategyFactor - 1) * 2;

        const equity = capital + position * price;
        peakEquity = Math.max(peakEquity, equity);
        const dd = peakEquity > 0 ? (peakEquity - equity) / peakEquity : 0;
        maxDrawdown = Math.max(maxDrawdown, dd);

        // 模拟信号——真正逻辑以后由后端来算
        let action: 'BUY' | 'SELL' | 'NONE' = 'NONE';
        const baseProb = 0.04 * tradeModeFactor * (1 + strategyFactor / 4);

        if (position === 0 && Math.random() < baseProb && rsi < 35) {
          action = 'BUY';
        } else if (position > 0 && Math.random() < baseProb && rsi > 65) {
          action = 'SELL';
        }

        if (action === 'BUY') {
          let qty = 0;
          if (params.tradeSizeMode === 'FixedShares') {
            qty = params.tradeSizeValue;
          } else {
            qty = Math.floor(
              (capital * (params.tradeSizeValue / 100)) / price
            );
          }

          if (qty > 0 && capital >= qty * price) {
            position = qty;
            entryPrice = price;
            capital -= qty * price * (1 + params.feeRate / 10000);

            trades.push({
              id: `T${trades.length + 1}`,
              date: dateStr,
              side: 'BUY',
              price,
              qty,
              profitPct: 0,
              cumProfitPct,
              positionDirection: 'LONG',
            });

            signals.push({ date: dateStr, type: 'BUY', price });
          }
        } else if (action === 'SELL' && position > 0) {
          const exitPrice = price;
          const qty = position;

          const gross = exitPrice * qty;
          const net =
            gross * (1 - params.feeRate / 10000) -
            entryPrice * qty * (1 + params.feeRate / 10000);
          const tradePct = (net / (entryPrice * qty)) * 100;

          capital += gross * (1 - params.feeRate / 10000);
          position = 0;
          cumProfitPct += tradePct;
          tradeCount += 1;
          if (tradePct > 0) winCount += 1;

          trades.push({
            id: `T${trades.length + 1}`,
            date: dateStr,
            side: 'SELL',
            price: exitPrice,
            qty,
            profitPct: tradePct,
            cumProfitPct,
            positionDirection: 'FLAT',
          });

          signals.push({ date: dateStr, type: 'SELL', price: exitPrice });
        }

        dataPoints.push({
          date: dateStr,
          price,
          equity,
          rsi,
        });
      }

      const finalEquity = capital + position * price;
      const totalReturn = (finalEquity / params.initialCapital) - 1;
      const years = Math.max(1 / 12, totalDays / 252);
      const annualReturn = Math.pow(1 + totalReturn, 1 / years) - 1;

      const winRate =
        tradeCount > 0 ? (winCount / tradeCount) * 100 : 0;

      // 简易夏普 & Calmar（只是演示）
      const sharpe = annualReturn / (maxDrawdown + 0.01);
      const calmar =
        maxDrawdown > 0 ? annualReturn / maxDrawdown : annualReturn;

      const result: BacktestResult = {
        returnPct: totalReturn * 100,
        annualReturnPct: annualReturn * 100,
        maxDrawdownPct: maxDrawdown * 100,
        sharpe: sharpe,
        winRatePct: winRate,
        calmar,
        priceSeries: dataPoints,
        equityCurve: dataPoints,
        indicatorSeries: dataPoints,
        tradeSignals: signals,
        trades,
      };

      resolve(result);
    }, 800);
  });
};

const mapBacktestResponseToResult = (
  resp: BacktestResponse,
): BacktestResult => {
  const equityByDate = new Map<string, number>(
    resp.equity_curve.map(pt => [pt.date, pt.equity])
  );

  const priceSource: PricePoint[] =
    resp.price_series && resp.price_series.length > 0
      ? resp.price_series
      : resp.equity_curve.map(pt => ({
          date: pt.date,
          open: pt.equity,
          high: pt.equity,
          low: pt.equity,
          close: pt.equity,
          volume: 0,
        }));

  const priceSeries: ChartDataPoint[] = priceSource.map(pt => {
    const equity = equityByDate.get(pt.date) ?? pt.close ?? pt.open ?? 0;
    return {
      date: pt.date,
      price: pt.close ?? pt.open ?? equity,
      equity,
      rsi: 50,
    };
  });

  const equityCurve: ChartDataPoint[] = resp.equity_curve.map(pt => ({
    date: pt.date,
    price: pt.equity,
    equity: pt.equity,
    rsi: 50,
  }));

  const tradeSignals: BacktestSignal[] = [];
  const trades: TradeRecord[] = [];
  let cumProfitPct = 0;
  let winCount = 0;
  let tradeCount = 0;
  let lastBuyPrice = 0;
  let lastBuyQty = 0;
  let lastBuyFee = 0;

  (resp.trades || []).forEach((t: ApiTradeRecord, idx) => {
    const side = (t.action || '').toUpperCase() === 'BUY' ? 'BUY' : 'SELL';
    tradeSignals.push({
      date: t.date || `T${idx + 1}`,
      type: side,
      price: t.price || 0,
    });

    if (side === 'BUY') {
      lastBuyPrice = t.price || 0;
      lastBuyQty = t.shares || 0;
      lastBuyFee = t.fee || 0;
      trades.push({
        id: `T${idx + 1}`,
        date: t.date || '',
        side: 'BUY',
        price: t.price || 0,
        qty: t.shares || 0,
        profitPct: 0,
        cumProfitPct,
        positionDirection: 'LONG',
      });
    } else {
      const sellFee = t.fee || 0;
      const base = lastBuyPrice * lastBuyQty;
      const profit =
        base > 0
          ? (t.price - lastBuyPrice) * lastBuyQty - (lastBuyFee + sellFee)
          : 0;
      const pct = base > 0 ? (profit / base) * 100 : 0;
      cumProfitPct += pct;
      tradeCount += 1;
      if (pct > 0) winCount += 1;
      trades.push({
        id: `T${idx + 1}`,
        date: t.date || '',
        side: 'SELL',
        price: t.price || 0,
        qty: t.shares || 0,
        profitPct: pct,
        cumProfitPct,
        positionDirection: 'FLAT',
      });
      lastBuyPrice = 0;
      lastBuyQty = 0;
      lastBuyFee = 0;
    }
  });

  const winRateFromBackend =
    (resp.stats as Record<string, number>)['win_rate'] ??
    resp.trade_stats?.win_rate;

  const winRatePct =
    typeof winRateFromBackend === 'number'
      ? winRateFromBackend * 100
      : tradeCount > 0
      ? (winCount / tradeCount) * 100
      : 0;

  const totalReturnPct = (resp.stats.total_return || 0) * 100;
  const annualReturnPct = (resp.stats.annual_return || 0) * 100;
  const maxDrawdownPct = (resp.stats.max_drawdown || 0) * 100;
  const calmar =
    resp.stats.max_drawdown && resp.stats.max_drawdown !== 0
      ? resp.stats.annual_return / Math.abs(resp.stats.max_drawdown)
      : resp.stats.annual_return;

  return {
    returnPct: totalReturnPct,
    annualReturnPct,
    maxDrawdownPct,
    sharpe: resp.stats.sharpe ?? 0,
    winRatePct,
    calmar,
    priceSeries,
    equityCurve,
    indicatorSeries: priceSeries,
    tradeSignals,
    trades,
  };
};

// --- Main Page Component ---

const StrategyBacktestPage: React.FC = () => {
  const [strategies, setStrategies] = useState<StrategyMeta[]>([]);
  const [strategiesLoading, setStrategiesLoading] = useState(false);
  const [strategiesError, setStrategiesError] = useState<string | null>(null);
  const [selectedStrategyIds, setSelectedStrategyIds] = useState<string[]>([]);
  const [combinationMode, setCombinationMode] =
    useState<CombinationMode>('OR');
  const [votingThreshold, setVotingThreshold] = useState<number>(2);

  const [params, setParams] = useState<BacktestParams>(() => {
    const { start, end } = getDefaultDateRange();
    return {
      symbol: '688192',
      initialCapital: 100000,
      startDate: start,
      endDate: end,
      feeRate: 0,
      slippage: 0,
      tradeSizeMode: 'FixedShares',
      tradeSizeValue: 100,
    };
  });

  const [isRunning, setIsRunning] = useState(false);
  const [backtestResult, setBacktestResult] =
    useState<BacktestResult | null>(null);
  const [activeTab, setActiveTab] = useState<'price' | 'equity' | 'indicator'>(
    'price'
  );
  const [searchTerm, setSearchTerm] = useState('');
  const [showSearch, setShowSearch] = useState(false);

  const groupedStrategies = useMemo(() => {
    const map: Record<string, StrategyMeta[]> = {};
    const source = strategies.length > 0 ? strategies : FALLBACK_STRATEGIES;

    source.forEach(s => {
      const cat = s.category || 'others';
      if (!map[cat]) {
        map[cat] = [];
      }
      map[cat].push(s);
    });

    Object.values(map).forEach(list =>
      list.sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'))
    );
    return map;
  }, [strategies]);

  const [expandedCats, setExpandedCats] = useState<Record<string, boolean>>({
    trend: true,
    reversal: true,
    volatility: true,
    volume: true,
    others: true,
  });

  useEffect(() => {
    const loadStrategies = async () => {
      setStrategiesLoading(true);
      try {
        const list = await fetchStrategies();
        setStrategies(list);
        setSelectedStrategyIds(prev =>
          prev.length > 0 || list.length === 0 ? prev : [list[0].id]
        );
        setStrategiesError(null);
      } catch (err) {
        console.error('[Backtest] 获取策略列表失败，使用兜底：', err);
        setStrategies(FALLBACK_STRATEGIES);
        setSelectedStrategyIds(prev =>
          prev.length > 0 || FALLBACK_STRATEGIES.length === 0
            ? prev
            : [FALLBACK_STRATEGIES[0].id]
        );
        setStrategiesError('策略列表获取失败，已使用本地预置策略');
      } finally {
        setStrategiesLoading(false);
      }
    };

    loadStrategies();
  }, []);

  const toggleCat = (cat: string) =>
    setExpandedCats(prev => ({ ...prev, [cat]: !prev[cat] }));

  const toggleStrategy = (id: string) => {
    setSelectedStrategyIds(prev =>
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    );
  };

  const handleRunBacktest = useCallback(async () => {
    if (selectedStrategyIds.length === 0) {
      alert('请至少勾选一个策略');
      return;
    }

    setIsRunning(true);
    setBacktestResult(null);
    try {
      const payload: BacktestRequest = {
        symbol: params.symbol,
        start_date: params.startDate,
        end_date: params.endDate,
        strategies: selectedStrategyIds.map(id => ({
          id,
          params:
            strategies.find(s => s.id === id)?.default_params ||
            FALLBACK_STRATEGIES.find(s => s.id === id)?.default_params ||
            {},
        })),
        mode: combinationMode,
        initial_capital: params.initialCapital,
        fee_rate_bps: params.feeRate,
        slippage: params.slippage,
      };

      const resp = await runBacktest(payload);
      const mapped = mapBacktestResponseToResult(resp);
      setBacktestResult(mapped);
      setActiveTab('price');
    } catch (e) {
      console.error('[Backtest] 后端回测失败，改用本地模拟：', e);
      alert('后端回测接口调用失败，已暂时使用本地模拟结果，请检查 8000 端口服务');
      try {
        const result = await runMockBacktest(
          params,
          selectedStrategyIds,
          combinationMode
        );
        setBacktestResult(result);
      } catch (mockErr) {
        console.error('本地模拟回测也失败：', mockErr);
        alert('本地模拟回测失败，请查看控制台错误。');
      }
    } finally {
      setIsRunning(false);
    }
  }, [combinationMode, params, selectedStrategyIds, strategies]);

  const filteredStocks = STOCK_OPTIONS.filter(
    s =>
      s.code.includes(searchTerm) ||
      s.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const currentSeries =
    activeTab === 'price'
      ? backtestResult?.priceSeries
      : activeTab === 'equity'
      ? backtestResult?.equityCurve
      : backtestResult?.indicatorSeries;

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Header */}
      <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950 shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-purple-600 rounded flex items-center justify-center font-bold text-white shadow shadow-purple-500/30">
            BT
          </div>
          <h1 className="text-lg font-bold tracking-tight text-white">
            策略回测 Strategy Backtesting
          </h1>
        </div>

        <div className="relative">
          <div className="relative w-64">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
              size={16}
            />
            <input
              type="text"
              placeholder="搜索股票代码或名称"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              onFocus={() => setShowSearch(true)}
              onBlur={() => setTimeout(() => setShowSearch(false), 150)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-purple-500"
            />
          </div>

          {showSearch && filteredStocks.length > 0 && (
            <div className="absolute right-0 mt-1 w-64 bg-slate-900 border border-slate-700 rounded-lg shadow-xl max-h-72 overflow-y-auto z-20">
              {filteredStocks.map(s => (
                <button
                  key={s.code}
                  onMouseDown={e => {
                    e.preventDefault();
                    setParams(prev => ({
                      ...prev,
                      symbol: s.code,
                    }));
                    setSearchTerm(`${s.code} ${s.name}`);
                    setShowSearch(false);
                  }}
                  className="w-full flex items-center justify-between px-3 py-2 text-sm hover:bg-slate-800 text-slate-100"
                >
                  <span className="font-mono">{s.code}</span>
                  <span className="text-xs text-slate-400">{s.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* 左侧 Sidebar：策略 / 组合方式 / 参数 */}
        <div className="w-80 flex-shrink-0 bg-slate-950 border-r border-slate-800 flex flex-col h-full overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
            {/* 1. 策略选择 */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-slate-200 font-bold">
                <Layers size={16} className="text-purple-500" />
                <h3>策略选择 (Strategies)</h3>
              </div>

              <div className="space-y-2">
                {strategiesLoading && (
                  <div className="text-xs text-slate-500 px-2">
                    正在加载策略列表...
                  </div>
                )}
                {strategiesError && !strategiesLoading && (
                  <div className="text-xs text-amber-400 px-2">
                    {strategiesError}
                  </div>
                )}
                {!strategiesLoading &&
                  Object.keys(groupedStrategies).length === 0 && (
                    <div className="text-xs text-slate-500 px-2">
                      暂无可用策略，请检查后端服务。
                    </div>
                  )}
                {Object.entries(groupedStrategies).map(([cat, items]) => (
                  <div
                    key={cat}
                    className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900/50"
                  >
                    <button
                      onClick={() => toggleCat(cat)}
                      className="w-full flex items-center justify-between p-2.5 bg-slate-900 text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
                    >
                      <span className="uppercase">{cat} Strategies</span>
                      {expandedCats[cat] ? (
                        <ChevronDown size={14} />
                      ) : (
                        <ChevronRight size={14} />
                      )}
                    </button>

                    {expandedCats[cat] && (
                      <div className="p-2 space-y-1">
                        {items.map(s => (
                          <div
                            key={s.id}
                            className="group flex items-center justify-between p-2 rounded hover:bg-slate-800 transition-colors"
                          >
                            <label className="flex items-center gap-2 cursor-pointer flex-1">
                              <input
                                type="checkbox"
                                checked={selectedStrategyIds.includes(s.id)}
                                onChange={() => toggleStrategy(s.id)}
                                className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-purple-600 focus:ring-offset-slate-900 focus:ring-purple-500"
                              />
                              <span
                                className={`text-sm ${
                                  selectedStrategyIds.includes(s.id)
                                    ? 'text-slate-200'
                                    : 'text-slate-400'
                                }`}
                              >
                                {s.name}
                              </span>
                            </label>
                            <div className="relative group/tooltip">
                              <Info
                                size={14}
                                className="text-slate-600 hover:text-slate-400 cursor-help"
                              />
                              <div className="absolute right-0 top-6 w-48 p-2 bg-slate-800 border border-slate-700 rounded shadow-xl text-xs text-slate-300 z-50 hidden group-hover/tooltip:block">
                                {s.description}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 2. 组合方式 */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-slate-200 font-bold">
                <Activity size={16} className="text-blue-500" />
                <h3>组合方式 (Mode)</h3>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2">
                {[
                  { id: 'AND', label: '所有策略同时满足 (AND)' },
                  { id: 'OR', label: '任意一个策略满足 (OR)' },
                  { id: 'VOTING', label: '投票模式 (Voting)' },
                ].map(opt => (
                  <label
                    key={opt.id}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <input
                      type="radio"
                      name="combMode"
                      value={opt.id}
                      checked={combinationMode === opt.id}
                      onChange={() =>
                        setCombinationMode(opt.id as CombinationMode)
                      }
                      className="w-3.5 h-3.5 border-slate-600 bg-slate-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-slate-900"
                    />
                    <span className="text-sm text-slate-300">
                      {opt.label}
                    </span>
                  </label>
                ))}

                {combinationMode === 'VOTING' && (
                  <div className="flex items-center gap-2 mt-2 pl-6">
                    <span className="text-xs text-slate-500">
                      至少满足:
                    </span>
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={votingThreshold}
                      onChange={e =>
                        setVotingThreshold(Number(e.target.value) || 1)
                      }
                      className="w-16 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white text-center focus:border-blue-500 outline-none"
                    />
                    <span className="text-xs text-slate-500">个策略</span>
                  </div>
                )}
              </div>
            </div>

            {/* 3. 回测参数 */}
            <div className="space-y-3 pb-20">
              <div className="flex items-center gap-2 text-slate-200 font-bold">
                <Settings size={16} className="text-emerald-500" />
                <h3>回测参数 (Parameters)</h3>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-4">
                {/* 股票代码 */}
                <div className="space-y-1">
                  <label className="text-xs text-slate-500 font-medium">
                    股票代码 (Symbol)
                  </label>
                  <input
                    type="text"
                    value={params.symbol}
                    onChange={e =>
                      setParams(prev => ({ ...prev, symbol: e.target.value }))
                    }
                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-white focus:border-emerald-500 outline-none font-mono"
                  />
                </div>

                {/* 初始资金 */}
                <div className="space-y-1">
                  <label className="text-xs text-slate-500 font-medium">
                    初始资金 (Initial Capital)
                  </label>
                  <div className="relative">
                    <Wallet
                      size={12}
                      className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
                    />
                    <input
                      type="number"
                      value={params.initialCapital}
                      onChange={e =>
                        setParams(prev => ({
                          ...prev,
                          initialCapital: Number(e.target.value) || 0,
                        }))
                      }
                      className="w-full bg-slate-800 border border-slate-700 rounded pl-8 pr-3 py-1.5 text-sm text-white focus:border-emerald-500 outline-none font-mono"
                    />
                  </div>
                </div>

                {/* 日期区间 */}
                <div className="space-y-1">
                  <label className="text-xs text-slate-500 font-medium">
                    回测区间 (Date Range)
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="date"
                      value={params.startDate}
                      onChange={e =>
                        setParams(prev => ({
                          ...prev,
                          startDate: e.target.value,
                        }))
                      }
                      className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none font-mono"
                    />
                    <input
                      type="date"
                      value={params.endDate}
                      onChange={e =>
                        setParams(prev => ({
                          ...prev,
                          endDate: e.target.value,
                        }))
                      }
                      className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none font-mono"
                    />
                  </div>
                </div>

                {/* 手续费 / 滑点 */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-xs text-slate-500 font-medium">
                      手续费 (bps)
                    </label>
                    <div className="relative">
                      <Percent
                        size={12}
                        className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
                      />
                      <input
                        type="number"
                        step="0.1"
                        value={params.feeRate}
                        onChange={e =>
                          setParams(prev => ({
                            ...prev,
                            feeRate: Number(e.target.value) || 0,
                          }))
                        }
                        className="w-full bg-slate-800 border border-slate-700 rounded pl-8 pr-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none text-right"
                      />
                    </div>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-slate-500 font-medium">
                      滑点 (元/股)
                    </label>
                    <div className="relative">
                      <Hash
                        size={12}
                        className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
                      />
                      <input
                        type="number"
                        step="0.01"
                        value={params.slippage}
                        onChange={e =>
                          setParams(prev => ({
                            ...prev,
                            slippage: Number(e.target.value) || 0,
                          }))
                        }
                        className="w-full bg-slate-800 border border-slate-700 rounded pl-8 pr-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none text-right"
                      />
                    </div>
                  </div>
                </div>

                {/* 交易规模设置 */}
                <div className="space-y-2 pt-2 border-t border-slate-800">
                  <label className="text-xs text-slate-500 font-medium block">
                    每笔交易方式
                  </label>
                  <div className="flex gap-4">
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="radio"
                        checked={params.tradeSizeMode === 'FixedShares'}
                        onChange={() =>
                          setParams(prev => ({
                            ...prev,
                            tradeSizeMode: 'FixedShares',
                          }))
                        }
                        className="bg-slate-800 border-slate-600 text-emerald-500 focus:ring-0"
                      />
                      <span className="text-xs text-slate-300">固定股数</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="radio"
                        checked={params.tradeSizeMode === 'FixedPercent'}
                        onChange={() =>
                          setParams(prev => ({
                            ...prev,
                            tradeSizeMode: 'FixedPercent',
                          }))
                        }
                        className="bg-slate-800 border-slate-600 text-emerald-500 focus:ring-0"
                      />
                      <span className="text-xs text-slate-300">
                        资金比例
                      </span>
                    </label>
                  </div>
                  <div className="relative">
                    <Hash
                      size={12}
                      className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
                    />
                    <input
                      type="number"
                      value={params.tradeSizeValue}
                      onChange={e =>
                        setParams(prev => ({
                          ...prev,
                          tradeSizeValue: Number(e.target.value) || 0,
                        }))
                      }
                      className="w-full bg-slate-800 border border-slate-700 rounded pl-8 pr-3 py-1.5 text-sm text-white focus:border-emerald-500 outline-none font-mono"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 底部 Run 按钮 */}
          <div className="p-4 border-t border-slate-800 bg-slate-950">
            <button
              onClick={handleRunBacktest}
              disabled={isRunning || selectedStrategyIds.length === 0}
              className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white py-3 rounded-xl font-bold flex justify-center items-center gap-2 transition-all shadow-lg shadow-purple-500/30"
            >
              {isRunning ? (
                <>
                  <Loader2 className="animate-spin" size={18} />
                  正在运行回测...
                </>
              ) : (
                <>
                  <Play size={18} />
                  运行组合回测
                </>
              )}
            </button>
          </div>
        </div>

        {/* 右侧：图表 + 结果 */}
        <div className="flex-1 flex flex-col h-full">
          {/* 顶部统计卡片 */}
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 p-4 border-b border-slate-800 bg-slate-950/80">
            {backtestResult ? (
              <>
                <StatCard
                  label="总收益率"
                  value={backtestResult.returnPct}
                  suffix="%"
                  positiveColor="text-red-400"
                  negativeColor="text-emerald-400"
                />
                <StatCard
                  label="年化收益"
                  value={backtestResult.annualReturnPct}
                  suffix="%"
                  positiveColor="text-red-400"
                  negativeColor="text-emerald-400"
                />
                <StatCard
                  label="最大回撤"
                  value={backtestResult.maxDrawdownPct}
                  suffix="%"
                  invert
                />
                <StatCard
                  label="夏普比率"
                  value={backtestResult.sharpe}
                  fractionDigits={2}
                />
                <StatCard
                  label="胜率"
                  value={backtestResult.winRatePct}
                  suffix="%"
                />
                <StatCard
                  label="Calmar"
                  value={backtestResult.calmar}
                  fractionDigits={2}
                />
              </>
            ) : (
              <div className="col-span-2 md:col-span-3 xl:col-span-6 flex items-center gap-2 text-slate-500 text-sm">
                <BarChart2 size={18} className="text-slate-600" />
                请在左侧选择策略并运行回测，结果会显示在这里。
              </div>
            )}
          </div>

          {/* 中部图表 */}
          <div className="flex-1 p-4 pb-2">
            <div className="bg-slate-900 rounded-xl border border-slate-800 h-full flex flex-col overflow-hidden">
              {/* Tab 切换 */}
              <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 bg-slate-900/80">
                <div className="flex items-center gap-4 text-xs">
                  <button
                    onClick={() => setActiveTab('price')}
                    className={`flex items-center gap-1 px-3 py-1.5 rounded-full border text-xs ${
                      activeTab === 'price'
                        ? 'bg-slate-800 border-slate-600 text-slate-100'
                        : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    <BarChart2 size={14} />
                    价格 & 信号
                  </button>
                  <button
                    onClick={() => setActiveTab('equity')}
                    className={`flex items-center gap-1 px-3 py-1.5 rounded-full border text-xs ${
                      activeTab === 'equity'
                        ? 'bg-slate-800 border-slate-600 text-slate-100'
                        : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    <TrendingUp size={14} />
                    权益曲线
                  </button>
                  <button
                    onClick={() => setActiveTab('indicator')}
                    className={`flex items-center gap-1 px-3 py-1.5 rounded-full border text-xs ${
                      activeTab === 'indicator'
                        ? 'bg-slate-800 border-slate-600 text-slate-100'
                        : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    <Activity size={14} />
                    指标 (RSI)
                  </button>
                </div>
              </div>

              <div className="flex-1 p-4">
                {backtestResult && currentSeries ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={currentSeries}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: '#9ca3af' }}
                      />
                      <YAxis
                        yAxisId="left"
                        tick={{ fontSize: 10, fill: '#9ca3af' }}
                        domain={['auto', 'auto']}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fontSize: 10, fill: '#9ca3af' }}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#020617',
                          border: '1px solid #1f2937',
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                        labelStyle={{ color: '#e5e7eb' }}
                      />
                      <Legend />
                      {activeTab === 'price' && (
                        <>
                          <Area
                            yAxisId="left"
                            type="monotone"
                            dataKey="price"
                            name="Price"
                            fill="rgba(96,165,250,0.2)"
                            stroke="#60a5fa"
                            strokeWidth={1.5}
                          />
                          {backtestResult.tradeSignals.map((s, idx) => (
                            <ReferenceDot
                              key={`${s.date}-${idx}`}
                              x={s.date}
                              y={s.price}
                              r={4}
                              yAxisId="left"
                              fill={
                                s.type === 'BUY' ? '#22c55e' : '#ef4444'
                              }
                              stroke="#020617"
                            />
                          ))}
                        </>
                      )}
                      {activeTab === 'equity' && (
                        <Line
                          yAxisId="left"
                          type="monotone"
                          dataKey="equity"
                          name="Equity"
                          stroke="#a855f7"
                          strokeWidth={1.5}
                          dot={false}
                        />
                      )}
                      {activeTab === 'indicator' && (
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="rsi"
                          name="RSI"
                          stroke="#22c55e"
                          strokeWidth={1.5}
                          dot={false}
                        />
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                    暂无数据，请先在左侧配置参数并运行回测。
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 底部交易明细 */}
          <div className="h-64 p-4 pt-0">
            <div className="bg-slate-900 rounded-xl border border-slate-800 h-full flex flex-col overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-950/80">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                  <TrendingUp size={16} className="text-slate-400" />
                  交易明细 (Trades)
                </div>
                {backtestResult && (
                  <span className="text-xs text-slate-500">
                    共 {backtestResult.trades.length} 笔
                  </span>
                )}
              </div>
              <div className="flex-1 overflow-auto custom-scrollbar">
                {backtestResult && backtestResult.trades.length > 0 ? (
                  <table className="w-full text-xs">
                    <thead className="bg-slate-950/80 sticky top-0 z-10">
                      <tr className="text-slate-400 border-b border-slate-800">
                        <th className="px-3 py-2 text-left">日期</th>
                        <th className="px-3 py-2 text-left">方向</th>
                        <th className="px-3 py-2 text-right">价格</th>
                        <th className="px-3 py-2 text-right">数量</th>
                        <th className="px-3 py-2 text-right">单笔收益%</th>
                        <th className="px-3 py-2 text-right">累计收益%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {backtestResult.trades.map(t => (
                        <tr
                          key={t.id}
                          className="border-b border-slate-800/60 hover:bg-slate-800/50"
                        >
                          <td className="px-3 py-1.5 font-mono text-slate-300">
                            {t.date}
                          </td>
                          <td className="px-3 py-1.5">
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded-full border text-[10px] font-semibold ${
                                t.side === 'BUY'
                                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/40'
                                  : 'bg-red-500/10 text-red-400 border-red-500/40'
                              }`}
                            >
                              {t.side === 'BUY' ? '买入' : '卖出'}
                            </span>
                          </td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-200">
                            {t.price.toFixed(2)}
                          </td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                            {t.qty}
                          </td>
                          <td
                            className={`px-3 py-1.5 text-right font-mono ${
                              t.profitPct > 0
                                ? 'text-red-400'
                                : t.profitPct < 0
                                ? 'text-emerald-400'
                                : 'text-slate-400'
                            }`}
                          >
                            {t.profitPct > 0 ? '+' : ''}
                            {t.profitPct.toFixed(2)}%
                          </td>
                          <td
                            className={`px-3 py-1.5 text-right font-mono ${
                              t.cumProfitPct > 0
                                ? 'text-red-400'
                                : t.cumProfitPct < 0
                                ? 'text-emerald-400'
                                : 'text-slate-400'
                            }`}
                          >
                            {t.cumProfitPct > 0 ? '+' : ''}
                            {t.cumProfitPct.toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                    暂无交易记录。
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// 小统计卡片组件
interface StatCardProps {
  label: string;
  value: number;
  suffix?: string;
  fractionDigits?: number;
  positiveColor?: string;
  negativeColor?: string;
  invert?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  suffix = '',
  fractionDigits = 2,
  positiveColor = 'text-emerald-400',
  negativeColor = 'text-red-400',
  invert = false,
}) => {
  const v = value ?? 0;
  const isPositive = invert ? v < 0 : v > 0;
  const color =
    v === 0 ? 'text-slate-300' : isPositive ? positiveColor : negativeColor;
  const Icon = isPositive ? ArrowUpRight : ArrowDownRight;

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 px-3 py-3 flex flex-col justify-between">
      <span className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">
        {label}
      </span>
      <div className="flex items-baseline justify-between gap-2">
        <div className={`text-xl font-mono font-bold ${color}`}>
          {v > 0 ? '+' : ''}
          {v.toFixed(fractionDigits)}
          {suffix}
        </div>
        {v !== 0 && <Icon size={16} className={color} />}
      </div>
    </div>
  );
};

export default StrategyBacktestPage;
