import React, { useState, useMemo } from 'react';
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
  Calendar, 
  Wallet, 
  Percent, 
  Hash,
  ArrowUpRight,
  ArrowDownRight,
  Loader2
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceDot,
  Legend,
  ReferenceLine
} from 'recharts';

// --- Types & Interfaces ---

type CombinationMode = 'AND' | 'OR' | 'VOTING';

interface StrategyItem {
  id: string;
  name: string;
  category: 'trend' | 'reversal' | 'volatility' | 'volume';
  description: string;
}

interface BacktestParams {
  symbol: string;
  initialCapital: number;
  startDate: string;
  endDate: string;
  feeRate: number;       // 万分比
  slippagePct: number;   // 百分比
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

// --- Mock Constants ---

const STOCK_OPTIONS: StockOption[] = [
  { code: '600519', name: '贵州茅台' },
  { code: '300750', name: '宁德时代' },
  { code: '600036', name: '招商银行' },
  { code: '000001', name: '平安银行' },
  { code: '601318', name: '中国平安' },
];

const ALL_STRATEGIES: StrategyItem[] = [
  { id: 'dual_ma', name: '双均线趋势 (MA5/20)', category: 'trend', description: '短期均线向上突破长期均线买入，反向卖出' },
  { id: 'ma_bull', name: '均线多头排列', category: 'trend', description: 'MA5 > MA10 > MA20 时买入' },
  { id: 'rsi_rev', name: 'RSI 超卖反转', category: 'reversal', description: 'RSI < 30 买入，RSI > 70 卖出' },
  { id: 'boll_rev', name: '布林带反向突破', category: 'reversal', description: '价格跌破下轨反弹买入' },
  { id: 'boll_break', name: '布林带突破', category: 'volatility', description: '价格突破上轨买入' },
  { id: 'vol_break', name: '成交量放大突破', category: 'volume', description: '量比 > 2 且价格上涨' },
];

// --- Mock Simulation Logic ---

const runMockBacktest = (
  params: BacktestParams, 
  strategies: string[], 
  mode: CombinationMode
): Promise<BacktestResult> => {
  return new Promise((resolve) => {
    // Simulate calculation delay
    setTimeout(() => {
      const days = 200;
      let price = 100 + Math.random() * 20;
      let capital = params.initialCapital;
      let position = 0;
      let entryPrice = 0;
      let peakEquity = capital;
      let maxDrawdown = 0;
      const trades: TradeRecord[] = [];
      const dataPoints: ChartDataPoint[] = [];
      const signals: BacktestSignal[] = [];

      const start = new Date(params.startDate);

      for (let i = 0; i < days; i++) {
        // Random walk
        const date = new Date(start);
        date.setDate(start.getDate() + i);
        const dateStr = date.toISOString().split('T')[0];
        
        const change = (Math.random() - 0.48) * 3; // Slight upward bias
        price += change;
        if (price < 10) price = 10;

        // Mock RSI
        const rsi = 30 + Math.random() * 40 + (Math.sin(i / 10) * 20);

        // Signal Generation (Randomized based on strategy count)
        // In real logic, we would evaluate strategies here
        let action: 'BUY' | 'SELL' | 'NONE' = 'NONE';
        const signalChance = strategies.length > 0 ? 0.05 : 0; // 5% chance per day if strategies selected
        
        if (position === 0 && Math.random() < signalChance) action = 'BUY';
        else if (position > 0 && Math.random() < signalChance) action = 'SELL';

        // Execute Trade
        if (action === 'BUY') {
          // Calculate Size
          let buyQty = 0;
          if (params.tradeSizeMode === 'FixedShares') {
            buyQty = params.tradeSizeValue;
          } else {
            buyQty = Math.floor((capital * (params.tradeSizeValue / 100)) / price);
          }
          
          if (buyQty > 0 && capital >= buyQty * price) {
            position = buyQty;
            const cost = buyQty * price;
            const fee = cost * (params.feeRate / 10000);
            capital -= (cost + fee);
            entryPrice = price;
            
            trades.push({
              id: `tr-${i}`, date: dateStr, side: 'BUY', price: Number(price.toFixed(2)), qty: buyQty,
              profitPct: 0, cumProfitPct: ((capital + position * price - params.initialCapital) / params.initialCapital) * 100,
              positionDirection: 'LONG'
            });
            signals.push({ date: dateStr, type: 'BUY', price: Number(price.toFixed(2)) });
          }
        } else if (action === 'SELL' && position > 0) {
          const revenue = position * price;
          const fee = revenue * (params.feeRate / 10000);
          const pnl = (price - entryPrice) * position - fee; // simplified
          const pnlPct = ((price - entryPrice) / entryPrice) * 100;
          
          capital += (revenue - fee);
          
          trades.push({
            id: `tr-${i}`, date: dateStr, side: 'SELL', price: Number(price.toFixed(2)), qty: position,
            profitPct: Number(pnlPct.toFixed(2)),
            cumProfitPct: ((capital - params.initialCapital) / params.initialCapital) * 100,
            positionDirection: 'FLAT'
          });
          signals.push({ date: dateStr, type: 'SELL', price: Number(price.toFixed(2)) });
          position = 0;
        }

        // Update Equity & Drawdown
        const currentEquity = capital + (position * price);
        if (currentEquity > peakEquity) peakEquity = currentEquity;
        const dd = (peakEquity - currentEquity) / peakEquity * 100;
        if (dd > maxDrawdown) maxDrawdown = dd;

        dataPoints.push({
          date: dateStr,
          price: Number(price.toFixed(2)),
          equity: Number(currentEquity.toFixed(2)),
          rsi: Number(rsi.toFixed(2))
        });
      }

      // Final Stats Calculation
      const finalEquity = dataPoints[dataPoints.length - 1].equity;
      const returnPct = ((finalEquity - params.initialCapital) / params.initialCapital) * 100;
      const winningTrades = trades.filter(t => t.side === 'SELL' && t.profitPct > 0).length;
      const sellTrades = trades.filter(t => t.side === 'SELL').length;
      const winRate = sellTrades > 0 ? (winningTrades / sellTrades) * 100 : 0;

      resolve({
        returnPct: Number(returnPct.toFixed(2)),
        annualReturnPct: Number((returnPct * (365/days)).toFixed(2)),
        maxDrawdownPct: Number(maxDrawdown.toFixed(2)),
        sharpe: Number((Math.random() * 2 + 0.5).toFixed(2)),
        winRatePct: Number(winRate.toFixed(1)),
        calmar: Number((returnPct / (maxDrawdown || 1)).toFixed(2)),
        priceSeries: dataPoints,
        equityCurve: dataPoints,
        indicatorSeries: dataPoints,
        tradeSignals: signals,
        trades: trades.reverse()
      });
    }, 800);
  });
};

// --- Sub Components ---

const MetricCard: React.FC<{ label: string; value: string | number; subValue?: string; highlight?: boolean }> = ({ label, value, subValue, highlight }) => (
  <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-between hover:border-slate-700 transition-all">
    <span className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1">{label}</span>
    <div className="flex items-end gap-2">
      <span className={`text-2xl font-mono font-bold ${highlight ? 'text-emerald-400' : 'text-slate-100'}`}>{value}</span>
      {subValue && <span className="text-xs text-slate-500 mb-1">{subValue}</span>}
    </div>
  </div>
);

const StrategySidebar: React.FC<{
  strategies: StrategyItem[];
  selectedIds: string[];
  onToggle: (id: string) => void;
  mode: CombinationMode;
  setMode: (m: CombinationMode) => void;
  threshold: number;
  setThreshold: (n: number) => void;
  params: BacktestParams;
  setParams: (p: BacktestParams) => void;
  onRun: () => void;
  isRunning: boolean;
}> = ({ strategies, selectedIds, onToggle, mode, setMode, threshold, setThreshold, params, setParams, onRun, isRunning }) => {
  
  // Group strategies by category
  const groupedStrategies = useMemo(() => {
    const groups: Record<string, StrategyItem[]> = {};
    strategies.forEach(s => {
      if (!groups[s.category]) groups[s.category] = [];
      groups[s.category].push(s);
    });
    return groups;
  }, [strategies]);

  const [expandedCats, setExpandedCats] = useState<Record<string, boolean>>({
    trend: true, reversal: true, volatility: true, volume: false
  });

  const toggleCat = (cat: string) => setExpandedCats(prev => ({ ...prev, [cat]: !prev[cat] }));

  return (
    <div className="w-80 flex-shrink-0 bg-slate-950 border-r border-slate-800 flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
        
        {/* 1. Strategy Selection */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-slate-200 font-bold">
            <Layers size={16} className="text-purple-500" />
            <h3>策略选择 (Strategies)</h3>
          </div>
          
          <div className="space-y-2">
            {Object.entries(groupedStrategies).map(([cat, items]) => (
              <div key={cat} className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900/50">
                <button 
                  onClick={() => toggleCat(cat)}
                  className="w-full flex items-center justify-between p-2.5 bg-slate-900 text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <span className="uppercase">{cat} Strategies</span>
                  {expandedCats[cat] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </button>
                
                {expandedCats[cat] && (
                  <div className="p-2 space-y-1">
                    {(items as StrategyItem[]).map(s => (
                      <div key={s.id} className="group flex items-center justify-between p-2 rounded hover:bg-slate-800 transition-colors">
                        <label className="flex items-center gap-2 cursor-pointer flex-1">
                          <input 
                            type="checkbox" 
                            checked={selectedIds.includes(s.id)}
                            onChange={() => onToggle(s.id)}
                            className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-purple-600 focus:ring-offset-slate-900 focus:ring-purple-500" 
                          />
                          <span className={`text-sm ${selectedIds.includes(s.id) ? 'text-slate-200' : 'text-slate-400'}`}>{s.name}</span>
                        </label>
                        <div className="relative group/tooltip">
                          <Info size={14} className="text-slate-600 hover:text-slate-400 cursor-help" />
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

        {/* 2. Combination Mode */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-slate-200 font-bold">
            <Activity size={16} className="text-blue-500" />
            <h3>组合方式 (Mode)</h3>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2">
            {([
              { id: 'AND', label: '所有策略同时满足 (AND)' },
              { id: 'OR', label: '任意一个策略满足 (OR)' },
              { id: 'VOTING', label: '投票模式 (Voting)' }
            ] as {id: string, label: string}[]).map(opt => (
              <label key={opt.id} className="flex items-center gap-2 cursor-pointer">
                <input 
                  type="radio" 
                  name="combMode"
                  value={opt.id}
                  checked={mode === opt.id}
                  onChange={() => setMode(opt.id as CombinationMode)}
                  className="w-3.5 h-3.5 border-slate-600 bg-slate-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-slate-900" 
                />
                <span className="text-sm text-slate-300">{opt.label}</span>
              </label>
            ))}
            
            {mode === 'VOTING' && (
              <div className="flex items-center gap-2 mt-2 pl-6">
                <span className="text-xs text-slate-500">至少满足:</span>
                <input 
                  type="number" 
                  min={1} max={10} 
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-16 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white text-center focus:border-blue-500 outline-none"
                />
                <span className="text-xs text-slate-500">个策略</span>
              </div>
            )}
          </div>
        </div>

        {/* 3. Parameters */}
        <div className="space-y-3 pb-20">
          <div className="flex items-center gap-2 text-slate-200 font-bold">
            <Settings size={16} className="text-emerald-500" />
            <h3>回测参数 (Parameters)</h3>
          </div>
          
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-4">
            <div className="space-y-1">
              <label className="text-xs text-slate-500 font-medium">股票代码 (Symbol)</label>
              <input 
                type="text" 
                value={params.symbol}
                onChange={(e) => setParams({...params, symbol: e.target.value})}
                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-white focus:border-emerald-500 outline-none font-mono"
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs text-slate-500 font-medium">初始资金 (Initial Capital)</label>
              <div className="relative">
                <Wallet size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input 
                  type="number" 
                  value={params.initialCapital}
                  onChange={(e) => setParams({...params, initialCapital: Number(e.target.value)})}
                  className="w-full bg-slate-800 border border-slate-700 rounded pl-8 pr-3 py-1.5 text-sm text-white focus:border-emerald-500 outline-none font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="text-xs text-slate-500 font-medium">起始日期</label>
                <input 
                  type="date" 
                  value={params.startDate}
                  onChange={(e) => setParams({...params, startDate: e.target.value})}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-500 font-medium">结束日期</label>
                <input 
                  type="date" 
                  value={params.endDate}
                  onChange={(e) => setParams({...params, endDate: e.target.value})}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="text-xs text-slate-500 font-medium">手续费 (万分比)</label>
                <div className="relative">
                  <Percent size={10} className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-500" />
                  <input 
                    type="number" 
                    value={params.feeRate}
                    onChange={(e) => setParams({...params, feeRate: Number(e.target.value)})}
                    className="w-full bg-slate-800 border border-slate-700 rounded pl-6 pr-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none text-right"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-500 font-medium">滑点 (%)</label>
                <div className="relative">
                  <Percent size={10} className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-500" />
                  <input 
                    type="number" 
                    step="0.1"
                    value={params.slippagePct}
                    onChange={(e) => setParams({...params, slippagePct: Number(e.target.value)})}
                    className="w-full bg-slate-800 border border-slate-700 rounded pl-6 pr-2 py-1.5 text-xs text-white focus:border-emerald-500 outline-none text-right"
                  />
                </div>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-800">
               <label className="text-xs text-slate-500 font-medium block">每笔交易方式</label>
               <div className="flex gap-4">
                 <label className="flex items-center gap-1.5 cursor-pointer">
                   <input type="radio" checked={params.tradeSizeMode === 'FixedShares'} onChange={() => setParams({...params, tradeSizeMode: 'FixedShares'})} className="bg-slate-800 border-slate-600 text-emerald-500 focus:ring-0"/>
                   <span className="text-xs text-slate-300">固定股数</span>
                 </label>
                 <label className="flex items-center gap-1.5 cursor-pointer">
                   <input type="radio" checked={params.tradeSizeMode === 'FixedPercent'} onChange={() => setParams({...params, tradeSizeMode: 'FixedPercent'})} className="bg-slate-800 border-slate-600 text-emerald-500 focus:ring-0"/>
                   <span className="text-xs text-slate-300">资金比例</span>
                 </label>
               </div>
               <div className="relative">
                 <Hash size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
                 <input 
                    type="number"
                    value={params.tradeSizeValue}
                    onChange={(e) => setParams({...params, tradeSizeValue: Number(e.target.value)})}
                    className="w-full bg-slate-800 border border-slate-700 rounded pl-8 pr-3 py-1.5 text-sm text-white focus:border-emerald-500 outline-none font-mono"
                 />
               </div>
            </div>
          </div>
        </div>

      </div>

      <div className="p-4 border-t border-slate-800 bg-slate-950">
        <button 
          onClick={onRun}
          disabled={isRunning || selectedIds.length === 0}
          className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white py-3 rounded-xl font-bold flex justify-center items-center gap-2 transition-all shadow-lg shadow-purple-900/20"
        >
          {isRunning ? <Loader2 size={20} className="animate-spin" /> : <Play size={20} className="fill-white" />}
          {isRunning ? 'Running...' : '运行回测 (Run Backtest)'}
        </button>
      </div>
    </div>
  );
};

// --- Main Page Component ---

export const StrategyBacktestPage: React.FC = () => {
  // State: Configuration
  const [selectedStrategyIds, setSelectedStrategyIds] = useState<string[]>(['rsi_rev']);
  const [combinationMode, setCombinationMode] = useState<CombinationMode>('OR');
  const [votingThreshold, setVotingThreshold] = useState<number>(2);
  const [params, setParams] = useState<BacktestParams>({
    symbol: '600519',
    initialCapital: 100000,
    startDate: '2024-01-01',
    endDate: '2025-12-07',
    feeRate: 2.5,
    slippagePct: 0.1,
    tradeSizeMode: 'FixedShares',
    tradeSizeValue: 100
  });

  // State: App Logic
  const [isRunning, setIsRunning] = useState(false);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [activeTab, setActiveTab] = useState<'price' | 'equity' | 'indicator'>('price');
  const [searchTerm, setSearchTerm] = useState('');
  const [showSearch, setShowSearch] = useState(false);

  // Handlers
  const toggleStrategy = (id: string) => {
    setSelectedStrategyIds(prev => prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]);
  };

  const handleRunBacktest = async () => {
    setIsRunning(true);
    setBacktestResult(null); // Clear previous
    try {
      const result = await runMockBacktest(params, selectedStrategyIds, combinationMode);
      setBacktestResult(result);
    } catch (e) {
      console.error(e);
    } finally {
      setIsRunning(false);
    }
  };

  const filteredStocks = STOCK_OPTIONS.filter(s => s.code.includes(searchTerm) || s.name.includes(searchTerm));

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      
      {/* Header */}
      <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950 shrink-0 z-10">
        <div className="flex items-center gap-3">
           <div className="w-8 h-8 bg-purple-600 rounded flex items-center justify-center font-bold text-white shadow shadow-purple-500/30">
             BT
           </div>
           <h1 className="text-lg font-bold tracking-tight text-white">策略回测 Strategy Backtesting</h1>
        </div>

        <div className="relative">
          <div className="relative w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
            <input 
              type="text" 
              placeholder="搜索股票代码或名称..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onFocus={() => setShowSearch(true)}
              onBlur={() => setTimeout(() => setShowSearch(false), 200)}
              className="w-full bg-slate-900 border border-slate-700 rounded-full pl-10 pr-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-purple-500 transition-all placeholder-slate-600"
            />
          </div>
          {showSearch && searchTerm && (
            <div className="absolute top-full right-0 mt-2 w-64 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
              {filteredStocks.length > 0 ? (
                filteredStocks.map(stock => (
                  <div 
                    key={stock.code}
                    onClick={() => {
                      setParams(p => ({...p, symbol: stock.code}));
                      setSearchTerm(`${stock.code} ${stock.name}`);
                    }}
                    className="px-4 py-2 hover:bg-slate-700 cursor-pointer flex justify-between items-center text-sm"
                  >
                    <span className="font-bold text-slate-200">{stock.code}</span>
                    <span className="text-slate-400">{stock.name}</span>
                  </div>
                ))
              ) : (
                <div className="px-4 py-3 text-xs text-slate-500 text-center">无匹配股票</div>
              )}
            </div>
          )}
        </div>
      </header>

      {/* Main Content Layout */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Left Sidebar */}
        <StrategySidebar 
          strategies={ALL_STRATEGIES}
          selectedIds={selectedStrategyIds}
          onToggle={toggleStrategy}
          mode={combinationMode}
          setMode={setCombinationMode}
          threshold={votingThreshold}
          setThreshold={setVotingThreshold}
          params={params}
          setParams={setParams}
          onRun={handleRunBacktest}
          isRunning={isRunning}
        />

        {/* Right Content Area */}
        <div className="flex-1 overflow-y-auto bg-slate-900/30 p-6 flex flex-col gap-6 scroll-smooth">
          
          {/* Loading State Overlay */}
          {isRunning && (
            <div className="w-full bg-blue-500/10 border border-blue-500/20 text-blue-400 px-4 py-2 rounded-lg text-sm flex items-center justify-center gap-2 animate-pulse">
              <Loader2 size={16} className="animate-spin"/> 正在模拟回测数据，请稍候...
            </div>
          )}

          {!backtestResult && !isRunning ? (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
              <BarChart2 size={64} className="mb-6 opacity-20" />
              <h2 className="text-xl font-semibold mb-2 text-slate-500">等待回测</h2>
              <p className="text-sm">请在左侧选择策略配置参数，然后点击“运行回测”</p>
            </div>
          ) : backtestResult ? (
            <>
              {/* 1. Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <MetricCard label="总收益率 (Return)" value={`${backtestResult.returnPct > 0 ? '+' : ''}${backtestResult.returnPct}%`} highlight={backtestResult.returnPct > 0} />
                <MetricCard label="年化收益 (Annual)" value={`${backtestResult.annualReturnPct}%`} />
                <MetricCard label="最大回撤 (Max DD)" value={`${backtestResult.maxDrawdownPct}%`} subValue="Drawdown" />
                <MetricCard label="夏普比率 (Sharpe)" value={backtestResult.sharpe} />
                <MetricCard label="胜率 (Win Rate)" value={`${backtestResult.winRatePct}%`} />
                <MetricCard label="收益回撤比 (Calmar)" value={backtestResult.calmar} />
              </div>

              {/* 2. Chart Section */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg min-h-[450px] flex flex-col">
                <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-2">
                  <div className="flex gap-2">
                     <button 
                      onClick={() => setActiveTab('price')} 
                      className={`px-4 py-1.5 text-sm font-medium rounded-lg transition-colors ${activeTab === 'price' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                     >
                       价格 & 信号
                     </button>
                     <button 
                      onClick={() => setActiveTab('equity')} 
                      className={`px-4 py-1.5 text-sm font-medium rounded-lg transition-colors ${activeTab === 'equity' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                     >
                       资金曲线
                     </button>
                     <button 
                      onClick={() => setActiveTab('indicator')} 
                      className={`px-4 py-1.5 text-sm font-medium rounded-lg transition-colors ${activeTab === 'indicator' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                     >
                       指标图 (RSI)
                     </button>
                  </div>
                  <div className="text-xs text-slate-500 font-mono">
                    区间: {params.startDate} ~ {params.endDate}
                  </div>
                </div>

                <div className="flex-1 w-full h-[380px]">
                  <ResponsiveContainer width="100%" height="100%">
                    {activeTab === 'price' ? (
                      <ComposedChart data={backtestResult.priceSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="date" stroke="#64748b" tick={{fontSize: 12}} minTickGap={30} tickLine={false} axisLine={false} />
                        <YAxis domain={['auto', 'auto']} stroke="#64748b" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc'}}
                          itemStyle={{color: '#e2e8f0'}} 
                        />
                        <Legend />
                        <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={2} dot={false} name="Price" />
                        {backtestResult.tradeSignals.map((signal, idx) => (
                          <ReferenceDot 
                            key={idx} 
                            x={signal.date} 
                            y={signal.price} 
                            r={4} 
                            fill={signal.type === 'BUY' ? '#10b981' : '#ef4444'} 
                            stroke="none"
                          />
                        ))}
                      </ComposedChart>
                    ) : activeTab === 'equity' ? (
                      <AreaChart data={backtestResult.equityCurve}>
                        <defs>
                          <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="date" stroke="#64748b" tick={{fontSize: 12}} minTickGap={30} tickLine={false} axisLine={false} />
                        <YAxis domain={['auto', 'auto']} stroke="#64748b" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc'}} />
                        <Area type="monotone" dataKey="equity" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorEquity)" strokeWidth={2} name="Equity" />
                      </AreaChart>
                    ) : (
                      <LineChart data={backtestResult.indicatorSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="date" stroke="#64748b" tick={{fontSize: 12}} minTickGap={30} tickLine={false} axisLine={false} />
                        <YAxis domain={[0, 100]} stroke="#64748b" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc'}} />
                        <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
                        <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="rsi" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="RSI" />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 3. Trade Log */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col min-h-[300px] mb-6 shadow-lg">
                <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900">
                  <h3 className="font-bold text-slate-200 flex items-center gap-2">
                    <TrendingUp size={18} className="text-slate-400" />
                    交易记录 (Trade Log)
                  </h3>
                  <span className="text-xs font-mono text-slate-500 bg-slate-950 px-2 py-1 rounded border border-slate-800">
                    Total Trades: {backtestResult.trades.length}
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-slate-950 text-xs text-slate-400 uppercase tracking-wider sticky top-0">
                      <tr>
                        <th className="px-6 py-3 font-medium">Date</th>
                        <th className="px-6 py-3 font-medium">Side</th>
                        <th className="px-6 py-3 font-medium">Price</th>
                        <th className="px-6 py-3 font-medium">Qty</th>
                        <th className="px-6 py-3 font-medium text-right">Profit %</th>
                        <th className="px-6 py-3 font-medium text-right">Cum Profit %</th>
                        <th className="px-6 py-3 font-medium text-center">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {backtestResult.trades.map(t => (
                        <tr key={t.id} className="hover:bg-slate-800/50 transition-colors">
                          <td className="px-6 py-3 font-mono text-slate-400">{t.date}</td>
                          <td className="px-6 py-3">
                             <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold border ${
                              t.side === 'BUY' 
                                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                                : 'bg-red-500/10 text-red-400 border-red-500/20'
                            }`}>
                              {t.side === 'BUY' ? <ArrowDownRight size={12}/> : <ArrowUpRight size={12}/>}
                              {t.side}
                            </span>
                          </td>
                          <td className="px-6 py-3 font-mono text-slate-200">{t.price.toFixed(2)}</td>
                          <td className="px-6 py-3 font-mono text-slate-400">{t.qty}</td>
                          <td className={`px-6 py-3 font-mono text-right font-bold ${t.profitPct > 0 ? 'text-emerald-400' : t.profitPct < 0 ? 'text-red-400' : 'text-slate-500'}`}>
                            {t.side === 'SELL' ? `${t.profitPct > 0 ? '+' : ''}${t.profitPct}%` : '-'}
                          </td>
                          <td className={`px-6 py-3 font-mono text-right ${t.cumProfitPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {t.cumProfitPct > 0 ? '+' : ''}{t.cumProfitPct.toFixed(2)}%
                          </td>
                          <td className="px-6 py-3 text-center">
                            <span className={`text-[10px] uppercase px-2 py-0.5 rounded-full ${
                              t.positionDirection === 'LONG' ? 'bg-emerald-900 text-emerald-300' : 'bg-slate-700 text-slate-400'
                            }`}>
                              {t.positionDirection}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default StrategyBacktestPage;