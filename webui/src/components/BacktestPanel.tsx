import React, { useState } from 'react';
import type { Strategy, BacktestResult, StockData, TradeRecord } from '../types';
import MarketChart from './MarketChart';
import { Play, Calendar, Search, BarChart2, TrendingUp, Clock, Wallet } from 'lucide-react';

interface BacktestPanelProps {
  strategies: Strategy[];
}

// Helper to generate mock backtest data (Simulating a backend)
const simulateBacktest = (symbol: string, startDate: string, endDate: string, strategyId: string, initialCapital: number): BacktestResult => {
  const data: StockData[] = [];
  const trades: TradeRecord[] = [];

  const strategyBias = strategyId.length % 10;
  let currentPrice = 100 + Math.random() * 50 + strategyBias;
  const start = new Date(startDate);
  const end = new Date(endDate);
  const totalDays = Math.ceil((end.getTime() - start.getTime()) / (1000 * 3600 * 24));
  
  // Generate OHLC Data
  let holding = false;
  let buyPrice = 0;

  for (let i = 0; i <= totalDays; i++) {
    const currentDate = new Date(start);
    currentDate.setDate(start.getDate() + i);
    
    // Skip weekends
    if (currentDate.getDay() === 0 || currentDate.getDay() === 6) continue;

    const volatility = currentPrice * 0.02; // 2% volatility
    const open = currentPrice + (Math.random() - 0.5) * volatility;
    const close = open + (Math.random() - 0.5) * volatility * 1.5;
    const high = Math.max(open, close) + Math.random() * volatility * 0.5;
    const low = Math.min(open, close) - Math.random() * volatility * 0.5;
    const volume = Math.floor(Math.random() * 10000000);

    const dateStr = currentDate.toISOString().split('T')[0];

    data.push({
      time: dateStr,
      open, high, low, close, volume,
      ma5: close, // Simplified for mock
      ma20: close // Simplified for mock
    });

    // Simple Mock Strategy Logic to generate trades
    // Buy randomly if not holding, Sell randomly if holding
    if (!holding && Math.random() > 0.9) {
      holding = true;
      buyPrice = close;
      trades.push({
        id: `trade-${i}-buy`,
        date: dateStr,
        type: 'BUY',
        price: parseFloat(close.toFixed(2)),
        amount: 1000
      });
    } else if (holding && Math.random() > 0.85) {
      holding = false;
      const profitPercent = ((close - buyPrice) / buyPrice) * 100;
      trades.push({
        id: `trade-${i}-sell`,
        date: dateStr,
        type: 'SELL',
        price: parseFloat(close.toFixed(2)),
        amount: 1000,
        profit: (close - buyPrice) * 1000,
        profitPercent: parseFloat(profitPercent.toFixed(2))
      });
    }

    currentPrice = close;
  }

  // Calculate Mock Stats
  const winRate = trades.filter(t => t.type === 'SELL' && (t.profitPercent || 0) > 0).length / (trades.filter(t => t.type === 'SELL').length || 1) * 100;
  const totalReturnPercent = (Math.random() * 40) - 10; // Random return -10% to +30%
  const totalProfit = initialCapital * (totalReturnPercent / 100);

  return {
    symbol,
    startDate,
    endDate,
    totalReturn: parseFloat(totalReturnPercent.toFixed(2)),
    totalProfit: parseFloat(totalProfit.toFixed(2)),
    maxDrawdown: parseFloat((Math.random() * 15).toFixed(2)),
    sharpeRatio: parseFloat((Math.random() * 2 + 0.5).toFixed(2)),
    winRate: parseFloat(winRate.toFixed(1)),
    trades,
    data
  };
};

const BacktestPanel: React.FC<BacktestPanelProps> = ({ strategies }) => {
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy>(strategies[0]);
  const [symbol, setSymbol] = useState('600519');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
  const [initialCapital, setInitialCapital] = useState('100000');
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = () => {
    setLoading(true);
    // Simulate async operation
    setTimeout(() => {
      const cap = parseFloat(initialCapital) || 100000;
      const res = simulateBacktest(symbol, startDate, endDate, selectedStrategy.id, cap);
      setResult(res);
      setLoading(false);
    }, 800);
  };

  return (
    <div className="flex flex-col xl:flex-row gap-6 min-h-[calc(100vh-140px)]">
      
      {/* Sidebar - Strategy List & Config */}
      {/* Made sticky so it stays visible when scrolling long trade logs */}
      <div className="xl:w-80 flex flex-col gap-6 xl:sticky xl:top-0 xl:h-fit shrink-0">
        
        {/* Strategy List */}
        <div className="bg-slate-800 rounded-xl p-4 shadow-lg border border-slate-700 max-h-[400px] overflow-y-auto custom-scrollbar">
          <h3 className="text-white font-bold mb-4 flex items-center gap-2">
            <BarChart2 size={18} className="text-purple-400" />
            选择策略 (Strategies)
          </h3>
          <div className="space-y-3">
            {strategies.map(s => (
              <div 
                key={s.id} 
                onClick={() => setSelectedStrategy(s)}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedStrategy.id === s.id 
                    ? 'bg-purple-900/30 border-purple-500 shadow-md shadow-purple-900/20' 
                    : 'bg-slate-700/30 border-transparent hover:bg-slate-700 hover:border-slate-600'
                }`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className={`font-semibold ${selectedStrategy.id === s.id ? 'text-purple-300' : 'text-slate-200'}`}>{s.name}</span>
                  {selectedStrategy.id === s.id && <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>}
                </div>
                <p className="text-xs text-slate-400 line-clamp-2">{s.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Configuration */}
        <div className="bg-slate-800 rounded-xl p-5 shadow-lg border border-slate-700">
           <h3 className="text-white font-bold mb-4 flex items-center gap-2">
            <Clock size={18} className="text-blue-400" />
            回测参数 (Parameters)
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="text-xs text-slate-400 font-semibold uppercase mb-1 block">股票代码 (Symbol)</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                <input 
                  type="text" 
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg pl-10 pr-3 py-2 text-sm text-white focus:border-purple-500 outline-none font-mono"
                />
              </div>
            </div>

            <div>
              <label className="text-xs text-slate-400 font-semibold uppercase mb-1 block">模拟初始资金 (Initial Capital)</label>
              <div className="relative">
                <Wallet className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                <input 
                  type="number" 
                  value={initialCapital}
                  onChange={(e) => setInitialCapital(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg pl-10 pr-3 py-2 text-sm text-white focus:border-purple-500 outline-none font-mono"
                  placeholder="100000"
                />
              </div>
            </div>

            <div>
              <label className="text-xs text-slate-400 font-semibold uppercase mb-1 block">日期范围 (Date Range)</label>
              <div className="flex flex-col gap-2">
                <div className="relative">
                  <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={14} />
                  <input 
                    type="date" 
                    value={startDate} 
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-600 rounded-lg pl-9 pr-2 py-2 text-xs text-white focus:border-purple-500 outline-none font-mono" 
                  />
                </div>
                <div className="relative">
                  <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={14} />
                  <input 
                    type="date" 
                    value={endDate} 
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-600 rounded-lg pl-9 pr-2 py-2 text-xs text-white focus:border-purple-500 outline-none font-mono" 
                  />
                </div>
              </div>
            </div>

            <button 
              onClick={handleRun}
              disabled={loading}
              className="w-full mt-2 bg-purple-600 hover:bg-purple-500 text-white py-2.5 rounded-lg font-bold flex justify-center items-center gap-2 transition-all disabled:opacity-50"
            >
              {loading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"/> : <Play size={18} />}
              运行回测 (Run)
            </button>
          </div>
        </div>
      </div>

      {/* Main View - Results & Chart - Allowed to grow to push footer down */}
      <div className="flex-1 flex flex-col gap-6 w-full min-w-0">
        
        {/* Statistics Cards */}
        {result && (
          <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">总回报率 (Return)</span>
              <div className={`text-2xl font-bold font-mono mt-1 ${result.totalReturn >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                {result.totalReturn > 0 ? '+' : ''}{result.totalReturn}%
              </div>
            </div>
             <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">最大回撤 (Max DD)</span>
              <div className="text-2xl font-bold font-mono mt-1 text-emerald-400">
                -{result.maxDrawdown}%
              </div>
            </div>
             <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">夏普比率 (Sharpe)</span>
              <div className="text-2xl font-bold font-mono mt-1 text-blue-400">
                {result.sharpeRatio}
              </div>
            </div>
            <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">胜率 (Win Rate)</span>
              <div className="text-2xl font-bold font-mono mt-1 text-purple-400">
                {result.winRate}%
              </div>
            </div>
             <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">总收益 (Profit)</span>
              <div className={`text-2xl font-bold font-mono mt-1 ${result.totalProfit >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                 {result.totalProfit >= 0 ? '+' : ''}{result.totalProfit.toLocaleString()}
              </div>
            </div>
             <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 xl:hidden">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">交易次数 (Trades)</span>
              <div className="text-2xl font-bold font-mono mt-1 text-slate-200">
                {result.trades.length}
              </div>
            </div>
          </div>
        )}

        {/* Chart Area - Fixed height for better visualization */}
        <div className="bg-slate-800 rounded-xl shadow-lg border border-slate-700 overflow-hidden relative h-[500px] shrink-0">
          {!result && !loading && (
             <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                <BarChart2 size={48} className="mb-4 opacity-50" />
                <p>请在左侧选择策略并点击运行回测</p>
             </div>
          )}
          
          {loading && (
             <div className="absolute inset-0 flex items-center justify-center bg-slate-800/80 z-10">
               <div className="text-purple-400 flex flex-col items-center">
                 <div className="w-10 h-10 border-4 border-current border-t-transparent rounded-full animate-spin mb-4"></div>
                 <p className="font-mono text-sm">正在计算策略回测数据...</p>
               </div>
             </div>
          )}

          {result && (
            <div className="h-full p-2">
              <MarketChart data={result.data} symbol={result.symbol} trades={result.trades} />
            </div>
          )}
        </div>
        
        {/* Trade Log - Now flows naturally with the page */}
        {result && result.trades.length > 0 && (
           <div className="bg-slate-800 rounded-xl border border-slate-700 flex flex-col overflow-hidden mb-6">
              <div className="px-4 py-3 border-b border-slate-700 bg-slate-900/50 flex justify-between items-center">
                <span className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                  <TrendingUp size={16} className="text-slate-400"/> 
                  交易记录 (Trade Log)
                </span>
                <span className="text-xs text-slate-500 font-mono">Total Trades: {result.trades.length}</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-900/30 text-xs text-slate-400">
                    <tr>
                      <th className="px-6 py-3 font-medium">日期</th>
                      <th className="px-6 py-3 font-medium">类型</th>
                      <th className="px-6 py-3 font-medium">成交价格</th>
                      <th className="px-6 py-3 font-medium">数量</th>
                      <th className="px-6 py-3 font-medium text-right">单笔盈亏</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/50">
                    {result.trades.map(t => (
                      <tr key={t.id} className="hover:bg-slate-700/30 transition-colors">
                        <td className="px-6 py-3 font-mono text-slate-300">{t.date}</td>
                        <td className="px-6 py-3">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-bold border ${
                            t.type === 'BUY' 
                              ? 'bg-red-500/10 text-red-400 border-red-500/20' 
                              : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                          }`}>
                            {t.type === 'BUY' ? '买入' : '卖出'}
                          </span>
                        </td>
                        <td className="px-6 py-3 font-mono text-slate-200">¥{t.price.toFixed(2)}</td>
                        <td className="px-6 py-3 font-mono text-slate-400">{t.amount}</td>
                        <td className={`px-6 py-3 font-mono text-right font-medium ${!t.profit ? 'text-slate-600' : t.profit > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                          {t.profit 
                            ? `${t.profit > 0 ? '+' : ''}¥${Math.abs(t.profit).toFixed(2)} (${t.profit > 0 ? '+' : ''}${t.profitPercent}%)` 
                            : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
           </div>
        )}

      </div>
    </div>
  );
};

export default BacktestPanel;
