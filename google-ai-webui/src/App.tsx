import React, { useState, useEffect } from 'react';
import { LayoutDashboard, LineChart, TrendingUp, Settings, Bell, Search } from 'lucide-react';
import MarketChart from './components/MarketChart';
import StrategyPanel from './components/StrategyPanel';
import AIAnalyst from './components/AIAnalyst';
import MarketStats from './components/MarketStats';
import StrategyBacktestPage from './components/StrategyBacktestPage';
import StockForecastPage from './components/StockForecastPage';
import StrategySettingsPage from './components/StrategySettingsPage';
import { StockData, Position, Strategy, IndustryData, MarketStatsData } from './types';

// Mock Data Generators (Simulating Daily bars with OHLC)
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
      ma20: close + (Math.random() - 0.5) * 150
    });
  }
  return data;
};

// Mock Industry Data
const mockIndustryData: IndustryData[] = [
  { name: '白酒', change: 2.35, sentiment: 'High' },
  { name: '新能源车', change: -1.20, sentiment: 'Low' },
  { name: '半导体', change: 0.85, sentiment: 'Medium' },
  { name: '银行', change: 0.45, sentiment: 'Medium' },
  { name: '医药生物', change: -0.65, sentiment: 'Low' },
];

// Mock Market Stats
const mockMarketStats: MarketStatsData = {
  limitUp: 45,
  up: 2800,
  flat: 300,
  down: 1800,
  limitDown: 12,
  sentimentScore: 68
};

// We keep positions in state for the AI analyst context, but don't display the table
const initialPositions: Position[] = [
  { symbol: '600519', name: '贵州茅台', amount: 200, avgPrice: 1680.50, currentPrice: 1750.20, pnl: 13940.0, pnlPercent: 4.15 },
  { symbol: '300750', name: '宁德时代', amount: 500, avgPrice: 195.00, currentPrice: 188.50, pnl: -3250.0, pnlPercent: -3.33 },
];

const initialStrategies: Strategy[] = [
  { id: '1', name: '双均线趋势 (MA5/20)', status: 'active', returnRate: 15.5, drawdown: 5.2, sharpeRatio: 1.6, description: '日线MA5上穿MA20买入，下穿卖出。' },
  { id: '2', name: 'RSI 超卖反转', status: 'paused', returnRate: -2.1, drawdown: 6.8, sharpeRatio: 0.5, description: 'RSI < 30 时分批建仓，RSI > 70 止盈。' },
  { id: '3', name: '布林带突破', status: 'stopped', returnRate: 6.4, drawdown: 1.5, sharpeRatio: 2.1, description: '股价突破布林带上轨追涨。' },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [marketData, setMarketData] = useState<StockData[]>(generateMarketData(60));
  const [strategies, setStrategies] = useState<Strategy[]>(initialStrategies);
  const [positions] = useState<Position[]>(initialPositions);

  // Simulate Live Data update
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketData(prev => {
        // Just subtly update the last candle's close for "live" feel
        const last = prev[prev.length - 1];
        const newClose = last.close + (Math.random() - 0.5) * 2;
        const updatedLast = {
          ...last,
          close: newClose,
          high: Math.max(last.high, newClose),
          low: Math.min(last.low, newClose)
        };
        return [...prev.slice(0, -1), updatedLast];
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const toggleStrategy = (id: string) => {
    setStrategies(prev => prev.map(s => 
      s.id === id ? { ...s, status: s.status === 'active' ? 'paused' : 'active' } : s
    ));
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 flex font-sans">
      
      {/* Sidebar */}
      <aside className="w-20 lg:w-64 bg-slate-950 border-r border-slate-800 flex flex-col fixed h-full z-10 transition-all duration-300">
        <div className="h-16 flex items-center justify-center lg:justify-start lg:px-6 border-b border-slate-800">
          <div className="w-8 h-8 bg-red-600 rounded-lg flex items-center justify-center font-bold text-white shadow-lg shadow-red-500/30">Q</div>
          <span className="ml-3 font-bold text-xl text-white hidden lg:block tracking-tight">QuantMind A股</span>
        </div>

        <nav className="flex-1 py-6 space-y-2 px-2 lg:px-4">
          {[
            { id: 'dashboard', icon: LayoutDashboard, label: '大盘总览 Dashboard' },
            { id: 'backtest', icon: LineChart, label: '策略回测 Backtest' },
            { id: 'forecast', icon: TrendingUp, label: '走势预测 Forecast' },
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
      <main className="flex-1 ml-20 lg:ml-64 p-4 lg:p-8 overflow-x-hidden h-screen flex flex-col">
        
        {/* Header */}
        {activeTab !== 'forecast' && activeTab !== 'settings' && (
          <header className="flex justify-between items-center mb-6 shrink-0">
            <div>
              <h1 className="text-2xl font-bold text-white">
                {activeTab === 'dashboard' && 'A股市场全景 (A-Share Overview)'}
                {activeTab === 'backtest' && '策略回测 (Strategy Backtesting)'}
              </h1>
              <p className="text-slate-400 text-sm mt-1">
                 {activeTab === 'dashboard' ? (
                   <>上证指数: <span className="text-red-400 font-bold">3,052.14 (+1.2%)</span> | 成交量: <span className="text-slate-200">8,500亿</span></>
                 ) : (
                   '专业量化回测引擎 | 支持日线级别多策略并发测试'
                 )}
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="relative hidden md:block">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
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
        )}

        {/* Content Switching Logic */}
        
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 overflow-y-auto">
            <div className="xl:col-span-2 space-y-6">
              <div className="h-[450px]">
                <MarketChart data={marketData} symbol="上证指数" />
              </div>
              <MarketStats industryData={mockIndustryData} marketStats={mockMarketStats} />
            </div>
            <div className="xl:col-span-1 space-y-6">
              <StrategyPanel strategies={strategies} onToggleStrategy={toggleStrategy} />
              <AIAnalyst positions={positions} strategies={strategies} />
            </div>
          </div>
        )}

        {activeTab === 'backtest' && (
          <StrategyBacktestPage />
        )}

        {activeTab === 'forecast' && (
          <StockForecastPage />
        )}

        {activeTab === 'settings' && (
          <StrategySettingsPage />
        )}

      </main>
    </div>
  );
};

export default App;