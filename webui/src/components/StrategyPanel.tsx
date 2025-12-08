import React, { useState } from 'react';
import type { Strategy } from '../types';
import { Play, Pause, Activity, Search, Calculator, Calendar } from 'lucide-react';

interface StrategyPanelProps {
  strategies: Strategy[];
  onToggleStrategy: (id: string) => void;
}

const StrategyPanel: React.FC<StrategyPanelProps> = ({ strategies, onToggleStrategy }) => {
  const [selectedStock, setSelectedStock] = useState('600519');
  const [startDate, setStartDate] = useState('2015-01-01');
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]); // Default to today
  const [isBacktesting, setIsBacktesting] = useState(false);

  const handleRunBacktest = () => {
    setIsBacktesting(true);
    setTimeout(() => {
      setIsBacktesting(false);
      // In a real app, this would update stats based on the selected date range
    }, 2000);
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 h-full flex flex-col">
      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <Activity className="text-purple-400" size={20} />
        策略运行实验室
      </h3>

      {/* Backtest Configuration Panel */}
      <div className="mb-6 bg-slate-900/50 p-4 rounded-lg border border-slate-700 space-y-4">
        
        {/* Stock Selection */}
        <div>
          <label className="text-xs text-slate-400 mb-1.5 block uppercase tracking-wider font-semibold">标的股票 (Target Asset)</label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
            <input 
              type="text" 
              value={selectedStock}
              onChange={(e) => setSelectedStock(e.target.value)}
              placeholder="输入代码 (如 600519)"
              className="w-full bg-slate-800 border border-slate-600 rounded-lg pl-10 pr-4 py-2 text-sm text-white focus:outline-none focus:border-purple-500 font-mono"
            />
          </div>
        </div>

        {/* Date Range Selection */}
        <div>
           <label className="text-xs text-slate-400 mb-1.5 block uppercase tracking-wider font-semibold flex items-center gap-1">
             <Calendar size={12} /> 回测区间 (Backtest Period)
           </label>
           <div className="flex items-center gap-2">
             <input 
                type="date" 
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-purple-500 font-mono"
             />
             <span className="text-slate-500">-</span>
             <input 
                type="date" 
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-purple-500 font-mono"
             />
           </div>
        </div>

        {/* Action Button */}
        <button 
          onClick={handleRunBacktest}
          disabled={isBacktesting}
          className="w-full bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white px-4 py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors shadow-lg shadow-purple-900/20"
        >
          {isBacktesting ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"/> : <Calculator size={16} />}
          开始回测 (Run Backtest)
        </button>
      </div>
      
      {/* Strategies List */}
      <div className="space-y-4 flex-1 overflow-y-auto pr-1 custom-scrollbar">
        {strategies.map((strategy) => (
          <div key={strategy.id} className="bg-slate-700/50 rounded-lg p-4 border border-slate-600 hover:border-slate-500 transition-all">
            <div className="flex justify-between items-start mb-3">
              <div>
                <h4 className="font-semibold text-white text-md flex items-center gap-2">
                   {strategy.name}
                   {strategy.status === 'active' && <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>}
                </h4>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">{strategy.description}</p>
              </div>
            </div>

            {/* Strategy Stats */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              <div className="bg-slate-800/80 rounded p-2 text-center">
                <span className="text-slate-500 text-[10px] block uppercase">Return</span>
                <span className={`font-mono text-sm font-bold ${strategy.returnRate >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                  {strategy.returnRate}%
                </span>
              </div>
              <div className="bg-slate-800/80 rounded p-2 text-center">
                <span className="text-slate-500 text-[10px] block uppercase">Drawdown</span>
                <span className="text-emerald-400 font-mono text-sm font-bold">-{strategy.drawdown}%</span>
              </div>
              <div className="bg-slate-800/80 rounded p-2 text-center">
                <span className="text-slate-500 text-[10px] block uppercase">Sharpe</span>
                <span className="text-blue-400 font-mono text-sm font-bold">{strategy.sharpeRatio}</span>
              </div>
            </div>

            <div className="flex justify-between items-center border-t border-slate-600/50 pt-3 mt-2">
              <span className="text-xs text-slate-500">
                区间: <span className="text-slate-300 font-mono">{startDate.split('-')[0]}..{endDate.split('-')[0]}</span>
              </span>
              <button 
                onClick={() => onToggleStrategy(strategy.id)}
                className={`text-xs px-3 py-1.5 rounded font-medium flex items-center gap-1 transition-colors ${
                  strategy.status === 'active' 
                    ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20' 
                    : 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20'
                }`}
              >
                {strategy.status === 'active' ? <Pause size={12} /> : <Play size={12} />}
                {strategy.status === 'active' ? '停止' : '运行'}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StrategyPanel;
