// webui/src/components/StrategyPanel.tsx
import React, { useState } from 'react';
import type { Strategy } from '../types';
import { Play, Pause, Activity, Search, Calculator, Calendar } from 'lucide-react';

interface StrategyPanelProps {
  strategies: Strategy[];
  onToggleStrategy: (id: string) => void;
}

const StrategyPanel: React.FC<StrategyPanelProps> = ({
  strategies,
  onToggleStrategy,
}) => {
  const [selectedStock, setSelectedStock] = useState('600519');
  const [startDate, setStartDate] = useState('2015-01-01');
  const [endDate, setEndDate] = useState(
    new Date().toISOString().split('T')[0]
  );
  const [isBacktesting, setIsBacktesting] = useState(false);

  const handleRunBacktest = () => {
    setIsBacktesting(true);
    setTimeout(() => {
      setIsBacktesting(false);
      // 这里预留给未来：根据区间更新一些统计
    }, 2000);
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 h-full flex flex-col">
      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <Activity className="text-purple-400" size={20} />
        策略运行实验室
      </h3>

      {/* 回测配置区域 */}
      <div className="mb-6 bg-slate-900/50 p-4 rounded-lg border border-slate-700 space-y-4">
        {/* 标的选择 */}
        <div>
          <label className="text-xs text-slate-400 mb-1.5 block uppercase tracking-wider font-semibold">
            标的股票 (Target Asset)
          </label>
          <div className="relative">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
              size={16}
            />
            <input
              type="text"
              value={selectedStock}
              onChange={e => setSelectedStock(e.target.value)}
              placeholder="输入代码 (如 600519)"
              className="w-full bg-slate-800 border border-slate-600 rounded-lg pl-10 pr-4 py-2 text-sm text-white focus:outline-none focus:border-purple-500 font-mono"
            />
          </div>
        </div>

        {/* 回测区间 */}
        <div>
          <label className="text-xs text-slate-400 mb-1.5 block uppercase tracking-wider font-semibold flex items-center gap-1">
            <Calendar size={12} />
            回测区间 (Backtest Period)
          </label>
          <div className="flex items-center gap-2">
            <input
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-purple-500 font-mono"
            />
            <span className="text-slate-500">-</span>
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-purple-500 font-mono"
            />
          </div>
        </div>

        {/* 占位——以后可以加手续费、滑点等参数 */}
        <div>
          <label className="text-xs text-slate-400 mb-1.5 block uppercase tracking-wider font-semibold flex items-center gap-1">
            <Calculator size={12} />
            参数预留 (Params)
          </label>
          <p className="text-xs text-slate-500">
            暂时仅作演示，未来可以在这里增加手续费、滑点、仓位等参数。
          </p>
        </div>

        {/* 运行按钮 */}
        <button
          onClick={handleRunBacktest}
          disabled={isBacktesting}
          className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-slate-700 disabled:text-slate-400 text-white py-2.5 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all"
        >
          {isBacktesting ? (
            <>
              <Pause size={16} className="animate-pulse" />
              正在模拟回测...
            </>
          ) : (
            <>
              <Play size={16} />
              快速测试选中策略
            </>
          )}
        </button>
      </div>

      {/* 已选策略列表 */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">
            已选策略 (Active Strategies)
          </span>
        </div>
        <div className="space-y-2">
          {strategies.map(s => (
            <div
              key={s.id}
              className="flex items-center justify-between bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2"
            >
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-slate-100 font-medium">
                    {s.name}
                  </span>
                  {s.status === 'active' && (
                    <span className="inline-flex px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] border border-emerald-500/40">
                      ACTIVE
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400 mt-0.5 line-clamp-2">
                  {s.description}
                </p>
              </div>
              <button
                onClick={() => onToggleStrategy(s.id)}
                className="text-xs text-slate-400 hover:text-red-400 transition-colors"
              >
                切换
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StrategyPanel;
