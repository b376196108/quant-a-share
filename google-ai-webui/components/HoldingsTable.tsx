import React from 'react';
import { Position } from '../types';
import { ArrowUpRight, ArrowDownRight, MoreHorizontal } from 'lucide-react';

interface HoldingsTableProps {
  positions: Position[];
}

const HoldingsTable: React.FC<HoldingsTableProps> = ({ positions }) => {
  return (
    <div className="bg-slate-800 rounded-xl shadow-lg border border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center">
        <h3 className="font-bold text-white text-lg">当前持仓组合</h3>
        <button className="text-slate-400 hover:text-white"><MoreHorizontal size={20}/></button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-750 text-slate-400 text-xs uppercase tracking-wider">
              <th className="p-4 font-medium">股票代码</th>
              <th className="p-4 font-medium">持仓数量 (股)</th>
              <th className="p-4 font-medium">成本均价</th>
              <th className="p-4 font-medium">最新价</th>
              <th className="p-4 font-medium text-right">盈亏 (%)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700 text-sm">
            {positions.map((pos) => (
              <tr key={pos.symbol} className="hover:bg-slate-700/50 transition-colors">
                <td className="p-4">
                  <div className="font-bold text-white">{pos.symbol}</div>
                  <div className="text-xs text-slate-500">{pos.name}</div>
                </td>
                <td className="p-4 text-slate-300 font-mono">{pos.amount.toLocaleString()}</td>
                <td className="p-4 text-slate-300 font-mono">¥{pos.avgPrice.toFixed(2)}</td>
                <td className="p-4 text-white font-bold font-mono">¥{pos.currentPrice.toFixed(2)}</td>
                <td className="p-4 text-right">
                  <div className={`flex items-center justify-end gap-1 font-bold font-mono ${pos.pnlPercent >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                    {/* A-Share Color convention: Red is Up, Green is Down */}
                    {pos.pnlPercent >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                    {pos.pnlPercent > 0 ? '+' : ''}{pos.pnlPercent}%
                  </div>
                  <div className={`text-xs ${pos.pnl >= 0 ? 'text-red-500/70' : 'text-emerald-500/70'}`}>
                    ¥{Math.abs(pos.pnl).toLocaleString()}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default HoldingsTable;