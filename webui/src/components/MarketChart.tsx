import React from 'react';
import {
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Bar,
  Line,
  Cell,
  ReferenceDot,
  Label,
} from 'recharts';
import type { TooltipContentProps } from 'recharts';
import type { StockData, TradeRecord } from '../types';

interface MarketChartProps {
  data: StockData[];
  symbol: string;
  trades?: TradeRecord[]; // Optional trades for backtesting visualization
}

type ChartDatum = StockData & {
  bodyRange: [number, number];
  color: string;
};

const MarketChart: React.FC<MarketChartProps> = ({ data, symbol, trades }) => {
  // Transform data for Recharts Range Bar
  const chartData: ChartDatum[] = data.map((d) => ({
    ...d,
    // [min, max] for the candle body
    bodyRange: [Math.min(d.open, d.close), Math.max(d.open, d.close)],
    // Custom color field
    color: d.close > d.open ? '#ef4444' : '#10b981'
  }));

  // Helper to format tooltip
  const renderTooltip = (props: TooltipContentProps<number, string>) => {
    const { active, payload, label } = props;
    if (active && payload && payload.length) {
      const datum = payload[0]?.payload as ChartDatum | undefined;
      if (!datum) return null;

      const labelStr = typeof label === 'string' ? label : label?.toString() ?? '';

      return (
        <div className="bg-slate-800 border border-slate-600 p-3 rounded shadow-xl text-xs text-slate-200">
          <p className="font-bold mb-2 text-white">{labelStr}</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <span className="text-slate-400">Open:</span>{' '}
            <span className="font-mono">{datum.open.toFixed(2)}</span>
            <span className="text-slate-400">High:</span>{' '}
            <span className="font-mono">{datum.high.toFixed(2)}</span>
            <span className="text-slate-400">Low:</span>{' '}
            <span className="font-mono">{datum.low.toFixed(2)}</span>
            <span className="text-slate-400">Close:</span>{' '}
            <span
              className={`font-mono ${datum.close > datum.open ? 'text-red-400' : 'text-emerald-400'}`}
            >
              {datum.close.toFixed(2)}
            </span>
            <span className="text-slate-400">Vol:</span>{' '}
            <span className="font-mono">{(datum.volume / 1000000).toFixed(1)}M</span>
          </div>
          {trades && (
            <div className="mt-2 pt-2 border-t border-slate-700">
              {trades
                .filter((t) => t.date === labelStr)
                .map((t) => (
                  <div
                    key={t.id}
                    className={`font-bold ${t.type === 'BUY' ? 'text-red-500' : 'text-emerald-500'}`}
                  >
                    {t.type === 'BUY' ? '买入' : '卖出'} @ {t.price}
                  </div>
                ))}
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="bg-red-600 px-2 py-0.5 rounded text-sm">A股大盘</span>
            {symbol} 行情走势
          </h3>
          <p className="text-slate-400 text-xs mt-1">代码: {symbol} - {trades ? '回测视角' : '日线视角'}</p>
        </div>
        <div className="flex gap-2">
           {['日线', '周线', '月线'].map((tf) => (
             <button key={tf} className={`px-3 py-1 rounded text-xs font-medium transition-colors ${tf === '日线' ? 'bg-red-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
               {tf}
             </button>
           ))}
        </div>
      </div>

      <div className="flex-grow w-full h-64 min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis 
              dataKey="time" 
              stroke="#94a3b8" 
              tick={{fontSize: 12}} 
              tickLine={false}
              axisLine={false}
              minTickGap={30}
            />
            <YAxis 
              stroke="#94a3b8" 
              tick={{fontSize: 12}} 
              tickLine={false}
              axisLine={false}
              domain={['auto', 'auto']}
            />
            <Tooltip content={renderTooltip} cursor={{fill: 'transparent', stroke: '#cbd5e1', strokeWidth: 1, strokeDasharray: '4 4'}} />
            
            {/* Candle Body */}
            <Bar dataKey="bodyRange" barSize={10} minPointSize={2}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>

            <Line type="monotone" dataKey="ma5" stroke="#f59e0b" strokeWidth={1} dot={false} name="MA5" />
            <Line type="monotone" dataKey="ma20" stroke="#3b82f6" strokeWidth={1} dot={false} name="MA20" />

            {/* Trade Annotations */}
            {trades && trades.map((trade) => (
              <ReferenceDot
                key={trade.id}
                x={trade.date}
                y={trade.price}
                r={6}
                fill={trade.type === 'BUY' ? '#ef4444' : '#10b981'}
                stroke="#fff"
                strokeWidth={2}
                ifOverflow="extendDomain"
              >
                <Label 
                  value={trade.type === 'BUY' ? 'B' : 'S'} 
                  position="center" 
                  fill="#fff" 
                  fontSize={10} 
                  fontWeight="bold"
                />
              </ReferenceDot>
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MarketChart;
