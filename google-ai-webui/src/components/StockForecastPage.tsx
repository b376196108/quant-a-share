import React, { useState } from 'react';
import { 
  Search, 
  TrendingUp, 
  AlertCircle, 
  Loader2, 
  Calendar,
  ArrowUpRight,
  ArrowDownRight
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
  Legend,
  ReferenceLine
} from 'recharts';
import { StockForecastResponse } from '../types';

const StockForecastPage: React.FC = () => {
  const [symbolInput, setSymbolInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<StockForecastResponse | null>(null);

  const handleSearch = async () => {
    if (!symbolInput.trim()) return;
    
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const response = await fetch(`http://localhost:8000/api/forecast?symbol=${symbolInput}&days=5`);
      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.detail || '预测请求失败');
      }
      const jsonData = await response.json();
      setData(jsonData);
    } catch (err: any) {
      setError(err.message || '网络连接错误，请确保后端服务已启动');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 p-6 overflow-y-auto">
      
      {/* Header & Search */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <TrendingUp className="text-purple-500" />
            A股走势预测
          </h1>
          <p className="text-slate-400 text-sm mt-1">基于时间序列模型的短期价格预测 (T+5)</p>
        </div>
        
        <div className="flex items-center gap-2 w-full md:w-auto">
          <div className="relative flex-1 md:w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
            <input 
              type="text" 
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="输入股票代码 (如 600519)" 
              className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-purple-500 placeholder-slate-600 transition-all font-mono"
            />
          </div>
          <button 
            onClick={handleSearch}
            disabled={loading || !symbolInput}
            className="bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-500 text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-lg shadow-purple-900/20"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : '预测'}
          </button>
        </div>
      </div>

      {/* Content Area */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl flex items-center gap-3 mb-6">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {!data && !loading && !error && (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-600 border-2 border-dashed border-slate-800 rounded-xl min-h-[400px]">
          <TrendingUp size={64} className="mb-4 opacity-20" />
          <p className="text-lg font-medium">准备就绪</p>
          <p className="text-sm">请输入股票代码并点击搜索，查看未来 5 个交易日的 AI 预测走势</p>
        </div>
      )}

      {loading && !data && (
        <div className="flex-1 flex flex-col items-center justify-center min-h-[400px]">
          <Loader2 size={48} className="text-purple-500 animate-spin mb-4" />
          <p className="text-slate-400 animate-pulse">正在运行预测模型...</p>
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          
          {/* Left Chart Area */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
            <div className="flex justify-between items-center mb-6">
               <div>
                 <h2 className="text-lg font-bold text-white flex items-center gap-2">
                   {data.name || data.symbol} 价格预测趋势
                 </h2>
                 <p className="text-xs text-slate-500 mt-1">
                   模型: <span className="text-purple-400 bg-purple-900/20 px-1.5 py-0.5 rounded">{data.method}</span>
                   <span className="mx-2">|</span>
                   基准日: {data.last_date} (收盘价: {data.last_close})
                 </p>
               </div>
            </div>
            
            <div className="flex-1 w-full min-h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={data.forecast} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorLower" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis 
                    dataKey="date" 
                    stroke="#64748b" 
                    tick={{fontSize: 12}} 
                    tickLine={false} 
                    axisLine={false} 
                  />
                  <YAxis 
                    domain={['auto', 'auto']} 
                    stroke="#64748b" 
                    tick={{fontSize: 12}} 
                    tickLine={false} 
                    axisLine={false}
                    tickFormatter={(val) => val.toFixed(2)}
                  />
                  <Tooltip 
                    contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc'}}
                    itemStyle={{color: '#e2e8f0'}}
                    labelStyle={{color: '#94a3b8'}}
                    formatter={(val: number) => val.toFixed(2)}
                  />
                  <Legend />
                  <ReferenceLine y={data.last_close} stroke="#64748b" strokeDasharray="3 3" label={{ position: 'right', value: 'Current', fill: '#64748b', fontSize: 10 }} />
                  
                  {/* Confidence Interval (Simulated with Area if upper/lower exist) */}
                  {data.forecast[0].upper && (
                     <Area 
                       type="monotone" 
                       dataKey="upper" 
                       stroke="transparent" 
                       fill="#8b5cf6" 
                       fillOpacity={0.1} 
                       name="置信区间"
                     />
                  )}
                  
                  <Line 
                    type="monotone" 
                    dataKey="predicted_close" 
                    stroke="#8b5cf6" 
                    strokeWidth={3} 
                    dot={{r: 4, fill: '#8b5cf6', strokeWidth: 2, stroke: '#fff'}} 
                    name="预测收盘价" 
                    animationDuration={1500}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Right Data Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Calendar size={18} className="text-blue-400" />
              详细数据 (5 Days)
            </h3>
            
            <div className="overflow-hidden rounded-lg border border-slate-800">
              <table className="w-full text-sm text-left">
                <thead className="bg-slate-950 text-slate-400 text-xs uppercase">
                  <tr>
                    <th className="px-4 py-3 font-medium">日期</th>
                    <th className="px-4 py-3 font-medium text-right">预测价</th>
                    <th className="px-4 py-3 font-medium text-right">涨跌幅</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 bg-slate-900/50">
                  {data.forecast.map((point, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/80 transition-colors">
                      <td className="px-4 py-3 font-mono text-slate-300">{point.date}</td>
                      <td className="px-4 py-3 font-mono text-right font-bold text-white">
                        {point.predicted_close.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 font-mono text-right">
                        <span className={`inline-flex items-center gap-1 ${
                          point.change_pct >= 0 ? 'text-red-400' : 'text-emerald-400'
                        }`}>
                          {point.change_pct >= 0 ? <ArrowUpRight size={12}/> : <ArrowDownRight size={12}/>}
                          {Math.abs(point.change_pct).toFixed(2)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-auto pt-6 text-xs text-slate-500 border-t border-slate-800/50">
              <p className="flex items-start gap-2">
                <AlertCircle size={14} className="shrink-0 mt-0.5" />
                <span>
                  免责声明：预测结果仅基于历史数据和统计模型生成，不代表未来实际走势。股市有风险，投资需谨慎。
                </span>
              </p>
            </div>
          </div>

        </div>
      )}
    </div>
  );
};

export default StockForecastPage;
