import React, { useMemo, useState } from 'react';
import {
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Bar,
  Line,
  Scatter,
  Cell,
} from 'recharts';
import type { TooltipContentProps } from 'recharts';
import type { StockForecastResponse, StockForecastPoint, StockData } from '../types';

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') || '';

type ChartDatum = StockData & {
  bodyRange?: [number, number];
  color?: string;
  forecast?: boolean;
  predicted_close?: number;
  change_pct?: number;
  bandLow?: number;
  bandHigh?: number;
  forecastBody?: [number, number];
};

const formatPct = (v: number) => `${(v * 100).toFixed(2)}%`;

const StockForecastPage: React.FC = () => {
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<StockForecastResponse | null>(null);

  const handlePredict = async () => {
    const trimmed = code.trim();
    if (!trimmed) {
      setError('请输入股票代码，例如 600519');
      setData(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const resp = await fetch(`${API_BASE}/api/forecast?symbol=${encodeURIComponent(trimmed)}&days=5`);
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || '服务返回错误');
      }
      const json = (await resp.json()) as StockForecastResponse;
      setData(json);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || '预测失败，请稍后重试');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const chartData: ChartDatum[] = useMemo(() => {
    if (!data) return [];
    const hist = (data.history || []).map((d) => ({
      ...d,
      bodyRange: [Math.min(d.open, d.close), Math.max(d.open, d.close)] as [number, number],
      color: d.close >= d.open ? '#ef4444' : '#10b981',
      forecast: false,
    }));

    const lastClose = data.last_close;

    let prevClose = lastClose;
    const fc = data.forecast.map((f) => {
      const open = prevClose;
      const close = f.predicted_close;
      prevClose = close;
      const high = f.upper ?? Math.max(open, close);
      const low = f.lower ?? Math.min(open, close);
      return {
        time: f.date,
        open,
        high,
        low,
        close,
        volume: 0,
        ma5: close,
        ma20: close,
        forecast: true,
        predicted_close: close,
        change_pct: f.change_pct,
        bandLow: low,
        bandHigh: high,
        color: close >= open ? '#e5e7eb' : '#3b82f6',
        forecastBody: [Math.min(open, close), Math.max(open, close)] as [number, number],
      };
    });

    if (hist.length > 0) {
      const anchor = {
        ...hist[hist.length - 1],
        forecast: true,
        bodyRange: undefined,
        color: '#e5e7eb',
        predicted_close: lastClose,
        change_pct: 0,
        bandLow: undefined,
        bandHigh: undefined,
      };
      fc.unshift(anchor);
    }

    return [...hist, ...fc];
  }, [data]);

  const renderTooltip = (props: TooltipContentProps<number, string>) => {
    const { active, payload, label } = props;
    if (active && payload && payload.length) {
      const datum = payload[0]?.payload as ChartDatum | undefined;
      if (!datum) return null;
      const labelStr = typeof label === 'string' ? label : label?.toString() ?? '';

      if (datum.forecast) {
        return (
          <div className="bg-slate-800 border border-slate-600 p-3 rounded shadow-xl text-xs text-slate-200">
            <p className="font-bold mb-2 text-white">{labelStr} · 预测</p>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <span className="text-slate-400">预测收盘:</span>
              <span className="font-mono">{datum.predicted_close?.toFixed(2)}</span>
              <span className="text-slate-400">日变动:</span>
              <span className="font-mono">{formatPct(datum.change_pct || 0)}</span>
              {datum.bandLow !== undefined && datum.bandHigh !== undefined && (
                <>
                  <span className="text-slate-400">区间:</span>
                  <span className="font-mono">
                    {datum.bandLow.toFixed(2)} ~ {datum.bandHigh.toFixed(2)}
                  </span>
                </>
              )}
            </div>
          </div>
        );
      }

      return (
        <div className="bg-slate-800 border border-slate-600 p-3 rounded shadow-xl text-xs text-slate-200">
          <p className="font-bold mb-2 text-white">{labelStr} · 实际</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <span className="text-slate-400">Open:</span>
            <span className="font-mono">{datum.open.toFixed(2)}</span>
            <span className="text-slate-400">High:</span>
            <span className="font-mono">{datum.high.toFixed(2)}</span>
            <span className="text-slate-400">Low:</span>
            <span className="font-mono">{datum.low.toFixed(2)}</span>
            <span className="text-slate-400">Close:</span>
            <span
              className={`font-mono ${datum.close >= datum.open ? 'text-red-400' : 'text-emerald-400'}`}
            >
              {datum.close.toFixed(2)}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  const signalColor =
    data?.signal === '买入' ? '#ef4444' : data?.signal === '卖出' ? '#3b82f6' : '#eab308';

  return (
    <div className="space-y-6">
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <div className="text-lg font-semibold text-white">A股走势预测（T+5）</div>
          {data?.method && <span className="text-xs text-slate-400">{data.method}</span>}
        </div>

        <div className="flex gap-3 flex-col md:flex-row">
          <input
            className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-red-500"
            placeholder="输入股票代码（例如 600519）"
            value={code}
            onChange={(e) => setCode(e.target.value)}
          />
          <button
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              loading ? 'bg-slate-700 text-slate-300' : 'bg-red-600 text-white hover:bg-red-500'
            }`}
            disabled={loading}
            onClick={handlePredict}
          >
            {loading ? '预测中...' : '预测'}
          </button>
        </div>

        {error && <div className="text-red-400 mt-3 text-sm">{error}</div>}

        {data && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
            <div className="bg-slate-900 border border-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-400">预测信号</div>
              <div className="text-xl font-bold" style={{ color: signalColor }}>
                {data.signal}
              </div>
              <div className="text-slate-400 text-xs mt-1">置信度 {data.confidence.toFixed(4)}</div>
            </div>
            <div className="bg-slate-900 border border-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-400">最新收盘</div>
              <div className="text-xl font-bold text-white">
                {data.last_close.toFixed(2)} <span className="text-sm text-slate-400">({data.last_date})</span>
              </div>
            </div>
            <div className="bg-slate-900 border border-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-400">模型</div>
              <div className="text-lg font-semibold text-white">TFT 推理 / Baseline 回退</div>
              <div className="text-slate-400 text-xs mt-1">自动选择可用模型</div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg min-h-[360px]">
        <div className="flex items-center justify-between mb-4">
          <div className="text-white font-semibold">走势 + 5 日预测</div>
          {data?.forecast && (
            <div className="flex gap-3 text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
                实际上涨
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                实际下跌
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-white inline-block" />
                预测上涨
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
                预测下跌
              </span>
            </div>
          )}
        </div>

        {chartData.length > 0 ? (
          <div className="w-full h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                <Tooltip content={renderTooltip} cursor={{ fill: 'transparent', stroke: '#cbd5e1', strokeWidth: 1, strokeDasharray: '4 4' }} />

                <Bar dataKey="bodyRange" barSize={10} minPointSize={2}>
                  {chartData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.forecast ? 'transparent' : entry.color || '#64748b'} />
                  ))}
                </Bar>

                <Bar dataKey="forecastBody" barSize={10} minPointSize={2}>
                  {chartData.map((entry, idx) => (
                    <Cell key={`fc-${idx}`} fill={entry.forecast ? entry.color || '#e5e7eb' : 'transparent'} />
                  ))}
                </Bar>

                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="#f97316"
                  strokeWidth={2}
                  dot={false}
                  connectNulls={false}
                  name="历史收盘"
                />

                <Line
                  type="monotone"
                  dataKey="predicted_close"
                  stroke="#e5e7eb"
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                  name="预测收盘"
                />

                <Scatter data={chartData.filter((d) => !d.forecast)} shape="circle">
                  {chartData
                    .filter((d) => !d.forecast)
                    .map((entry, idx) => (
                      <Cell key={idx} fill={entry.color || '#64748b'} r={3} />
                    ))}
                </Scatter>

                <Scatter data={chartData.filter((d) => d.forecast)} shape="circle">
                  {chartData
                    .filter((d) => d.forecast)
                    .map((entry, idx) => (
                      <Cell key={idx} fill={entry.color || '#e5e7eb'} r={5} />
                    ))}
                </Scatter>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-slate-400">请输入股票代码并点击预测，系统将返回历史走势与未来 5 日预测。</div>
        )}
      </div>
    </div>
  );
};

export default StockForecastPage;
