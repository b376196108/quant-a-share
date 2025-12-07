import React from 'react';
import { IndustryData, MarketStatsData } from '../types';
import { TrendingUp, Thermometer } from 'lucide-react';

interface MarketStatsProps {
  industryData: IndustryData[];
  marketStats: MarketStatsData | null;
  loading?: boolean;
  error?: string | null;
}

const MarketStats: React.FC<MarketStatsProps> = ({
  industryData,
  marketStats,
  loading,
  error,
}) => {
  const getSentimentColor = (s: string) => {
    switch (s) {
      case 'High':
        return 'text-red-400 bg-red-400/10';
      case 'Medium':
        return 'text-yellow-400 bg-yellow-400/10';
      case 'Low':
        return 'text-emerald-400 bg-emerald-400/10';
      default:
        return 'text-slate-400';
    }
  };

  const getSentimentLabel = (s: string) => {
    switch (s) {
      case 'High':
        return '高涨';
      case 'Medium':
        return '中性';
      case 'Low':
        return '低迷';
      default:
        return s;
    }
  };

  // ---------- 状态处理：加载中 / 错误 / 无数据 ----------

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="text-blue-400" size={20} />
            行业板块涨跌 & 情绪
          </h3>
          <p className="text-slate-400 text-sm">正在加载行业数据...</p>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Thermometer className="text-orange-400" size={20} />
            市场全景统计
          </h3>
          <p className="text-slate-400 text-sm">正在加载市场统计...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 text-red-400 text-sm">
        获取市场统计失败：{error}
      </div>
    );
  }

  if (!marketStats) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 text-slate-400 text-sm">
        暂无市场统计数据。
      </div>
    );
  }

  const { limitDown, down, flat, up, limitUp, sentimentScore } = marketStats;

  // ---------- 正常渲染 ----------

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Industry Stats */}
      <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700">
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="text-blue-400" size={20} />
          行业板块涨跌 & 情绪
        </h3>
        <div className="space-y-3">
          {industryData.map((ind, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg hover:bg-slate-700/50 transition-colors"
            >
              <span className="text-slate-200 font-medium">{ind.name}</span>
              <div className="flex items-center gap-4">
                <span
                  className={`font-mono font-bold ${
                    ind.change >= 0 ? 'text-red-400' : 'text-emerald-400'
                  }`}
                >
                  {ind.change > 0 ? '+' : ''}
                  {ind.change}%
                </span>
                <span
                  className={`text-xs px-2 py-1 rounded ${getSentimentColor(ind.sentiment)}`}
                >
                  {getSentimentLabel(ind.sentiment)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Stock Stats & Market Sentiment */}
      <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 flex flex-col">
        <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
          <Thermometer className="text-orange-400" size={20} />
          市场全景统计
        </h3>

        {/* Individual Stock Distribution */}
        <div className="mb-8">
          <div className="flex justify-between text-xs text-slate-400 mb-2">
            <span>跌停 {limitDown}</span>
            <span>跌 {down}</span>
            <span>平 {flat}</span>
            <span>涨 {up}</span>
            <span>涨停 {limitUp}</span>
          </div>
          <div className="flex h-6 rounded-full overflow-hidden w-full">
            <div
              style={{ width: `${(limitDown / 5000) * 100}%` }}
              className="bg-emerald-700"
              title="跌停"
            />
            <div
              style={{ width: `${(down / 5000) * 100}%` }}
              className="bg-emerald-500"
              title="跌"
            />
            <div
              style={{ width: `${(flat / 5000) * 100}%` }}
              className="bg-slate-500"
              title="平"
            />
            <div
              style={{ width: `${(up / 5000) * 100}%` }}
              className="bg-red-500"
              title="涨"
            />
            <div
              style={{ width: `${(limitUp / 5000) * 100}%` }}
              className="bg-red-700"
              title="涨停"
            />
          </div>
        </div>

        {/* Market Sentiment Gauge */}
        <div className="flex-1 flex flex-col justify-center items-center">
          <div className="relative w-full h-4 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="absolute top-0 bottom-0 left-0 bg-gradient-to-r from-emerald-500 via-slate-400 to-red-500 transition-all duration-1000"
              style={{ width: `${sentimentScore}%` }}
            ></div>
          </div>
          <div className="flex justify-between w-full mt-2 text-sm font-bold">
            <span className="text-emerald-500">恐慌</span>
            <span className="text-white text-lg">{sentimentScore}</span>
            <span className="text-red-500">贪婪</span>
          </div>
          <p className="text-slate-400 text-xs mt-2">今日市场情绪指数</p>
        </div>
      </div>
    </div>
  );
};

export default MarketStats;
