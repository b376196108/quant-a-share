import React from 'react';
import { IndustryData, MarketStatsData } from '../types';
import { TrendingUp, TrendingDown, Minus, Thermometer } from 'lucide-react';

interface MarketStatsProps {
  industryData: IndustryData[];
  marketStats: MarketStatsData;
}

const MarketStats: React.FC<MarketStatsProps> = ({ industryData, marketStats }) => {
  const getSentimentColor = (s: string) => {
    switch (s) {
      case 'High': return 'text-red-400 bg-red-400/10';
      case 'Medium': return 'text-yellow-400 bg-yellow-400/10';
      case 'Low': return 'text-emerald-400 bg-emerald-400/10';
      default: return 'text-slate-400';
    }
  };

  const getSentimentLabel = (s: string) => {
    switch (s) {
      case 'High': return '高涨';
      case 'Medium': return '中性';
      case 'Low': return '低迷';
      default: return s;
    }
  };

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
            <div key={idx} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg hover:bg-slate-700/50 transition-colors">
              <span className="text-slate-200 font-medium">{ind.name}</span>
              <div className="flex items-center gap-4">
                <span className={`font-mono font-bold ${ind.change >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                  {ind.change > 0 ? '+' : ''}{ind.change}%
                </span>
                <span className={`text-xs px-2 py-1 rounded ${getSentimentColor(ind.sentiment)}`}>
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
            <span>跌停 {marketStats.limitDown}</span>
            <span>跌 {marketStats.down}</span>
            <span>平 {marketStats.flat}</span>
            <span>涨 {marketStats.up}</span>
            <span>涨停 {marketStats.limitUp}</span>
          </div>
          <div className="flex h-6 rounded-full overflow-hidden w-full">
            <div style={{ width: `${(marketStats.limitDown / 5000) * 100}%` }} className="bg-emerald-700" title="跌停" />
            <div style={{ width: `${(marketStats.down / 5000) * 100}%` }} className="bg-emerald-500" title="跌" />
            <div style={{ width: `${(marketStats.flat / 5000) * 100}%` }} className="bg-slate-500" title="平" />
            <div style={{ width: `${(marketStats.up / 5000) * 100}%` }} className="bg-red-500" title="涨" />
            <div style={{ width: `${(marketStats.limitUp / 5000) * 100}%` }} className="bg-red-700" title="涨停" />
          </div>
        </div>

        {/* Market Sentiment Gauge */}
        <div className="flex-1 flex flex-col justify-center items-center">
          <div className="relative w-full h-4 bg-slate-700 rounded-full overflow-hidden">
             <div 
               className="absolute top-0 bottom-0 left-0 bg-gradient-to-r from-emerald-500 via-slate-400 to-red-500 transition-all duration-1000"
               style={{ width: `${marketStats.sentimentScore}%` }}
             ></div>
          </div>
          <div className="flex justify-between w-full mt-2 text-sm font-bold">
             <span className="text-emerald-500">恐慌</span>
             <span className="text-white text-lg">{marketStats.sentimentScore}</span>
             <span className="text-red-500">贪婪</span>
          </div>
          <p className="text-slate-400 text-xs mt-2">今日市场情绪指数</p>
        </div>

      </div>
    </div>
  );
};

export default MarketStats;