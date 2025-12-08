
import React, { useState, useEffect } from 'react';
import { Newspaper, RefreshCw, Zap, ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { NewsItem } from '../types';
import { generateDailyBriefing } from '../services/geminiService';

// Mock Data for News
const MOCK_NEWS: NewsItem[] = [
  {
    id: '1',
    title: '央行：加大稳健货币政策实施力度，保持流动性合理充裕',
    summary: '中国人民银行发布公告，将通过公开市场操作等多种货币政策工具，确保银行体系流动性保持合理充裕，支持实体经济发展。',
    source: '央行官网',
    time: '10:30',
    sentiment: 'Bullish',
    tags: ['宏观', '货币政策']
  },
  {
    id: '2',
    title: '宁德时代发布新一代麒麟电池，续航突破1000公里',
    summary: '全球电池龙头宁德时代今日发布第三代CTP技术——麒麟电池，体积利用率突破72%，能量密度达255Wh/kg。',
    source: '证券时报',
    time: '09:15',
    sentiment: 'Bullish',
    tags: ['新能源', '电池']
  },
  {
    id: '3',
    title: '北向资金早盘净流出超30亿元，白酒板块承压',
    summary: '受外围市场波动影响，北向资金今日早盘呈现净流出态势，贵州茅台、五粮液等白酒权重股跌幅居前。',
    source: '东方财富',
    time: '11:00',
    sentiment: 'Bearish',
    tags: ['资金流向', '白酒']
  },
  {
    id: '4',
    title: '半导体行业库存去化接近尾声，AI芯片需求爆发',
    summary: '多家券商研报指出，半导体行业周期底部已现，随着AI大模型训练需求激增，算力芯片供不应求。',
    source: '财联社',
    time: '08:45',
    sentiment: 'Bullish',
    tags: ['半导体', 'AI']
  },
  {
    id: '5',
    title: '多家上市房企发布业绩预警，房地产板块震荡整理',
    summary: '受限于销售回款放缓，多家头部房企上半年业绩预减，板块整体呈现横盘震荡走势。',
    source: '每日经济新闻',
    time: '13:20',
    sentiment: 'Neutral',
    tags: ['房地产', '财报']
  },
  {
    id: '6',
    title: '证监会：深化资本市场改革，支持科技企业上市融资',
    summary: '证监会主席在论坛上表示，将进一步完善科创板制度，畅通科技型企业上市融资渠道。',
    source: '新华社',
    time: '14:10',
    sentiment: 'Bullish',
    tags: ['政策', '科创板']
  }
];

const NewsPage: React.FC = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [aiBriefing, setAiBriefing] = useState<string | null>(null);
  const [generatingBriefing, setGeneratingBriefing] = useState(false);

  useEffect(() => {
    // Simulate API fetch
    setTimeout(() => {
      setNews(MOCK_NEWS);
      setLoading(false);
    }, 800);
  }, []);

  const handleGenerateBriefing = async () => {
    if (news.length === 0) return;
    setGeneratingBriefing(true);
    const briefing = await generateDailyBriefing(news);
    setAiBriefing(briefing);
    setGeneratingBriefing(false);
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'Bullish': return <TrendingUp size={16} className="text-red-500" />; // China Red = Up
      case 'Bearish': return <TrendingDown size={16} className="text-emerald-500" />; // China Green = Down
      default: return <Minus size={16} className="text-slate-400" />;
    }
  };

  const getSentimentClass = (sentiment: string) => {
    switch (sentiment) {
      case 'Bullish': return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'Bearish': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      default: return 'bg-slate-700/50 text-slate-400 border-slate-600';
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 p-6 overflow-y-auto font-sans">
      
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Newspaper className="text-orange-500" />
            财经新闻快讯 (Financial News)
          </h1>
          <p className="text-slate-400 text-sm mt-1">7x24小时 A股市场动态与深度解读</p>
        </div>
        <button 
          onClick={() => { setLoading(true); setTimeout(() => setLoading(false), 800); }}
          className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw size={20} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* Left: News Feed */}
        <div className="xl:col-span-2 space-y-4">
          {loading ? (
             <div className="flex flex-col items-center justify-center h-64 text-slate-500">
               <RefreshCw size={32} className="animate-spin mb-4" />
               <p>正在刷新资讯...</p>
             </div>
          ) : (
            news.map((item) => (
              <div key={item.id} className="bg-slate-900 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-all group">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex gap-2 items-center">
                    <span className={`px-2 py-0.5 text-xs font-bold rounded border flex items-center gap-1 ${getSentimentClass(item.sentiment)}`}>
                      {getSentimentIcon(item.sentiment)}
                      {item.sentiment === 'Bullish' ? '利好' : item.sentiment === 'Bearish' ? '利空' : '中性'}
                    </span>
                    {item.tags.map(tag => (
                      <span key={tag} className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded">#{tag}</span>
                    ))}
                  </div>
                  <span className="text-xs text-slate-500 font-mono">{item.time}</span>
                </div>
                
                <h3 className="text-lg font-bold text-slate-200 mb-2 group-hover:text-blue-400 transition-colors cursor-pointer">
                  {item.title}
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed mb-3">
                  {item.summary}
                </p>
                
                <div className="flex justify-between items-center text-xs text-slate-500 border-t border-slate-800 pt-3 mt-3">
                  <span>来源: {item.source}</span>
                  <button className="flex items-center gap-1 hover:text-blue-400 transition-colors">
                    查看原文 <ExternalLink size={12} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Right: AI Analysis & Trending */}
        <div className="flex flex-col gap-6">
          
          {/* AI Briefing Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Zap size={120} />
            </div>
            
            <div className="relative z-10">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Zap className="text-yellow-400 fill-yellow-400" size={20} />
                AI 市场早报
              </h3>
              
              {!aiBriefing ? (
                <div className="text-center py-8">
                  <p className="text-sm text-slate-400 mb-4">
                    使用 Gemini 模型基于今日新闻生成市场核心观点摘要。
                  </p>
                  <button 
                    onClick={handleGenerateBriefing}
                    disabled={generatingBriefing}
                    className="bg-yellow-600 hover:bg-yellow-500 text-white px-4 py-2 rounded-lg text-sm font-bold shadow-lg shadow-yellow-900/20 transition-all flex items-center justify-center gap-2 w-full"
                  >
                    {generatingBriefing ? <RefreshCw className="animate-spin" size={16}/> : <Zap size={16} />}
                    {generatingBriefing ? '正在分析...' : '生成今日简报'}
                  </button>
                </div>
              ) : (
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                  <div className="prose prose-invert prose-sm max-w-none">
                    <div className="text-slate-300 text-sm whitespace-pre-line leading-relaxed">
                      {aiBriefing}
                    </div>
                  </div>
                  <button 
                    onClick={() => setAiBriefing(null)}
                    className="mt-4 text-xs text-slate-500 hover:text-white underline w-full text-center"
                  >
                    重新生成
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Trending Topics */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">
              热门话题 (Trending)
            </h3>
            <div className="flex flex-wrap gap-2">
              {['#中特估', '#人工智能', '#半导体复苏', '#美联储加息', '#人民币汇率', '#新能源出海'].map((tag, idx) => (
                <span 
                  key={idx} 
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs rounded-full cursor-pointer transition-colors border border-slate-700"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default NewsPage;
