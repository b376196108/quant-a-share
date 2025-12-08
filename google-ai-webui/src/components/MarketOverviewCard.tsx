import React, { useEffect, useState } from 'react';
import { MarketOverview } from '../types';

const MarketOverviewCard: React.FC = () => {
  const [data, setData] = useState<MarketOverview | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/overview');
      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }
      const jsonData = await response.json();
      setData(jsonData);
    } catch (err: any) {
      setError(err.message || '获取数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getSentimentClass = (sentiment: string) => {
    if (sentiment.includes('高潮') || sentiment.includes('热')) return 'sentiment-hot';
    if (sentiment.includes('冰点') || sentiment.includes('冷')) return 'sentiment-cold';
    return 'sentiment-normal';
  };

  if (loading) {
    return (
      <div className="card">
        <h2 className="card-title">市场整体情绪概览</h2>
        <div className="loading-container">
          <div className="spinner"></div>
          <div>加载数据中...</div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="card">
        <h2 className="card-title">市场整体情绪概览</h2>
        <div className="error-container">
          <p>{error || '暂无数据'}</p>
          <button className="retry-btn" onClick={fetchData}>重试</button>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="card-title">
        市场整体情绪概览
        <span style={{ fontSize: '0.8rem', fontWeight: 'normal', color: '#6b7280' }}>
          {data["交易日"]}
        </span>
      </h2>

      <div className="dashboard-grid-inner" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Sentiment Banner */}
        <div style={{ textAlign: 'center', padding: '10px 0' }}>
          <div className="metric-label">当前市场情绪</div>
          <div className={`sentiment-tag ${getSentimentClass(data["市场情绪"])}`} style={{ fontSize: '1.2rem', padding: '6px 20px', marginTop: '5px' }}>
            {data["市场情绪"]}
          </div>
        </div>

        {/* Up/Down Bar */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.9rem' }}>
            <span className="text-up">上涨: {data["上涨家数"]}</span>
            <span className="text-down">下跌: {data["下跌家数"]}</span>
          </div>
          <div className="progress-bar-container">
            <div 
              className="progress-bar-fill" 
              style={{ width: `${(data["上涨占比"] * 100).toFixed(1)}%` }}
            ></div>
          </div>
          <div style={{ textAlign: 'right', fontSize: '0.8rem', color: '#6b7280', marginTop: '2px' }}>
            上涨占比: {(data["上涨占比"] * 100).toFixed(1)}%
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="metric-grid">
          <div className="metric-item">
            <span className="metric-label">总成交额</span>
            <span className="metric-value">{data["总成交额(亿元)"].toLocaleString()} 亿</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">总股票数</span>
            <span className="metric-value">{data["总股票数"]}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">涨停 / 跌停</span>
            <span className="metric-value">
              <span className="text-up">{data["涨停家数"]}</span> / <span className="text-down">{data["跌停家数"]}</span>
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">平均 / 中位涨幅</span>
            <span className="metric-value">
              <span className={data["平均涨幅(%)"] >= 0 ? 'text-up' : 'text-down'}>{data["平均涨幅(%)"]}%</span>
              {' / '}
              <span className={data["中位数涨幅(%)"] >= 0 ? 'text-up' : 'text-down'}>{data["中位数涨幅(%)"]}%</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketOverviewCard;