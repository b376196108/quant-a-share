import React, { useEffect, useState } from 'react';
import type { MarketOverview } from '../types';

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
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '获取数据失败';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const asNumber = (value: string | number | undefined, fallback = 0): number => {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  };

  const asString = (value: string | number | undefined): string => {
    if (value === undefined) return '';
    return String(value);
  };

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

  const tradeDate = asString(data['交易日'] as string | number | undefined);
  const sentiment = asString(data['市场情绪'] as string | number | undefined);
  const upRatio = asNumber(data['上涨占比'] as number | string | undefined);
  const turnover = asNumber(data['总成交额(亿元)'] as number | string | undefined);
  const totalStocks = asNumber(data['总股票数'] as number | string | undefined);
  const limitUp = asNumber(data['涨停家数'] as number | string | undefined);
  const limitDown = asNumber(data['跌停家数'] as number | string | undefined);
  const upCount = asNumber(data['上涨家数'] as number | string | undefined);
  const downCount = asNumber(data['下跌家数'] as number | string | undefined);
  const avgChange = asNumber(data['平均涨幅(%)'] as number | string | undefined);
  const medianChange = asNumber(data['中位数涨幅(%)'] as number | string | undefined);

  return (
    <div className="card">
      <h2 className="card-title">
        市场整体情绪概览
        <span style={{ fontSize: '0.8rem', fontWeight: 'normal', color: '#6b7280' }}>
          {tradeDate}
        </span>
      </h2>

      <div className="dashboard-grid-inner" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Sentiment Banner */}
        <div style={{ textAlign: 'center', padding: '10px 0' }}>
          <div className="metric-label">当前市场情绪</div>
          <div className={`sentiment-tag ${getSentimentClass(sentiment)}`} style={{ fontSize: '1.2rem', padding: '6px 20px', marginTop: '5px' }}>
            {sentiment}
          </div>
        </div>

        {/* Up/Down Bar */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.9rem' }}>
            <span className="text-up">上涨: {upCount}</span>
            <span className="text-down">下跌: {downCount}</span>
          </div>
          <div className="progress-bar-container">
            <div 
              className="progress-bar-fill" 
              style={{ width: `${(upRatio * 100).toFixed(1)}%` }}
            ></div>
          </div>
          <div style={{ textAlign: 'right', fontSize: '0.8rem', color: '#6b7280', marginTop: '2px' }}>
            上涨占比: {(upRatio * 100).toFixed(1)}%
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="metric-grid">
          <div className="metric-item">
            <span className="metric-label">总成交额</span>
            <span className="metric-value">{turnover.toLocaleString()} 亿</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">总股票数</span>
            <span className="metric-value">{totalStocks}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">涨停 / 跌停</span>
            <span className="metric-value">
              <span className="text-up">{limitUp}</span> / <span className="text-down">{limitDown}</span>
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">平均 / 中位涨幅</span>
            <span className="metric-value">
              <span className={avgChange >= 0 ? 'text-up' : 'text-down'}>{avgChange}%</span>
              {' / '}
              <span className={medianChange >= 0 ? 'text-up' : 'text-down'}>{medianChange}%</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketOverviewCard;
