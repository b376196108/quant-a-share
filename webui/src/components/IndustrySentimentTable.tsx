import React, { useEffect, useState } from 'react';
import type { IndustrySentiment } from '../types';

const IndustrySentimentTable: React.FC = () => {
  const [data, setData] = useState<IndustrySentiment[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/industry-sentiment?limit=20');
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

  const renderContent = () => {
    if (loading) {
      return (
        <div className="loading-container">
          <div className="spinner"></div>
          <div>正在加载行业数据...</div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="error-container">
          <p>{error}</p>
          <button className="retry-btn" onClick={fetchData}>重试</button>
        </div>
      );
    }

    if (data.length === 0) {
      return <div className="error-container">暂无行业数据</div>;
    }

    return (
      <div className="table-container">
        <table className="industry-table">
          <thead>
            <tr>
              <th>行业</th>
              <th>情绪</th>
              <th>涨幅 (Avg/Med)</th>
              <th>上涨占比</th>
              <th>成交额(亿)</th>
              <th>涨跌家数</th>
              <th>涨跌停</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => {
              const industry = asString(item['行业'] as string | number | undefined);
              const sentiment = asString(item['情绪'] as string | number | undefined);
              const avgChange = asNumber(item['平均涨幅(%)'] as number | string | undefined);
              const medianChange = asNumber(item['中位涨幅(%)'] as number | string | undefined);
              const upRatio = asNumber(item['上涨占比'] as number | string | undefined);
              const turnover = asNumber(item['总成交额(亿元)'] as number | string | undefined);
              const upCount = asNumber(item['上涨家数'] as number | string | undefined);
              const downCount = asNumber(item['下跌家数'] as number | string | undefined);
              const limitUp = asNumber(item['涨停家数'] as number | string | undefined);
              const limitDown = asNumber(item['跌停家数'] as number | string | undefined);
              const upRatioPct = (upRatio * 100).toFixed(0);

              return (
                <tr key={index}>
                  <td style={{ fontWeight: 500 }}>{industry}</td>
                  <td>
                    <span className={`sentiment-tag ${getSentimentClass(sentiment)}`}>
                      {sentiment}
                    </span>
                  </td>
                  <td>
                    <span className={avgChange >= 0 ? 'text-up' : 'text-down'}>
                      {avgChange.toFixed(2)}%
                    </span>
                    <span style={{ color: '#9ca3af', margin: '0 4px' }}>/</span>
                    <span className={medianChange >= 0 ? 'text-up' : 'text-down'}>
                      {medianChange.toFixed(2)}%
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div
                        style={{
                          width: '50px',
                          height: '4px',
                          backgroundColor: '#e5e7eb',
                          borderRadius: '2px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${upRatio * 100}%`,
                            height: '100%',
                            backgroundColor: 'var(--color-up)',
                          }}
                        ></div>
                      </div>
                      <span>{`${upRatioPct}%`}</span>
                    </div>
                  </td>
                  <td>{turnover.toLocaleString()}</td>
                  <td>
                    <span className="text-up">{upCount}</span>
                    <span style={{ margin: '0 2px' }}>:</span>
                    <span className="text-down">{downCount}</span>
                  </td>
                  <td>
                    <span className="text-up">{limitUp}</span>
                    <span style={{ margin: '0 2px' }}>/</span>
                    <span className="text-down">{limitDown}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="card">
      <h2 className="card-title">
        行业情绪列表
        <span style={{ fontSize: '0.85rem', fontWeight: 'normal', color: '#6b7280' }}>
          按情绪热度排序 (Top 20)
        </span>
      </h2>
      {renderContent()}
    </div>
  );
};

export default IndustrySentimentTable;
