import React, { useEffect, useState } from 'react';
import { IndustrySentiment } from '../types';

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
            {data.map((item, index) => (
              <tr key={index}>
                <td style={{ fontWeight: 500 }}>{item["行业"]}</td>
                <td>
                  <span className={`sentiment-tag ${getSentimentClass(item["情绪"])}`}>
                    {item["情绪"]}
                  </span>
                </td>
                <td>
                  <span className={item["平均涨幅(%)"] >= 0 ? 'text-up' : 'text-down'}>
                    {item["平均涨幅(%)"].toFixed(2)}%
                  </span>
                  <span style={{ color: '#9ca3af', margin: '0 4px' }}>/</span>
                  <span className={item["中位涨幅(%)"] >= 0 ? 'text-up' : 'text-down'}>
                    {item["中位涨幅(%)"].toFixed(2)}%
                  </span>
                </td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '50px', height: '4px', backgroundColor: '#e5e7eb', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{ width: `${item["上涨占比"] * 100}%`, height: '100%', backgroundColor: 'var(--color-up)' }}></div>
                    </div>
                    <span>{(item["上涨占比"] * 100).toFixed(0)}%</span>
                  </div>
                </td>
                <td>{item["总成交额(亿元)"].toLocaleString()}</td>
                <td>
                  <span className="text-up">{item["上涨家数"]}</span>
                  <span style={{ margin: '0 2px' }}>:</span>
                  <span className="text-down">{item["下跌家数"]}</span>
                </td>
                <td>
                   <span className="text-up">{item["涨停家数"]}</span>
                   <span style={{ margin: '0 2px' }}>/</span>
                   <span className="text-down">{item["跌停家数"]}</span>
                </td>
              </tr>
            ))}
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