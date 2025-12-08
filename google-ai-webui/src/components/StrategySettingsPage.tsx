import React, { useEffect, useState } from 'react';
import { Settings, Save, RefreshCw, Cpu, ToggleLeft, ToggleRight, CheckCircle, AlertCircle } from 'lucide-react';
import { StrategyConfig } from '../types';

const StrategySettingsPage: React.FC = () => {
  const [strategies, setStrategies] = useState<StrategyConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [toast, setToast] = useState<{msg: string, type: 'success' | 'error'} | null>(null);

  const fetchConfigs = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/strategies');
      if (!response.ok) throw new Error('Failed to load strategy configs');
      const data = await response.json();
      // Assume API returns { strategies: [...] } or just [...]
      setStrategies(Array.isArray(data) ? data : data.strategies || []);
    } catch (err: any) {
      setError('无法加载策略配置，请检查后端服务。');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  const handleParamChange = (strategyId: string, key: string, value: string | number) => {
    setStrategies(prev => prev.map(s => {
      if (s.id !== strategyId) return s;
      return {
        ...s,
        params: {
          ...s.params,
          [key]: value
        }
      };
    }));
  };

  const handleToggleEnabled = (strategyId: string) => {
    setStrategies(prev => prev.map(s => {
      if (s.id !== strategyId) return s;
      return { ...s, enabled: !s.enabled };
    }));
  };

  const handleSave = async (strategy: StrategyConfig) => {
    setSavingId(strategy.id);
    setToast(null);
    try {
      const response = await fetch(`http://localhost:8000/api/strategies/${strategy.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: strategy.enabled,
          params: strategy.params
        })
      });
      
      if (!response.ok) throw new Error('Update failed');
      
      setToast({ msg: `"${strategy.name}" 配置已保存`, type: 'success' });
      setTimeout(() => setToast(null), 3000);
    } catch (err) {
      setToast({ msg: '保存失败，请重试', type: 'error' });
    } finally {
      setSavingId(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500">
        <RefreshCw className="animate-spin mr-2" /> 加载配置中...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500">
        <AlertCircle size={48} className="mb-4 opacity-50" />
        <p>{error}</p>
        <button onClick={fetchConfigs} className="mt-4 px-4 py-2 bg-slate-800 rounded hover:bg-slate-700 text-slate-200">
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="p-6 bg-slate-950 text-slate-100 min-h-full overflow-y-auto">
      
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Settings className="text-blue-500" />
            系统策略参数设置
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            配置量化回测与实盘信号生成的底层参数
          </p>
        </div>
        {toast && (
          <div className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 animate-in fade-in slide-in-from-top-2 ${
            toast.type === 'success' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'
          }`}>
            {toast.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
            {toast.msg}
          </div>
        )}
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {strategies.map(strategy => (
          <div key={strategy.id} className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col hover:border-slate-700 transition-all">
            
            {/* Card Header */}
            <div className="flex justify-between items-start mb-4 pb-4 border-b border-slate-800">
              <div className="flex gap-3">
                <div className={`p-2 rounded-lg h-fit ${strategy.enabled ? 'bg-blue-500/10 text-blue-400' : 'bg-slate-800 text-slate-500'}`}>
                  <Cpu size={20} />
                </div>
                <div>
                  <h3 className="font-bold text-lg text-white">{strategy.name}</h3>
                  <code className="text-xs text-slate-500 bg-slate-950 px-1.5 py-0.5 rounded">{strategy.id}</code>
                </div>
              </div>
              <button 
                onClick={() => handleToggleEnabled(strategy.id)}
                className={`transition-colors ${strategy.enabled ? 'text-emerald-400' : 'text-slate-600'}`}
                title={strategy.enabled ? "已启用" : "已禁用"}
              >
                {strategy.enabled ? <ToggleRight size={32} /> : <ToggleLeft size={32} />}
              </button>
            </div>

            {/* Description */}
            <p className="text-xs text-slate-400 mb-6 min-h-[32px]">
              {strategy.description || "暂无描述"}
            </p>

            {/* Parameters Form */}
            <div className="space-y-4 flex-1">
              {strategy.schema.map(field => (
                <div key={field.key}>
                  <label className="block text-xs font-medium text-slate-500 mb-1.5 uppercase tracking-wide">
                    {field.label}
                  </label>
                  {field.type === 'int' || field.type === 'float' ? (
                    <input 
                      type="number"
                      value={strategy.params[field.key] || ''}
                      onChange={(e) => handleParamChange(strategy.id, field.key, field.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                      min={field.min}
                      max={field.max}
                      step={field.step || (field.type === 'int' ? 1 : 0.1)}
                      className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-blue-500 focus:outline-none transition-colors font-mono"
                    />
                  ) : field.type === 'select' && field.options ? (
                    <select
                      value={strategy.params[field.key] || ''}
                      onChange={(e) => handleParamChange(strategy.id, field.key, e.target.value)}
                      className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-blue-500 focus:outline-none"
                    >
                      {field.options.map(opt => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  ) : (
                    <input 
                      type="text"
                      value={strategy.params[field.key] || ''}
                      onChange={(e) => handleParamChange(strategy.id, field.key, e.target.value)}
                      className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-blue-500 focus:outline-none"
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Card Footer */}
            <div className="mt-6 pt-4 border-t border-slate-800">
              <button 
                onClick={() => handleSave(strategy)}
                disabled={savingId === strategy.id}
                className="w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
              >
                {savingId === strategy.id ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" /> 保存中...
                  </>
                ) : (
                  <>
                    <Save size={16} /> 保存配置
                  </>
                )}
              </button>
            </div>

          </div>
        ))}
      </div>
    </div>
  );
};

export default StrategySettingsPage;
