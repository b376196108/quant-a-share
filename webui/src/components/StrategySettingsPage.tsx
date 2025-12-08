import React, { useEffect, useState, useMemo } from 'react';
import { 
  Save, 
  RefreshCw, 
  Cpu, 
  Search, 
  CheckCircle, 
  AlertCircle, 
  Sliders,
  Power,
  RotateCcw,
  LayoutGrid
} from 'lucide-react';
import type { StrategyConfig, StrategyParamSchema } from '../types';

// Extended type locally for UI purposes (mocking categories)
interface ExtendedStrategyConfig extends StrategyConfig {
  category?: string;
  isDirty?: boolean;
}

const StrategySettingsPage: React.FC = () => {
  const [strategies, setStrategies] = useState<ExtendedStrategyConfig[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'enabled' | 'disabled'>('all');
  const [toast, setToast] = useState<{msg: string, type: 'success' | 'error'} | null>(null);

  // Mock Data Loading
  const fetchConfigs = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 600));
      
      const mockStrategies: ExtendedStrategyConfig[] = [
        {
          id: 'trend_follow_v1',
          name: '趋势跟随策略 V1',
          category: '趋势类 (Trend)',
          enabled: true,
          description: '基于双均线交叉的基础趋势跟踪策略，捕捉日线级别大趋势。适合波动率较大的单边行情。',
          params: { fast_window: 5, slow_window: 20, stop_loss: 0.05, position_size: 1000 },
          schema: [
            { key: 'fast_window', label: '快线周期 (Fast MA)', type: 'int', min: 2, max: 50 },
            { key: 'slow_window', label: '慢线周期 (Slow MA)', type: 'int', min: 10, max: 200 },
            { key: 'stop_loss', label: '止损比例 (Stop Loss %)', type: 'float', min: 0.01, max: 0.2, step: 0.01 },
            { key: 'position_size', label: '单笔开仓股数', type: 'int', min: 100 }
          ]
        },
        {
          id: 'trend_bollinger_break',
          name: '布林带突破策略',
          category: '趋势类 (Trend)',
          enabled: false,
          description: '当价格突破布林带上轨且伴随成交量放大时买入。',
          params: { period: 20, std_dev: 2.0, volume_factor: 1.5 },
          schema: [
            { key: 'period', label: '布林带周期', type: 'int', min: 5 },
            { key: 'std_dev', label: '标准差倍数', type: 'float', min: 1.0, max: 4.0 },
            { key: 'volume_factor', label: '成交量放大倍数', type: 'float', min: 1.0 }
          ]
        },
        {
          id: 'mean_reversion_rsi',
          name: 'RSI 均值回归',
          category: '震荡类 (Reversion)',
          enabled: false,
          description: '基于RSI指标的震荡策略：超卖区域买入，超买区域卖出。适合箱体震荡行情。',
          params: { rsi_period: 14, overbought: 70, oversold: 30 },
          schema: [
            { key: 'rsi_period', label: 'RSI 计算周期', type: 'int', min: 2 },
            { key: 'overbought', label: '超买阈值 (Overbought)', type: 'int', min: 50, max: 100 },
            { key: 'oversold', label: '超卖阈值 (Oversold)', type: 'int', min: 0, max: 50 }
          ]
        },
        {
          id: 'alpha_multi_factor',
          name: '多因子 Alpha 选股',
          category: 'Alpha 类 (Alpha)',
          enabled: true,
          description: '结合估值(PE/PB)、成长(G)和动量因子进行综合打分的选股策略。',
          params: { top_k: 10, rebalance_days: 20, market_cap_threshold: 50 },
          schema: [
            { key: 'top_k', label: '持仓股票数量', type: 'int', min: 1, max: 50 },
            { key: 'rebalance_days', label: '调仓周期 (天)', type: 'int', min: 1 },
            { key: 'market_cap_threshold', label: '最小市值过滤 (亿)', type: 'int', min: 10 }
          ]
        },
        {
          id: 'grid_trading_basic',
          name: '网格交易 V2',
          category: '高频/做市 (HFT)',
          enabled: false,
          description: '在固定价格区间内高抛低吸，利用市场波动赚取差价。',
          params: { grid_levels: 10, lower_price: 100, upper_price: 150 },
          schema: [
            { key: 'grid_levels', label: '网格层数', type: 'int', min: 2 },
            { key: 'lower_price', label: '网格下限', type: 'float' },
            { key: 'upper_price', label: '网格上限', type: 'float' }
          ]
        }
      ];
      setStrategies(mockStrategies);
      if (mockStrategies.length > 0) {
        setSelectedId(mockStrategies[0].id);
      }
    } catch (err: any) {
      setError('无法加载策略配置，请检查后端服务。');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  // Filter Logic
  const filteredStrategies = useMemo(() => {
    return strategies.filter(s => {
      const matchesSearch = s.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                            s.id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFilter = filterType === 'all' 
                            ? true 
                            : filterType === 'enabled' ? s.enabled : !s.enabled;
      return matchesSearch && matchesFilter;
    });
  }, [strategies, searchTerm, filterType]);

  // Group by Category
  const groupedStrategies = useMemo(() => {
    const groups: Record<string, ExtendedStrategyConfig[]> = {};
    filteredStrategies.forEach(s => {
      const cat = s.category || '其他 (General)';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(s);
    });
    return groups;
  }, [filteredStrategies]);

  const selectedStrategy = useMemo(() => 
    strategies.find(s => s.id === selectedId), 
  [strategies, selectedId]);

  // Handlers
  const handleParamChange = (key: string, value: string | number) => {
    if (!selectedId) return;
    setStrategies(prev => prev.map(s => {
      if (s.id !== selectedId) return s;
      return {
        ...s,
        isDirty: true,
        params: { ...s.params, [key]: value }
      };
    }));
  };

  const handleToggleEnabled = () => {
    if (!selectedId) return;
    setStrategies(prev => prev.map(s => {
      if (s.id !== selectedId) return s;
      return { ...s, isDirty: true, enabled: !s.enabled };
    }));
  };

  const handleSave = async () => {
    if (!selectedStrategy) return;
    setSaving(true);
    setToast(null);
    try {
      // Simulate API
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Clear dirty flag
      setStrategies(prev => prev.map(s => 
        s.id === selectedId ? { ...s, isDirty: false } : s
      ));
      
      setToast({ msg: '配置已保存成功', type: 'success' });
      setTimeout(() => setToast(null), 3000);
    } catch (err) {
      setToast({ msg: '保存失败', type: 'error' });
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    // In a real app, this might fetch the original config from server again
    // For now, we just mock a toast
    setToast({ msg: '已重置为上次保存的状态', type: 'success' });
    setTimeout(() => setToast(null), 3000);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 bg-slate-950">
        <RefreshCw className="animate-spin mr-2" /> 正在加载策略库...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 bg-slate-950">
        <AlertCircle size={48} className="mb-4 opacity-50" />
        <p>{error}</p>
        <button onClick={fetchConfigs} className="mt-4 px-4 py-2 bg-slate-800 rounded hover:bg-slate-700 text-slate-200">
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full bg-slate-950 overflow-hidden font-sans">
      
      {/* 1. Left Sidebar: Strategy List */}
      <div className="w-80 flex flex-col border-r border-slate-800 bg-slate-950 shrink-0">
        {/* Sidebar Header */}
        <div className="p-4 border-b border-slate-800">
          <div className="flex items-center gap-2 mb-4 text-white font-bold">
            <LayoutGrid size={20} className="text-blue-500" />
            <span>策略库</span>
            <span className="text-xs font-normal text-slate-500 ml-auto">{strategies.length} total</span>
          </div>
          
          {/* Search */}
          <div className="relative mb-3">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={14} />
            <input 
              type="text" 
              placeholder="搜索策略名称或ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-xs text-white focus:border-blue-500 outline-none"
            />
          </div>

          {/* Filters */}
          <div className="flex bg-slate-900 p-1 rounded-lg">
            {(['all', 'enabled', 'disabled'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={`flex-1 py-1 text-[10px] font-medium rounded transition-all uppercase tracking-wider ${
                  filterType === t 
                    ? 'bg-slate-700 text-white shadow' 
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {t === 'all' ? '全部' : t === 'enabled' ? '运行中' : '已停用'}
              </button>
            ))}
          </div>
        </div>

        {/* List Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {Object.entries(groupedStrategies).map(([category, items]) => (
            <div key={category}>
              <div className="px-4 py-2 bg-slate-900/50 text-[10px] font-bold text-slate-500 uppercase tracking-wider sticky top-0 backdrop-blur-sm border-y border-slate-800/50">
                {category}
              </div>
              <div>
                {items.map(s => (
                  <div 
                    key={s.id}
                    onClick={() => setSelectedId(s.id)}
                    className={`px-4 py-3 cursor-pointer border-l-2 transition-all hover:bg-slate-800/50 ${
                      selectedId === s.id 
                        ? 'bg-blue-500/5 border-blue-500' 
                        : 'border-transparent'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-1">
                      <span className={`text-sm font-medium line-clamp-1 ${selectedId === s.id ? 'text-blue-400' : 'text-slate-200'}`}>
                        {s.name}
                      </span>
                      {s.isDirty && <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 mt-1.5" title="Unsaved changes" />}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full ${s.enabled ? 'bg-emerald-500' : 'bg-slate-600'}`} />
                      <span className="text-xs text-slate-500 font-mono truncate">{s.id}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          
          {Object.keys(groupedStrategies).length === 0 && (
             <div className="p-8 text-center text-slate-500 text-xs">
               未找到匹配的策略
             </div>
          )}
        </div>
      </div>

      {/* 2. Right Main Content: Detail View */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-900">
        
        {selectedStrategy ? (
          <>
            {/* Detail Header */}
            <div className="px-8 py-6 border-b border-slate-800 bg-slate-950 flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-400 border border-slate-700 font-mono">
                    {selectedStrategy.category || 'General'}
                  </span>
                  {selectedStrategy.isDirty && (
                    <span className="px-2 py-0.5 rounded text-[10px] bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 flex items-center gap-1">
                      <AlertCircle size={10} /> 未保存
                    </span>
                  )}
                </div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                  {selectedStrategy.name}
                </h1>
                <p className="text-slate-400 text-sm mt-2 max-w-2xl leading-relaxed">
                  {selectedStrategy.description}
                </p>
              </div>

              <div className="flex flex-col items-end gap-3">
                 <button 
                  onClick={handleToggleEnabled}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    selectedStrategy.enabled 
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20' 
                      : 'bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700'
                  }`}
                 >
                   <Power size={18} />
                   {selectedStrategy.enabled ? '策略运行中' : '策略已停止'}
                 </button>
                 <span className="text-xs text-slate-600 font-mono">ID: {selectedStrategy.id}</span>
              </div>
            </div>

            {/* Detail Form Content */}
            <div className="flex-1 overflow-y-auto p-8">
              <div className="max-w-4xl mx-auto">
                
                <div className="flex items-center gap-2 mb-6 text-slate-200 font-bold border-b border-slate-800 pb-2">
                  <Sliders size={18} className="text-blue-500" />
                  <h2>核心参数配置 (Core Parameters)</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                  {(selectedStrategy.schema as StrategyParamSchema[]).map(field => (
                    <div key={field.key} className="group">
                      <label className="block text-xs font-medium text-slate-400 mb-2 uppercase tracking-wide group-hover:text-blue-400 transition-colors">
                        {field.label} 
                        {/* Show raw key name on hover for developers */}
                        <span className="ml-2 text-[10px] text-slate-600 font-mono font-normal opacity-0 group-hover:opacity-100 transition-opacity">
                          ({field.key})
                        </span>
                      </label>
                      
                      <div className="relative">
                        {field.type === 'select' && field.options ? (
                           <select
                            value={selectedStrategy.params[field.key] || ''}
                            onChange={(e) => handleParamChange(field.key, e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-sm text-white focus:border-blue-500 focus:outline-none transition-all appearance-none"
                          >
                            {field.options.map(opt => (
                              <option key={opt.value} value={opt.value}>{opt.label}</option>
                            ))}
                          </select>
                        ) : (
                          <input 
                            type={field.type === 'int' || field.type === 'float' ? 'number' : 'text'}
                            value={selectedStrategy.params[field.key] || ''}
                            onChange={(e) => handleParamChange(field.key, field.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                            step={field.step || (field.type === 'int' ? 1 : 0.01)}
                            min={field.min}
                            max={field.max}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-sm text-white focus:border-blue-500 focus:outline-none transition-all font-mono"
                          />
                        )}
                        {/* Decorator line on focus */}
                        <div className="absolute bottom-0 left-0 h-0.5 bg-blue-500 w-0 transition-all duration-300 group-focus-within:w-full"/>
                      </div>
                      
                      {/* Helper text based on min/max */}
                      {(field.min !== undefined || field.max !== undefined) && (
                         <div className="mt-1 text-[10px] text-slate-600 text-right">
                           {field.min !== undefined && `Min: ${field.min}`}
                           {field.max !== undefined && ` Max: ${field.max}`}
                         </div>
                      )}
                    </div>
                  ))}
                </div>

                <div className="mt-12 bg-blue-900/10 border border-blue-900/30 rounded-lg p-4 flex gap-3 text-sm text-blue-200/80">
                  <div className="shrink-0 mt-0.5"><CheckCircle size={16} /></div>
                  <p>
                    修改参数后，建议在 <strong>“策略回测”</strong> 模块使用新参数进行历史数据验证，以确保策略逻辑符合预期风险偏好。
                  </p>
                </div>

              </div>
            </div>

            {/* Action Footer */}
            <div className="p-4 border-t border-slate-800 bg-slate-950 flex justify-between items-center z-10">
               <div className="flex items-center">
                  {toast && (
                    <div className={`text-sm flex items-center gap-2 animate-in slide-in-from-bottom-2 fade-in ${toast.type === 'success' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {toast.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                      {toast.msg}
                    </div>
                  )}
               </div>
               <div className="flex gap-4">
                 <button 
                  onClick={handleReset}
                  className="px-6 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors flex items-center gap-2"
                 >
                   <RotateCcw size={16} /> 重置
                 </button>
                 <button 
                  onClick={handleSave}
                  disabled={saving || !selectedStrategy.isDirty}
                  className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white px-8 py-2.5 rounded-lg text-sm font-bold shadow-lg shadow-blue-900/20 transition-all flex items-center gap-2"
                 >
                   {saving ? <RefreshCw size={16} className="animate-spin"/> : <Save size={16} />}
                   {saving ? '保存中...' : '保存更改'}
                 </button>
               </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
             <Cpu size={64} className="mb-4 opacity-20" />
             <p className="text-lg font-medium">请从左侧列表选择一个策略</p>
             <p className="text-sm">Select a strategy to configure parameters</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default StrategySettingsPage;
