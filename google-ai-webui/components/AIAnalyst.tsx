import React, { useState, useRef, useEffect } from 'react';
import { AIChatMessage, Position, Strategy } from '../types';
import { analyzePortfolio, chatWithQuantAI } from '../services/geminiService';
import { Send, Bot, Sparkles, X, User } from 'lucide-react';

interface AIAnalystProps {
  positions: Position[];
  strategies: Strategy[];
}

const AIAnalyst: React.FC<AIAnalystProps> = ({ positions, strategies }) => {
  const [messages, setMessages] = useState<AIChatMessage[]>([
    {
      id: 'welcome',
      role: 'model',
      text: '你好，我是 QuantMind 智能助手。我可以为您分析A股持仓风险、解释量化因子，或提供Python策略代码建议。今天有什么可以帮您？',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMsg: AIChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsThinking(true);

    // Prepare history for Gemini
    const history = messages.map(m => ({
      role: m.role === 'model' ? 'model' : 'user',
      parts: [{ text: m.text }]
    }));

    const responseText = await chatWithQuantAI(history, userMsg.text);

    const modelMsg: AIChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'model',
      text: responseText,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, modelMsg]);
    setIsThinking(false);
  };

  const handleQuickAnalysis = async () => {
    setIsThinking(true);
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role: 'user',
      text: '请对我的A股持仓和当前运行策略进行一次全面的风险诊断。',
      timestamp: new Date()
    }]);

    const analysis = await analyzePortfolio(positions, strategies);

    setMessages(prev => [...prev, {
      id: (Date.now() + 1).toString(),
      role: 'model',
      text: analysis,
      timestamp: new Date()
    }]);
    setIsThinking(false);
  };

  return (
    <div className="bg-slate-800 rounded-xl shadow-lg border border-slate-700 flex flex-col h-[600px]">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50 backdrop-blur rounded-t-xl">
        <h3 className="font-bold text-white flex items-center gap-2">
          <Bot className="text-purple-400" />
          QuantMind 智能顾问
        </h3>
        <button 
          onClick={handleQuickAnalysis}
          className="text-xs bg-purple-600 hover:bg-purple-500 text-white px-3 py-1.5 rounded-full flex items-center gap-1 transition-all"
        >
          <Sparkles size={12} />
          一键风险诊断
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={scrollRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] p-3 rounded-2xl ${
              msg.role === 'user' 
                ? 'bg-blue-600 text-white rounded-tr-sm' 
                : 'bg-slate-700 text-slate-200 rounded-tl-sm'
            }`}>
              {msg.role === 'model' && (
                <div className="flex items-center gap-1 text-xs text-purple-300 mb-1 font-bold">
                  <Bot size={12} /> QuantMind
                </div>
              )}
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}
        {isThinking && (
          <div className="flex justify-start">
            <div className="bg-slate-700 p-3 rounded-2xl rounded-tl-sm text-slate-400 text-xs flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce delay-75"></div>
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce delay-150"></div>
              正在分析市场数据...
            </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-slate-700 bg-slate-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="询问关于茅台的趋势，或者多因子选股策略..."
            className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 text-sm text-white focus:outline-none focus:border-purple-500 placeholder-slate-500"
          />
          <button 
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isThinking}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white p-2 rounded-lg transition-colors"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default AIAnalyst;