
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { Position, Strategy, MarketTrend, NewsItem } from '../types';

// Initialize Gemini Client
// Note: In a real production app, you might proxy this through a backend to protect the key.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const analyzePortfolio = async (
  positions: Position[],
  strategies: Strategy[]
): Promise<string> => {
  try {
    const portfolioSummary = positions.map(p => 
      `${p.symbol} (${p.name}): 持仓 ${p.amount} 股 @ 成本 ¥${p.avgPrice} (现价: ¥${p.currentPrice}, 盈亏比: ${p.pnlPercent}%)`
    ).join('\n');

    const strategySummary = strategies.map(s => 
      `${s.name} [状态: ${s.status}]: 回报率 ${s.returnRate}%, 夏普比率 ${s.sharpeRatio}`
    ).join('\n');

    const prompt = `
      作为一名资深的A股量化交易分析师，请根据以下的日线级别策略和持仓情况，提供一份简明的风险分析和优化建议。
      
      当前持仓 (A股):
      ${portfolioSummary}

      运行策略:
      ${strategySummary}

      请提供以下分析 (请使用中文):
      1. 整体风险评估 (低/中/高)，考虑到A股市场的特性。
      2. 当前组合的一个主要优势。
      3. 一条具体的优化建议，以提高夏普比率或控制最大回撤。
      
      保持语气专业、客观、数据驱动。回答限制在200字以内。
    `;

    const response: GenerateContentResponse = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    return response.text || "暂时无法进行分析。";
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    return "由于网络或API限制，无法生成分析报告。";
  }
};

export const chatWithQuantAI = async (
  history: {role: string, parts: {text: string}[]}[],
  newMessage: string
): Promise<string> => {
  try {
    const chat = ai.chats.create({
      model: 'gemini-2.5-flash',
      config: {
        systemInstruction: "你是 'QuantMind'，一位精通中国A股市场的量化交易助手。你擅长解释日线级别的趋势信号、量化因子（如动量、价值、波动率）以及A股特有的市场规则（如涨跌停、T+1）。回答要简洁、专业且通俗易懂。请始终使用中文回答。",
      },
      history: history
    });

    const result = await chat.sendMessage({ message: newMessage });
    return result.text || "";
  } catch (error) {
    console.error("Gemini Chat Error:", error);
    return "我现在连接不稳定，请稍后再试。";
  }
};

export const generateDailyBriefing = async (news: NewsItem[]): Promise<string> => {
  try {
    const newsContent = news.slice(0, 10).map(n => `- [${n.sentiment}] ${n.title} (${n.source})`).join('\n');
    
    const prompt = `
      基于以下今日A股市场的财经新闻要点，请生成一份简短的市场研报摘要（Daily Briefing）。
      
      新闻列表:
      ${newsContent}

      请包含:
      1. 市场整体情绪判定 (贪婪/恐慌/观望)。
      2. 核心热点板块或概念。
      3. 对明日交易的一个简短策略提示。
      
      格式要求：使用Markdown，简洁明了，总字数不超过300字。
    `;

    const response: GenerateContentResponse = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    return response.text || "无法生成简报。";
  } catch (error) {
    console.error("News Briefing Error:", error);
    return "AI 服务暂时不可用。";
  }
};
