// webui/src/services/geminiService.ts

import type { Position, Strategy, MarketTrend, NewsItem } from '../types';

/**
 * Demo Gemini service placeholders. No external API calls to keep dev env simple.
 */
export const analyzePortfolio = async (
  positions: Position[],
  strategies: Strategy[]
): Promise<string> => {
  const totalPositions = positions.length;
  const winning = positions.filter(p => p.pnlPercent > 0).length;
  const losing = positions.filter(p => p.pnlPercent <= 0).length;

  const avgPnl =
    positions.length > 0
      ? positions.reduce((sum, p) => sum + p.pnlPercent, 0) / positions.length
      : 0;

  const activeStrategies = strategies.filter(s => s.status === 'active').length;

  return [
    '[Demo mode | No real Gemini call]',
    '',
    `Holdings: ${totalPositions} positions (${winning} up / ${losing} down), avg PnL ~${avgPnl.toFixed(2)}%.`,
    `Active strategies: ${activeStrategies}. Keep concentration and drawdown in check.`,
    '',
    'Hook up the real Gemini API later for richer analysis.'
  ].join('\n');
};

export const chatWithQuantAI = async (
  userMessage: string,
  context: {
    positions: Position[];
    strategies: Strategy[];
    trend: MarketTrend;
  }
): Promise<string> => {
  const hint =
    context.trend === 'Bullish'
      ? 'Market leans bullish; avoid overexposure and leave some safety buffer.'
      : context.trend === 'Bearish'
      ? 'Market leans bearish; manage sizing and avoid emotional averaging down.'
      : 'Market is range-bound; grid/mean-reversion styles may fit better.';

  return [
    '[Demo mode | Chat not calling real Gemini]',
    '',
    `You said: ${userMessage}`,
    '',
    `Context hint: ${hint}`,
    '',
    'Provide a Gemini API key later for tailored responses.'
  ].join('\n');
};

export const generateDailyBriefing = async (news: NewsItem[]): Promise<string> => {
  const total = news.length;
  const bullish = news.filter(n => n.sentiment === 'Bullish').length;
  const bearish = news.filter(n => n.sentiment === 'Bearish').length;

  return [
    '【演示模式｜未真实调用 Gemini】',
    total > 0
      ? `共收到 ${total} 条新闻，其中多头 ${bullish} 条、空头 ${bearish} 条。`
      : '当前未提供新闻列表，将使用默认示例进行判断。',
    '等后端接入真实财经新闻和 Gemini API 后，这里会生成每日市场早报摘要。'
  ].join('\n');
};
