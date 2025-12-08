// webui/src/services/geminiService.ts

import type { Position, Strategy, MarketTrend } from '../types';

/**
 * 演示版 Gemini 服务：
 * 先不真连 Google AI，避免因为环境变量 / 网络问题把整个前端页面搞崩。
 * 以后如果要接正式的 Gemini，再在这里替换实现即可。
 */

export const analyzePortfolio = async (
  positions: Position[],
  strategies: Strategy[]
): Promise<string> => {
  // 简单拼一段“持仓+策略总结”的文案，假装是 AI 分析结果
  const totalPositions = positions.length;
  const winning = positions.filter(p => p.pnlPercent > 0).length;
  const losing = positions.filter(p => p.pnlPercent <= 0).length;

  const avgPnl =
    positions.length > 0
      ? positions.reduce((sum, p) => sum + p.pnlPercent, 0) / positions.length
      : 0;

  const activeStrategies = strategies.filter(s => s.status === 'active').length;

  return [
    '【演示模式｜本回答未调用真实 AI】',
    '',
    `当前持仓共 ${totalPositions} 个标的，其中盈利 ${winning} 个、亏损 ${losing} 个，整体平均收益率约为 ${avgPnl.toFixed(2)}%。`,
    `正在运行的量化策略共有 ${activeStrategies} 个，建议控制单一策略和单一行业的集中度，防止回撤过度集中。`,
    '',
    '后续接入真实 Gemini API 后，我可以基于更多历史数据、波动率、回撤曲线给出更精细的风险评估和调仓建议。'
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
  // 同样给一个简单回声 + 一点提示，避免前端报错
  const hint =
    context.trend === 'Bullish'
      ? '当前市场偏多头，注意不要满仓梭哈，留出一定安全垫。'
      : context.trend === 'Bearish'
      ? '当前市场偏空头，控制仓位、防止情绪化加仓是重点。'
      : '当前市场处于震荡区间，量化策略可以多做一些网格 / 高抛低吸类配置。';

  return [
    '【演示模式｜聊天未调用真实 AI】',
    '',
    `你刚才说：${userMessage}`,
    '',
    `结合你当前的持仓与策略配置，给一个通用提示：${hint}`,
    '',
    '等你后面配置好 Gemini API Key 之后，这里会返回更贴合你账户和策略的专业回答。'
  ].join('\n');
};
