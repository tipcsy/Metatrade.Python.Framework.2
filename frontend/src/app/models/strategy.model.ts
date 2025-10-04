export interface Strategy {
  id: number;
  name: string;
  mode: 'live' | 'paper' | 'off';
  symbols: string[];
  timeframe: string;
  status: 'running' | 'stopped' | 'error';
  profitToday: number;
  totalProfit: number;
  openPositions: number;
  winRate?: number;
  maxDrawdown?: number;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface CreateStrategyRequest {
  name: string;
  mode: 'live' | 'paper' | 'off';
  symbols: string[];
  timeframe: string;
  parameters?: Record<string, any>;
}

export interface Position {
  id: number;
  strategyId: number;
  symbol: string;
  type: 'BUY' | 'SELL';
  volume: number;
  openPrice: number;
  currentPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  profit: number;
  openTime: Date;
  closeTime?: Date;
  status: 'open' | 'closed';
}

export interface StrategyPerformance {
  strategyId: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalProfit: number;
  totalLoss: number;
  netProfit: number;
  maxDrawdown: number;
  sharpeRatio?: number;
  profitFactor?: number;
}
