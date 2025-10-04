export interface BacktestRequest {
  symbol: string;
  timeframe: string;
  strategyName: string;
  startDate: string;
  endDate: string;
  initialBalance: number;
  parameters?: Record<string, any>;
}

export interface BacktestResult {
  id: number;
  symbol: string;
  timeframe: string;
  strategyName: string;
  startDate: string;
  endDate: string;
  initialBalance: number;
  finalBalance: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  maxDrawdown: number;
  sharpeRatio: number;
  profitFactor: number;
  equityCurve: EquityPoint[];
  trades: BacktestTrade[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  createdAt: Date;
  completedAt?: Date;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown: number;
}

export interface BacktestTrade {
  entryTime: string;
  exitTime: string;
  type: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice: number;
  volume: number;
  profit: number;
  profitPercent: number;
  duration: number;
}
