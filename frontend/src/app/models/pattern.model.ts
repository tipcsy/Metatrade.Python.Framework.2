export interface TechnicalIndicator {
  name: string;
  symbol: string;
  timeframe: string;
  value: number | number[];
  signal?: 'BUY' | 'SELL' | 'NEUTRAL';
  timestamp: Date;
}

export interface CandlestickPattern {
  name: string;
  symbol: string;
  timeframe: string;
  type: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  timestamp: Date;
  description?: string;
}

export interface PatternScanRequest {
  symbols: string[];
  timeframe: string;
  indicators?: string[];
}

export interface PatternScanResult {
  symbol: string;
  patterns: CandlestickPattern[];
  indicators: TechnicalIndicator[];
  recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  score: number;
}

export interface ChartData {
  symbol: string;
  timeframe: string;
  candles: Candle[];
  indicators: Record<string, number[]>;
}

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
