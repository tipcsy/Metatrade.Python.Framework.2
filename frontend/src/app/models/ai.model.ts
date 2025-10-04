export interface AIModel {
  id: number;
  name: string;
  type: 'classification' | 'regression' | 'reinforcement';
  status: 'training' | 'ready' | 'error' | 'idle';
  accuracy?: number;
  lastTrained?: Date;
  trainingProgress?: number;
  parameters?: Record<string, any>;
}

export interface TrainModelRequest {
  name: string;
  type: 'classification' | 'regression' | 'reinforcement';
  symbol: string;
  timeframe: string;
  features: string[];
  targetVariable: string;
  trainingDataStart: string;
  trainingDataEnd: string;
  parameters?: Record<string, any>;
}

export interface PredictionRequest {
  modelId: number;
  symbol: string;
  currentData?: Record<string, number>;
}

export interface PredictionResult {
  modelId: number;
  symbol: string;
  prediction: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  probability?: {
    buy: number;
    sell: number;
    hold: number;
  };
  features?: Record<string, number>;
  timestamp: Date;
}

export interface ModelPerformance {
  modelId: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix?: number[][];
  rocAuc?: number;
}
