"""
Data models for Strategy Service
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class StrategyCreateRequest(BaseModel):
    """Request model for creating a strategy"""
    strategy_type: str = Field(..., description="Strategy type (MA_CROSSOVER, RSI)")
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (e.g., M15, H1, H4)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class StrategyResponse(BaseModel):
    """Response model for strategy information"""
    strategy_id: str
    name: str
    symbol: str
    timeframe: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    signals_count: int = 0
    trades_count: int = 0
    parameters: Dict[str, Any] = {}


class PositionResponse(BaseModel):
    """Response model for position information"""
    position_id: str
    symbol: str
    type: str
    volume: float
    open_price: float
    close_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: Optional[str] = None
    status: str
    open_time: str
    close_time: Optional[str] = None
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0


class ClosePositionRequest(BaseModel):
    """Request model for closing a position"""
    position_id: str = Field(..., description="Position ID to close")
    close_price: Optional[float] = Field(None, description="Close price (if not provided, uses current market price)")


class RiskStatusResponse(BaseModel):
    """Response model for risk status"""
    account_balance: float
    initial_balance: float
    peak_balance: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    daily_loss: float
    daily_loss_pct: float
    max_daily_loss_pct: float
    max_positions: int
    max_risk_per_trade_pct: float
    risk_limits_active: bool
    profit_loss: float


class PositionStatisticsResponse(BaseModel):
    """Response model for position statistics"""
    total_positions: int
    open_positions: int
    closed_positions: int
    total_profit: float
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_profit: float


class StrategyStatisticsResponse(BaseModel):
    """Response model for strategy statistics"""
    total_strategies: int
    running_strategies: int
    stopped_strategies: int
    paused_strategies: int
    mock_mode: bool
