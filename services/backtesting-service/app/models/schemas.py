"""
Pydantic models for Backtesting Service API
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class BacktestStatus(str, Enum):
    """Backtest execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StrategyType(str, Enum):
    """Strategy types supported"""
    MA_CROSSOVER = "MA_CROSSOVER"
    RSI = "RSI"
    CUSTOM = "CUSTOM"


class BacktestRequest(BaseModel):
    """Request to start a backtest"""
    strategy_type: StrategyType = Field(..., description="Type of strategy to backtest")
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (e.g., M5, M15, H1, H4, D1)")
    from_time: int = Field(..., description="Start timestamp (Unix epoch)")
    to_time: int = Field(..., description="End timestamp (Unix epoch)")
    initial_balance: float = Field(10000.0, description="Initial account balance")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    commission: float = Field(0.0, description="Commission per trade (USD)")
    spread_pips: float = Field(1.0, description="Spread in pips")
    slippage_pips: float = Field(0.5, description="Slippage in pips")
    position_size: float = Field(0.01, description="Position size in lots")


class OptimizationRequest(BaseModel):
    """Request to optimize strategy parameters"""
    strategy_type: StrategyType = Field(..., description="Type of strategy to optimize")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    from_time: int = Field(..., description="Start timestamp for in-sample data")
    to_time: int = Field(..., description="End timestamp for in-sample data")
    initial_balance: float = Field(10000.0, description="Initial account balance")
    parameter_ranges: Dict[str, List[Any]] = Field(..., description="Parameter ranges to test")
    commission: float = Field(0.0, description="Commission per trade")
    spread_pips: float = Field(1.0, description="Spread in pips")
    slippage_pips: float = Field(0.5, description="Slippage in pips")
    position_size: float = Field(0.01, description="Position size in lots")
    optimization_metric: str = Field("sharpe_ratio", description="Metric to optimize (sharpe_ratio, profit_factor, etc.)")
    walk_forward: bool = Field(False, description="Enable walk-forward analysis")
    walk_forward_params: Optional[Dict[str, Any]] = Field(None, description="Walk-forward analysis parameters")


class TradeRecord(BaseModel):
    """Individual trade record"""
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    position_type: str  # "BUY" or "SELL"
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float
    swap: float = 0.0
    profit: Optional[float] = None
    profit_pips: Optional[float] = None
    mae: Optional[float] = None  # Maximum Adverse Excursion
    mfe: Optional[float] = None  # Maximum Favorable Excursion
    duration_seconds: Optional[int] = None
    exit_reason: Optional[str] = None  # "TP", "SL", "SIGNAL", "EOD"


class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profit metrics
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    # Return metrics
    total_return: float = 0.0  # Percentage
    annualized_return: float = 0.0  # Percentage

    # Average metrics
    average_win: float = 0.0
    average_loss: float = 0.0
    average_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Consecutive metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Drawdown metrics
    max_drawdown: float = 0.0  # Percentage
    max_drawdown_duration_days: float = 0.0
    average_drawdown: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sterling_ratio: float = 0.0

    # Statistical metrics
    expectancy: float = 0.0  # Expected value per trade
    recovery_factor: float = 0.0  # Net profit / Max drawdown

    # MAE/MFE metrics
    average_mae: float = 0.0
    average_mfe: float = 0.0

    # Trade duration
    average_trade_duration_hours: float = 0.0

    # Account metrics
    initial_balance: float = 0.0
    final_balance: float = 0.0
    peak_balance: float = 0.0

    # Commission and costs
    total_commission: float = 0.0
    total_swap: float = 0.0


class EquityPoint(BaseModel):
    """Single equity curve data point"""
    timestamp: datetime
    balance: float
    equity: float
    drawdown: float
    drawdown_percent: float


class BacktestResult(BaseModel):
    """Complete backtest result"""
    backtest_id: str
    status: BacktestStatus
    strategy_type: StrategyType
    symbol: str
    timeframe: str
    from_time: datetime
    to_time: datetime
    parameters: Dict[str, Any]

    # Results
    performance: Optional[PerformanceMetrics] = None
    trades: Optional[List[TradeRecord]] = None
    equity_curve: Optional[List[EquityPoint]] = None

    # Metadata
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None

    # Backtest settings
    initial_balance: float
    commission: float
    spread_pips: float
    slippage_pips: float
    position_size: float


class OptimizationResult(BaseModel):
    """Parameter optimization result"""
    optimization_id: str
    status: BacktestStatus
    strategy_type: StrategyType
    symbol: str
    timeframe: str

    # Best parameters found
    best_parameters: Optional[Dict[str, Any]] = None
    best_metric_value: Optional[float] = None
    optimization_metric: str

    # All results
    all_results: Optional[List[Dict[str, Any]]] = None
    total_combinations: int = 0
    completed_combinations: int = 0

    # Walk-forward results
    walk_forward_results: Optional[List[Dict[str, Any]]] = None

    # Metadata
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None


class BacktestListItem(BaseModel):
    """Summary item for listing backtests"""
    backtest_id: str
    status: BacktestStatus
    strategy_type: StrategyType
    symbol: str
    timeframe: str
    created_at: datetime
    net_profit: Optional[float] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None


class BacktestDeleteResponse(BaseModel):
    """Response for backtest deletion"""
    success: bool
    message: str
    backtest_id: str
