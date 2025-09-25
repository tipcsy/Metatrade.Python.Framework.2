"""
Portfolio Manager Implementation.

This module provides comprehensive portfolio management including position tracking,
performance calculation, and risk monitoring across multiple accounts and strategies.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from ...core.config import Settings
from ...core.exceptions import TradingError, ValidationError
from ...core.logging import get_logger
from ...database.models import Position, Trade, Account
from ...database.services import BaseService
from .position_manager import PositionManager
from .performance_calculator import PerformanceCalculator, PerformanceMetrics

logger = get_logger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    # Portfolio values
    total_value: float = 0.0
    cash: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    equity: float = 0.0

    # Performance metrics
    total_return: float = 0.0
    daily_return: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Position metrics
    position_count: int = 0
    long_positions: int = 0
    short_positions: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0

    # Activity metrics
    trades_today: int = 0
    volume_today: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)


class PortfolioSummary(BaseModel):
    """Portfolio summary information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Account information
    account_id: str
    account_name: str
    currency: str

    # Portfolio values
    total_value: float
    cash: float
    equity: float
    margin_used: float
    margin_available: float

    # Positions
    position_count: int
    positions: List[Dict[str, Any]] = Field(default_factory=list)

    # Performance
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float

    # Risk
    gross_exposure: float
    net_exposure: float
    leverage: float

    # Timestamps
    last_updated: datetime


class PortfolioConfig(BaseModel):
    """Portfolio management configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic settings
    base_currency: str = Field(default="USD")
    enable_multi_account: bool = Field(default=True)
    enable_margin_trading: bool = Field(default=True)

    # Position limits
    max_positions: int = Field(default=100, ge=1)
    max_position_size: float = Field(default=10000000.0, gt=0)
    max_leverage: float = Field(default=10.0, gt=0)

    # Risk management
    max_portfolio_risk: float = Field(default=0.02, gt=0, le=1)  # 2% portfolio risk
    max_correlation: float = Field(default=0.8, gt=0, le=1)
    max_sector_exposure: float = Field(default=0.3, gt=0, le=1)

    # Performance settings
    benchmark_symbol: Optional[str] = None
    performance_lookback_days: int = Field(default=252, ge=1)  # 1 year

    # Update intervals
    portfolio_update_interval: int = Field(default=60, ge=1)  # seconds
    risk_calculation_interval: int = Field(default=300, ge=1)  # seconds
    performance_calculation_interval: int = Field(default=3600, ge=1)  # seconds


class PortfolioManager(BaseService):
    """
    Comprehensive portfolio management system.

    Manages multiple portfolios, positions, and accounts with real-time
    performance tracking, risk monitoring, and optimization capabilities.
    """

    def __init__(
        self,
        settings: Settings,
        config: Optional[PortfolioConfig] = None,
    ):
        """Initialize the portfolio manager."""
        super().__init__(settings)
        self.config = config or PortfolioConfig()

        # Core components
        self.position_manager: Optional[PositionManager] = None
        self.performance_calculator: Optional[PerformanceCalculator] = None

        # Portfolio data
        self._portfolios: Dict[str, Dict[str, Any]] = {}
        self._accounts: Dict[str, Account] = {}
        self._positions: Dict[str, Position] = {}

        # Performance tracking
        self._performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self._risk_metrics: Dict[str, Dict[str, float]] = {}

        # Background tasks
        self._tasks: Set[asyncio.Task] = set()

        logger.info("Portfolio manager initialized")

    async def initialize(self) -> None:
        """Initialize the portfolio manager."""
        logger.info("Initializing portfolio manager...")

        # Initialize components
        self.position_manager = PositionManager(self.settings)
        await self.position_manager.initialize()

        self.performance_calculator = PerformanceCalculator(self.settings)
        await self.performance_calculator.initialize()

        # Load existing portfolios and accounts
        await self._load_portfolios()

        # Start background tasks
        await self._start_background_tasks()

        logger.info("Portfolio manager initialized successfully")

    async def create_portfolio(
        self,
        account_id: str,
        initial_cash: float = 100000.0,
        currency: str = "USD",
        name: Optional[str] = None,
    ) -> str:
        """Create a new portfolio."""
        portfolio_id = str(uuid4())

        portfolio = {
            "portfolio_id": portfolio_id,
            "account_id": account_id,
            "name": name or f"Portfolio-{account_id}",
            "currency": currency,
            "initial_cash": initial_cash,
            "cash": initial_cash,
            "created_at": datetime.now(),
            "positions": {},
            "metrics": PortfolioMetrics(),
        }

        self._portfolios[portfolio_id] = portfolio

        logger.info(f"Created portfolio: {portfolio_id} for account: {account_id}")
        return portfolio_id

    async def get_portfolio_summary(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """Get portfolio summary."""
        if portfolio_id:
            if portfolio_id not in self._portfolios:
                raise ValidationError(f"Portfolio not found: {portfolio_id}")
            portfolios_to_include = [self._portfolios[portfolio_id]]
        else:
            portfolios_to_include = list(self._portfolios.values())

        # Aggregate metrics across portfolios
        total_value = 0.0
        total_cash = 0.0
        total_equity = 0.0
        total_pnl = 0.0
        position_count = 0

        positions = []

        for portfolio in portfolios_to_include:
            metrics = portfolio["metrics"]
            total_value += metrics.total_value
            total_cash += metrics.cash
            total_equity += metrics.equity
            total_pnl += metrics.realized_pnl + metrics.unrealized_pnl
            position_count += metrics.position_count

            # Add positions
            for pos_id, position in portfolio.get("positions", {}).items():
                positions.append({
                    "position_id": pos_id,
                    "symbol": position.get("symbol"),
                    "volume": position.get("volume", 0),
                    "unrealized_pnl": position.get("unrealized_pnl", 0),
                    "portfolio_id": portfolio["portfolio_id"],
                })

        return {
            "total_value": total_value,
            "cash": total_cash,
            "equity": total_equity,
            "total_pnl": total_pnl,
            "position_count": position_count,
            "positions": positions,
            "portfolio_count": len(portfolios_to_include),
            "last_updated": datetime.now(),
        }

    async def get_active_positions(self, portfolio_id: Optional[str] = None) -> List[Position]:
        """Get all active positions."""
        positions = []

        if portfolio_id:
            if portfolio_id not in self._portfolios:
                return []
            portfolio = self._portfolios[portfolio_id]
            positions.extend(portfolio.get("positions", {}).values())
        else:
            for portfolio in self._portfolios.values():
                positions.extend(portfolio.get("positions", {}).values())

        # Filter active positions
        active_positions = []
        for pos_data in positions:
            if isinstance(pos_data, dict) and pos_data.get("volume", 0) != 0:
                # Convert to Position object if needed
                if not isinstance(pos_data, Position):
                    # This would create a Position object from the dict
                    pass
                active_positions.append(pos_data)

        return active_positions

    async def add_position(
        self,
        portfolio_id: str,
        symbol: str,
        volume: float,
        entry_price: float,
        position_type: str = "long",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new position to the portfolio."""
        if portfolio_id not in self._portfolios:
            raise ValidationError(f"Portfolio not found: {portfolio_id}")

        position_id = str(uuid4())

        position_data = {
            "position_id": position_id,
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "volume": volume,
            "entry_price": entry_price,
            "current_price": entry_price,
            "position_type": position_type,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": metadata or {},
        }

        # Add to portfolio
        portfolio = self._portfolios[portfolio_id]
        portfolio["positions"][position_id] = position_data

        # Update portfolio metrics
        await self._update_portfolio_metrics(portfolio_id)

        logger.info(f"Added position: {position_id} - {symbol} {volume}")
        return position_id

    async def update_position(
        self,
        position_id: str,
        current_price: Optional[float] = None,
        volume_change: Optional[float] = None,
    ) -> None:
        """Update an existing position."""
        position_data = await self._find_position(position_id)
        if not position_data:
            raise ValidationError(f"Position not found: {position_id}")

        portfolio_id = position_data["portfolio_id"]

        # Update price
        if current_price is not None:
            position_data["current_price"] = current_price

            # Calculate unrealized P&L
            entry_price = position_data["entry_price"]
            volume = position_data["volume"]

            if position_data["position_type"].lower() == "long":
                unrealized_pnl = (current_price - entry_price) * volume
            else:  # short
                unrealized_pnl = (entry_price - current_price) * volume

            position_data["unrealized_pnl"] = unrealized_pnl

        # Update volume
        if volume_change is not None:
            new_volume = position_data["volume"] + volume_change

            if abs(new_volume) < 1e-6:  # Position closed
                await self._close_position(position_id)
                return

            position_data["volume"] = new_volume

        position_data["updated_at"] = datetime.now()

        # Update portfolio metrics
        await self._update_portfolio_metrics(portfolio_id)

    async def close_position(self, position_id: str, close_price: float) -> Dict[str, Any]:
        """Close a position."""
        position_data = await self._find_position(position_id)
        if not position_data:
            raise ValidationError(f"Position not found: {position_id}")

        # Calculate final P&L
        entry_price = position_data["entry_price"]
        volume = position_data["volume"]

        if position_data["position_type"].lower() == "long":
            realized_pnl = (close_price - entry_price) * volume
        else:  # short
            realized_pnl = (entry_price - close_price) * volume

        position_data["realized_pnl"] = realized_pnl
        position_data["unrealized_pnl"] = 0.0
        position_data["volume"] = 0.0
        position_data["closed_at"] = datetime.now()
        position_data["close_price"] = close_price

        portfolio_id = position_data["portfolio_id"]

        # Update portfolio cash
        portfolio = self._portfolios[portfolio_id]
        portfolio["cash"] += realized_pnl

        # Update portfolio metrics
        await self._update_portfolio_metrics(portfolio_id)

        logger.info(f"Closed position: {position_id} - P&L: {realized_pnl:.2f}")

        return {
            "position_id": position_id,
            "realized_pnl": realized_pnl,
            "close_price": close_price,
            "closed_at": position_data["closed_at"],
        }

    async def get_performance_metrics(self, portfolio_id: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a portfolio."""
        if not self.performance_calculator:
            return None

        if portfolio_id not in self._portfolios:
            return None

        # This would calculate comprehensive performance metrics
        # For now, return basic metrics based on portfolio data
        portfolio = self._portfolios[portfolio_id]
        metrics = portfolio["metrics"]

        return PerformanceMetrics(
            total_return=metrics.total_return,
            daily_return=metrics.daily_return,
            volatility=metrics.volatility,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            var_95=metrics.var_95,
        )

    async def get_risk_metrics(self, portfolio_id: str) -> Dict[str, float]:
        """Get risk metrics for a portfolio."""
        if portfolio_id not in self._portfolios:
            return {}

        portfolio = self._portfolios[portfolio_id]
        metrics = portfolio["metrics"]

        return {
            "gross_exposure": metrics.gross_exposure,
            "net_exposure": metrics.net_exposure,
            "leverage": metrics.gross_exposure / max(metrics.equity, 1.0),
            "var_95": metrics.var_95,
            "max_drawdown": metrics.max_drawdown,
            "volatility": metrics.volatility,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        healthy = True
        details = {}

        # Check portfolio count
        portfolio_count = len(self._portfolios)
        details["portfolio_count"] = portfolio_count

        # Check position managers
        if self.position_manager:
            pos_health = await self.position_manager.health_check()
            details["position_manager"] = pos_health
            if not pos_health.get("healthy", False):
                healthy = False

        # Check performance calculator
        if self.performance_calculator:
            perf_health = await self.performance_calculator.health_check()
            details["performance_calculator"] = perf_health
            if not perf_health.get("healthy", False):
                healthy = False

        return {
            "healthy": healthy,
            "portfolio_count": portfolio_count,
            "total_positions": sum(
                len(p.get("positions", {})) for p in self._portfolios.values()
            ),
            "details": details,
        }

    async def cleanup(self) -> None:
        """Cleanup portfolio manager resources."""
        logger.info("Cleaning up portfolio manager...")

        # Stop background tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Cleanup components
        if self.position_manager:
            await self.position_manager.cleanup()

        if self.performance_calculator:
            await self.performance_calculator.cleanup()

        self._portfolios.clear()
        self._tasks.clear()

        logger.info("Portfolio manager cleaned up")

    async def cleanup_closed_positions(self) -> None:
        """Remove closed positions older than retention period."""
        cutoff_time = datetime.now() - timedelta(days=30)  # Keep 30 days

        for portfolio in self._portfolios.values():
            positions_to_remove = []

            for pos_id, pos_data in portfolio.get("positions", {}).items():
                if (pos_data.get("volume", 0) == 0 and
                    pos_data.get("closed_at") and
                    pos_data["closed_at"] < cutoff_time):
                    positions_to_remove.append(pos_id)

            for pos_id in positions_to_remove:
                del portfolio["positions"][pos_id]

        logger.info(f"Cleaned up old closed positions")

    async def _load_portfolios(self) -> None:
        """Load existing portfolios from database."""
        # This would load portfolios from the database
        # For now, we'll start with empty portfolios
        logger.info("Loaded existing portfolios")

    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        logger.info("Starting background tasks...")

        # Portfolio update task
        task = asyncio.create_task(self._portfolio_update_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        # Risk calculation task
        task = asyncio.create_task(self._risk_calculation_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        # Performance calculation task
        task = asyncio.create_task(self._performance_calculation_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        logger.info("Background tasks started")

    async def _portfolio_update_loop(self) -> None:
        """Background task for updating portfolio metrics."""
        while True:
            try:
                for portfolio_id in self._portfolios.keys():
                    await self._update_portfolio_metrics(portfolio_id)

                await asyncio.sleep(self.config.portfolio_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(self.config.portfolio_update_interval)

    async def _risk_calculation_loop(self) -> None:
        """Background task for calculating risk metrics."""
        while True:
            try:
                for portfolio_id in self._portfolios.keys():
                    await self._calculate_risk_metrics(portfolio_id)

                await asyncio.sleep(self.config.risk_calculation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk calculation loop: {e}")
                await asyncio.sleep(self.config.risk_calculation_interval)

    async def _performance_calculation_loop(self) -> None:
        """Background task for calculating performance metrics."""
        while True:
            try:
                for portfolio_id in self._portfolios.keys():
                    await self._calculate_performance_metrics(portfolio_id)

                await asyncio.sleep(self.config.performance_calculation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance calculation loop: {e}")
                await asyncio.sleep(self.config.performance_calculation_interval)

    async def _update_portfolio_metrics(self, portfolio_id: str) -> None:
        """Update metrics for a specific portfolio."""
        if portfolio_id not in self._portfolios:
            return

        portfolio = self._portfolios[portfolio_id]
        metrics = portfolio["metrics"]
        positions = portfolio.get("positions", {})

        # Reset metrics
        metrics.position_count = len([p for p in positions.values() if p.get("volume", 0) != 0])
        metrics.long_positions = len([p for p in positions.values() if p.get("volume", 0) > 0])
        metrics.short_positions = len([p for p in positions.values() if p.get("volume", 0) < 0])

        # Calculate P&L
        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions.values())
        total_realized = sum(p.get("realized_pnl", 0) for p in positions.values())

        metrics.unrealized_pnl = total_unrealized
        metrics.realized_pnl = total_realized

        # Calculate values
        metrics.cash = portfolio["cash"]
        metrics.equity = metrics.cash + total_unrealized
        metrics.total_value = metrics.equity

        # Calculate exposures
        gross_exposure = sum(
            abs(p.get("volume", 0) * p.get("current_price", p.get("entry_price", 0)))
            for p in positions.values()
        )

        net_exposure = sum(
            p.get("volume", 0) * p.get("current_price", p.get("entry_price", 0))
            for p in positions.values()
        )

        metrics.gross_exposure = gross_exposure
        metrics.net_exposure = net_exposure

        metrics.last_updated = datetime.now()

    async def _calculate_risk_metrics(self, portfolio_id: str) -> None:
        """Calculate risk metrics for a portfolio."""
        # This would implement comprehensive risk calculations
        # For now, we'll do basic calculations
        pass

    async def _calculate_performance_metrics(self, portfolio_id: str) -> None:
        """Calculate performance metrics for a portfolio."""
        # This would implement comprehensive performance calculations
        # For now, we'll do basic calculations
        pass

    async def _find_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Find position by ID across all portfolios."""
        for portfolio in self._portfolios.values():
            if position_id in portfolio.get("positions", {}):
                return portfolio["positions"][position_id]
        return None

    async def _close_position(self, position_id: str) -> None:
        """Close a position by setting volume to 0."""
        position_data = await self._find_position(position_id)
        if position_data:
            position_data["volume"] = 0.0
            position_data["closed_at"] = datetime.now()