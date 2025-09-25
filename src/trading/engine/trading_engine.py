"""
Core Trading Engine Implementation.

This module provides the main trading engine that coordinates all trading activities
including strategy execution, order management, risk control, and performance monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from ...core.config import Settings
from ...core.exceptions import TradingError, ValidationError
from ...core.logging import get_logger
from ...database.models import Strategy, Order, Position, Trade
from ...database.services import BaseService
from .strategy_manager import StrategyManager, StrategyInstance
from .execution_engine import ExecutionEngine, ExecutionResult
from ..portfolio.portfolio_manager import PortfolioManager
from ..risk.risk_manager import RiskManager
from ..orders.order_manager import OrderManager

logger = get_logger(__name__)


class EngineState(str, Enum):
    """Trading engine state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineMetrics:
    """Trading engine performance metrics."""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    active_positions: int = 0
    active_orders: int = 0
    strategies_running: int = 0
    average_execution_time: float = 0.0
    risk_violations: int = 0
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class TradingEngineConfig(BaseModel):
    """Trading engine configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Engine settings
    max_concurrent_strategies: int = Field(default=10, ge=1, le=100)
    max_concurrent_orders: int = Field(default=1000, ge=1)
    max_position_size: float = Field(default=1000000.0, gt=0)

    # Risk management
    enable_risk_management: bool = Field(default=True)
    max_daily_loss: float = Field(default=10000.0, gt=0)
    max_drawdown_percent: float = Field(default=20.0, gt=0, le=100)

    # Performance settings
    metrics_update_interval: int = Field(default=60, ge=1)  # seconds
    cleanup_interval: int = Field(default=3600, ge=60)  # seconds
    position_check_interval: int = Field(default=30, ge=1)  # seconds

    # Strategy settings
    strategy_timeout: int = Field(default=300, ge=1)  # seconds
    max_strategy_errors: int = Field(default=5, ge=1)

    # Order management
    order_timeout: int = Field(default=30, ge=1)  # seconds
    max_slippage_percent: float = Field(default=0.5, ge=0, le=10)

    # Execution settings
    enable_paper_trading: bool = Field(default=False)
    enable_order_validation: bool = Field(default=True)
    enable_position_tracking: bool = Field(default=True)


class TradingEngine(BaseService):
    """
    Advanced multi-strategy trading engine.

    This engine coordinates all trading activities including:
    - Strategy execution and management
    - Order placement and execution
    - Position and portfolio management
    - Risk management and monitoring
    - Performance tracking and reporting
    """

    def __init__(
        self,
        settings: Settings,
        config: Optional[TradingEngineConfig] = None,
    ):
        """
        Initialize the trading engine.

        Args:
            settings: Application settings
            config: Trading engine configuration
        """
        super().__init__(settings)
        self.config = config or TradingEngineConfig()
        self.state = EngineState.STOPPED
        self.metrics = EngineMetrics()

        # Core components
        self.strategy_manager: Optional[StrategyManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None

        # Internal state
        self._start_time: Optional[datetime] = None
        self._tasks: Set[asyncio.Task] = set()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._strategy_instances: Dict[str, StrategyInstance] = {}
        self._active_orders: Dict[str, Order] = {}
        self._active_positions: Dict[str, Position] = {}

        # Performance tracking
        self._last_metrics_update = datetime.now()
        self._execution_times: List[float] = []

        logger.info("Trading engine initialized")

    async def start(self) -> None:
        """Start the trading engine."""
        try:
            logger.info("Starting trading engine...")
            self.state = EngineState.STARTING
            self._start_time = datetime.now()

            # Initialize components
            await self._initialize_components()

            # Start background tasks
            await self._start_background_tasks()

            # Set state to running
            self.state = EngineState.RUNNING

            await self._emit_event("engine_started", {"timestamp": self._start_time})
            logger.info("Trading engine started successfully")

        except Exception as e:
            self.state = EngineState.ERROR
            logger.error(f"Failed to start trading engine: {e}")
            raise TradingError(f"Engine startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop the trading engine."""
        try:
            logger.info("Stopping trading engine...")
            self.state = EngineState.STOPPING

            # Stop all strategies
            if self.strategy_manager:
                await self.strategy_manager.stop_all_strategies()

            # Cancel all pending orders
            if self.order_manager:
                await self.order_manager.cancel_all_orders()

            # Stop background tasks
            await self._stop_background_tasks()

            # Cleanup components
            await self._cleanup_components()

            self.state = EngineState.STOPPED
            await self._emit_event("engine_stopped", {"timestamp": datetime.now()})
            logger.info("Trading engine stopped successfully")

        except Exception as e:
            self.state = EngineState.ERROR
            logger.error(f"Error stopping trading engine: {e}")
            raise

    async def pause(self) -> None:
        """Pause the trading engine."""
        if self.state != EngineState.RUNNING:
            raise TradingError(f"Cannot pause engine in state: {self.state}")

        logger.info("Pausing trading engine...")
        self.state = EngineState.PAUSED

        # Pause all strategies
        if self.strategy_manager:
            await self.strategy_manager.pause_all_strategies()

        await self._emit_event("engine_paused", {"timestamp": datetime.now()})
        logger.info("Trading engine paused")

    async def resume(self) -> None:
        """Resume the trading engine."""
        if self.state != EngineState.PAUSED:
            raise TradingError(f"Cannot resume engine in state: {self.state}")

        logger.info("Resuming trading engine...")
        self.state = EngineState.RUNNING

        # Resume all strategies
        if self.strategy_manager:
            await self.strategy_manager.resume_all_strategies()

        await self._emit_event("engine_resumed", {"timestamp": datetime.now()})
        logger.info("Trading engine resumed")

    async def add_strategy(
        self,
        strategy_config: Dict[str, Any],
    ) -> str:
        """
        Add a new trading strategy.

        Args:
            strategy_config: Strategy configuration

        Returns:
            Strategy instance ID
        """
        if not self.strategy_manager:
            raise TradingError("Strategy manager not initialized")

        if len(self._strategy_instances) >= self.config.max_concurrent_strategies:
            raise TradingError("Maximum number of strategies reached")

        # Create and start strategy
        instance_id = await self.strategy_manager.create_strategy_instance(
            strategy_config
        )

        # Track the instance
        instance = await self.strategy_manager.get_strategy_instance(instance_id)
        if instance:
            self._strategy_instances[instance_id] = instance
            logger.info(f"Added strategy: {instance_id}")
            await self._emit_event("strategy_added", {
                "instance_id": instance_id,
                "config": strategy_config
            })

        return instance_id

    async def remove_strategy(self, instance_id: str) -> None:
        """
        Remove a trading strategy.

        Args:
            instance_id: Strategy instance ID
        """
        if not self.strategy_manager:
            raise TradingError("Strategy manager not initialized")

        if instance_id not in self._strategy_instances:
            raise ValidationError(f"Strategy not found: {instance_id}")

        # Stop and remove strategy
        await self.strategy_manager.stop_strategy_instance(instance_id)
        del self._strategy_instances[instance_id]

        logger.info(f"Removed strategy: {instance_id}")
        await self._emit_event("strategy_removed", {"instance_id": instance_id})

    async def execute_trade(
        self,
        symbol: str,
        action: str,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a trade.

        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            volume: Trade volume
            price: Limit price (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            strategy_id: Strategy instance ID (optional)

        Returns:
            Execution result
        """
        if not self.execution_engine:
            raise TradingError("Execution engine not initialized")

        if self.state not in [EngineState.RUNNING]:
            raise TradingError(f"Cannot execute trades in state: {self.state}")

        # Risk check
        if self.risk_manager and self.config.enable_risk_management:
            risk_check = await self.risk_manager.check_trade_risk(
                symbol=symbol,
                action=action,
                volume=volume,
                price=price
            )

            if not risk_check.approved:
                raise TradingError(f"Trade rejected by risk manager: {risk_check.reason}")

        # Execute the trade
        start_time = datetime.now()
        try:
            result = await self.execution_engine.execute_trade(
                symbol=symbol,
                action=action,
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={"strategy_id": strategy_id} if strategy_id else None
            )

            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._update_execution_metrics(result, execution_time)

            # Emit event
            await self._emit_event("trade_executed", {
                "result": result.dict(),
                "execution_time": execution_time
            })

            return result

        except Exception as e:
            self.metrics.failed_trades += 1
            logger.error(f"Trade execution failed: {e}")
            raise

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        if not self.portfolio_manager:
            return {}

        return await self.portfolio_manager.get_portfolio_summary()

    async def get_active_positions(self) -> List[Position]:
        """Get all active positions."""
        if not self.portfolio_manager:
            return []

        return await self.portfolio_manager.get_active_positions()

    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        if not self.order_manager:
            return []

        return await self.order_manager.get_active_orders()

    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategies."""
        if not self.strategy_manager:
            return {}

        return await self.strategy_manager.get_all_strategies_status()

    def get_metrics(self) -> EngineMetrics:
        """Get current engine metrics."""
        # Update uptime
        if self._start_time:
            self.metrics.uptime_seconds = (
                datetime.now() - self._start_time
            ).total_seconds()

        # Update active counts
        self.metrics.active_positions = len(self._active_positions)
        self.metrics.active_orders = len(self._active_orders)
        self.metrics.strategies_running = len(self._strategy_instances)

        # Update average execution time
        if self._execution_times:
            self.metrics.average_execution_time = sum(self._execution_times) / len(self._execution_times)

        self.metrics.last_updated = datetime.now()
        return self.metrics

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self.state

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "healthy": self.state == EngineState.RUNNING,
            "state": self.state.value,
            "uptime_seconds": self.metrics.uptime_seconds,
            "components": {}
        }

        # Check components
        if self.strategy_manager:
            health["components"]["strategy_manager"] = await self.strategy_manager.health_check()

        if self.execution_engine:
            health["components"]["execution_engine"] = await self.execution_engine.health_check()

        if self.portfolio_manager:
            health["components"]["portfolio_manager"] = await self.portfolio_manager.health_check()

        if self.risk_manager:
            health["components"]["risk_manager"] = await self.risk_manager.health_check()

        if self.order_manager:
            health["components"]["order_manager"] = await self.order_manager.health_check()

        return health

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def _initialize_components(self) -> None:
        """Initialize all engine components."""
        logger.info("Initializing trading engine components...")

        # Initialize strategy manager
        self.strategy_manager = StrategyManager(self.settings)
        await self.strategy_manager.initialize()

        # Initialize execution engine
        self.execution_engine = ExecutionEngine(self.settings)
        await self.execution_engine.initialize()

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(self.settings)
        await self.portfolio_manager.initialize()

        # Initialize risk manager
        if self.config.enable_risk_management:
            self.risk_manager = RiskManager(self.settings)
            await self.risk_manager.initialize()

        # Initialize order manager
        self.order_manager = OrderManager(self.settings)
        await self.order_manager.initialize()

        logger.info("All components initialized successfully")

    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        logger.info("Starting background tasks...")

        # Metrics update task
        task = asyncio.create_task(self._metrics_update_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        # Cleanup task
        task = asyncio.create_task(self._cleanup_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        # Position monitoring task
        task = asyncio.create_task(self._position_monitoring_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        logger.info("Background tasks started")

    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        logger.info("Stopping background tasks...")

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.info("Background tasks stopped")

    async def _cleanup_components(self) -> None:
        """Cleanup all components."""
        logger.info("Cleaning up components...")

        components = [
            self.strategy_manager,
            self.execution_engine,
            self.portfolio_manager,
            self.risk_manager,
            self.order_manager,
        ]

        for component in components:
            if component:
                try:
                    await component.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up component: {e}")

        logger.info("Components cleaned up")

    async def _metrics_update_loop(self) -> None:
        """Background task for updating metrics."""
        while self.state in [EngineState.RUNNING, EngineState.PAUSED]:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.config.metrics_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(self.config.metrics_update_interval)

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while self.state in [EngineState.RUNNING, EngineState.PAUSED]:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    async def _position_monitoring_loop(self) -> None:
        """Background task for position monitoring."""
        while self.state in [EngineState.RUNNING, EngineState.PAUSED]:
            try:
                await self._monitor_positions()
                await asyncio.sleep(self.config.position_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitoring loop: {e}")
                await asyncio.sleep(self.config.position_check_interval)

    async def _update_metrics(self) -> None:
        """Update engine metrics."""
        try:
            # Update component metrics
            if self.portfolio_manager:
                portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
                self.metrics.total_pnl = portfolio_summary.get("total_pnl", 0.0)
                self.metrics.active_positions = portfolio_summary.get("position_count", 0)

            if self.order_manager:
                order_stats = await self.order_manager.get_order_statistics()
                self.metrics.active_orders = order_stats.get("active_orders", 0)
                self.metrics.total_trades = order_stats.get("total_orders", 0)
                self.metrics.successful_trades = order_stats.get("filled_orders", 0)
                self.metrics.failed_trades = order_stats.get("rejected_orders", 0)

            if self.strategy_manager:
                strategy_stats = await self.strategy_manager.get_strategy_statistics()
                self.metrics.strategies_running = strategy_stats.get("active_strategies", 0)

            self._last_metrics_update = datetime.now()

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _perform_cleanup(self) -> None:
        """Perform periodic cleanup tasks."""
        try:
            # Clean up old execution times (keep last 1000)
            if len(self._execution_times) > 1000:
                self._execution_times = self._execution_times[-1000:]

            # Clean up completed orders and positions
            if self.order_manager:
                await self.order_manager.cleanup_completed_orders()

            if self.portfolio_manager:
                await self.portfolio_manager.cleanup_closed_positions()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _monitor_positions(self) -> None:
        """Monitor active positions for risk management."""
        try:
            if not self.portfolio_manager or not self.risk_manager:
                return

            positions = await self.portfolio_manager.get_active_positions()

            for position in positions:
                # Check position risk
                risk_check = await self.risk_manager.check_position_risk(position)

                if risk_check.requires_action:
                    logger.warning(f"Position risk violation: {risk_check.reason}")
                    await self._handle_risk_violation(position, risk_check)

        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    async def _handle_risk_violation(self, position: Position, risk_check: Any) -> None:
        """Handle risk violation for a position."""
        try:
            self.metrics.risk_violations += 1

            # Implement risk violation handling logic
            # This could include closing positions, reducing size, etc.
            logger.warning(f"Handling risk violation for position {position.id}")

            await self._emit_event("risk_violation", {
                "position_id": position.id,
                "violation_type": risk_check.violation_type,
                "reason": risk_check.reason
            })

        except Exception as e:
            logger.error(f"Error handling risk violation: {e}")

    async def _update_execution_metrics(self, result: ExecutionResult, execution_time: float) -> None:
        """Update execution-related metrics."""
        self._execution_times.append(execution_time)

        if result.success:
            self.metrics.successful_trades += 1
            if result.volume:
                self.metrics.total_volume += result.volume
        else:
            self.metrics.failed_trades += 1

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to all registered handlers."""
        try:
            if event_type in self._event_handlers:
                for handler in self._event_handlers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event_type, data)
                        else:
                            handler(event_type, data)
                    except Exception as e:
                        logger.error(f"Error in event handler {handler}: {e}")
        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")