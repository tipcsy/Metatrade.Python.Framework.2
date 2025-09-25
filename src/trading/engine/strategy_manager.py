"""
Strategy Manager Implementation.

This module manages trading strategies, their lifecycle, and execution coordination.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type
from uuid import uuid4
import importlib
import inspect

from pydantic import BaseModel, Field, ConfigDict

from ...core.config import Settings
from ...core.exceptions import TradingError, ValidationError
from ...core.logging import get_logger
from ...database.models import Strategy, StrategySession, StrategyParameter
from ...database.services import BaseService

logger = get_logger(__name__)


class StrategyStatus(str, Enum):
    """Strategy execution status."""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    total_signals: int = 0
    executed_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    execution_time: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class StrategyConfig(BaseModel):
    """Strategy configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic settings
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    strategy_class: str = Field(..., min_length=1)  # Full class path
    enabled: bool = Field(default=True)

    # Execution settings
    symbols: List[str] = Field(default_factory=list)
    timeframes: List[str] = Field(default_factory=list)
    max_positions: int = Field(default=5, ge=1)
    max_risk_per_trade: float = Field(default=0.02, gt=0, le=1)

    # Parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Risk management
    stop_loss_pct: Optional[float] = Field(default=None, gt=0, le=1)
    take_profit_pct: Optional[float] = Field(default=None, gt=0)
    max_drawdown_pct: float = Field(default=0.2, gt=0, le=1)

    # Scheduling
    start_time: Optional[str] = None  # Format: "HH:MM"
    end_time: Optional[str] = None    # Format: "HH:MM"
    trading_days: List[int] = Field(default_factory=lambda: list(range(7)))  # 0=Monday

    # Performance settings
    warmup_period: int = Field(default=100, ge=0)  # Number of bars
    timeout_seconds: int = Field(default=300, ge=1)


class StrategyInstance:
    """Represents a running strategy instance."""

    def __init__(
        self,
        instance_id: str,
        config: StrategyConfig,
        strategy_class: Type,
    ):
        """Initialize strategy instance."""
        self.instance_id = instance_id
        self.config = config
        self.strategy_class = strategy_class
        self.status = StrategyStatus.CREATED
        self.metrics = StrategyMetrics()

        # Runtime state
        self.strategy_object: Optional[Any] = None
        self.task: Optional[asyncio.Task] = None
        self.start_time: Optional[datetime] = None
        self.last_signal_time: Optional[datetime] = None
        self.error_count = 0
        self.max_errors = 10

        # Data and context
        self.market_data: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}

        logger.info(f"Created strategy instance: {instance_id}")

    async def initialize(self) -> None:
        """Initialize the strategy instance."""
        try:
            self.status = StrategyStatus.INITIALIZING
            logger.info(f"Initializing strategy: {self.instance_id}")

            # Create strategy object
            self.strategy_object = self.strategy_class(
                config=self.config.parameters,
                instance_id=self.instance_id
            )

            # Initialize strategy if it has an initialize method
            if hasattr(self.strategy_object, 'initialize'):
                if asyncio.iscoroutinefunction(self.strategy_object.initialize):
                    await self.strategy_object.initialize()
                else:
                    self.strategy_object.initialize()

            self.status = StrategyStatus.RUNNING
            self.start_time = datetime.now()
            logger.info(f"Strategy initialized: {self.instance_id}")

        except Exception as e:
            self.status = StrategyStatus.ERROR
            self.error_count += 1
            logger.error(f"Failed to initialize strategy {self.instance_id}: {e}")
            raise

    async def start(self) -> None:
        """Start the strategy execution."""
        if self.status != StrategyStatus.RUNNING:
            raise TradingError(f"Cannot start strategy in status: {self.status}")

        logger.info(f"Starting strategy execution: {self.instance_id}")

        # Create execution task
        self.task = asyncio.create_task(self._execution_loop())

    async def stop(self) -> None:
        """Stop the strategy execution."""
        logger.info(f"Stopping strategy: {self.instance_id}")

        self.status = StrategyStatus.STOPPED

        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        # Cleanup strategy object
        if self.strategy_object and hasattr(self.strategy_object, 'cleanup'):
            try:
                if asyncio.iscoroutinefunction(self.strategy_object.cleanup):
                    await self.strategy_object.cleanup()
                else:
                    self.strategy_object.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up strategy {self.instance_id}: {e}")

        logger.info(f"Strategy stopped: {self.instance_id}")

    async def pause(self) -> None:
        """Pause the strategy execution."""
        if self.status != StrategyStatus.RUNNING:
            raise TradingError(f"Cannot pause strategy in status: {self.status}")

        self.status = StrategyStatus.PAUSED
        logger.info(f"Strategy paused: {self.instance_id}")

    async def resume(self) -> None:
        """Resume the strategy execution."""
        if self.status != StrategyStatus.PAUSED:
            raise TradingError(f"Cannot resume strategy in status: {self.status}")

        self.status = StrategyStatus.RUNNING
        logger.info(f"Strategy resumed: {self.instance_id}")

    async def update_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update market data for the strategy."""
        self.market_data[symbol] = data

        # Process the data if strategy is running
        if self.status == StrategyStatus.RUNNING and self.strategy_object:
            try:
                await self._process_market_data(symbol, data)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error processing market data in {self.instance_id}: {e}")

                if self.error_count >= self.max_errors:
                    self.status = StrategyStatus.ERROR
                    logger.error(f"Strategy {self.instance_id} disabled due to too many errors")

    async def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "instance_id": self.instance_id,
            "name": self.config.name,
            "status": self.status.value,
            "uptime_seconds": uptime,
            "error_count": self.error_count,
            "metrics": {
                "total_signals": self.metrics.total_signals,
                "executed_trades": self.metrics.executed_trades,
                "successful_trades": self.metrics.successful_trades,
                "total_pnl": self.metrics.total_pnl,
                "win_rate": self.metrics.win_rate,
            },
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
        }

    async def _execution_loop(self) -> None:
        """Main strategy execution loop."""
        logger.info(f"Starting execution loop for strategy: {self.instance_id}")

        try:
            while self.status in [StrategyStatus.RUNNING, StrategyStatus.PAUSED]:
                try:
                    # Skip execution if paused
                    if self.status == StrategyStatus.PAUSED:
                        await asyncio.sleep(1)
                        continue

                    # Check if it's time to trade
                    if not self._is_trading_time():
                        await asyncio.sleep(60)  # Check again in 1 minute
                        continue

                    # Execute strategy logic
                    await self._execute_strategy_cycle()

                    # Sleep for a short interval
                    await asyncio.sleep(1)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error in execution loop for {self.instance_id}: {e}")

                    if self.error_count >= self.max_errors:
                        self.status = StrategyStatus.ERROR
                        break

                    await asyncio.sleep(5)  # Wait before retrying

        except Exception as e:
            logger.error(f"Fatal error in execution loop for {self.instance_id}: {e}")
            self.status = StrategyStatus.ERROR

        logger.info(f"Execution loop ended for strategy: {self.instance_id}")

    async def _execute_strategy_cycle(self) -> None:
        """Execute one cycle of the strategy."""
        if not self.strategy_object:
            return

        start_time = datetime.now()

        try:
            # Call strategy's on_tick or execute method
            if hasattr(self.strategy_object, 'on_tick'):
                if asyncio.iscoroutinefunction(self.strategy_object.on_tick):
                    signals = await self.strategy_object.on_tick(self.market_data)
                else:
                    signals = self.strategy_object.on_tick(self.market_data)
            elif hasattr(self.strategy_object, 'execute'):
                if asyncio.iscoroutinefunction(self.strategy_object.execute):
                    signals = await self.strategy_object.execute(self.market_data)
                else:
                    signals = self.strategy_object.execute(self.market_data)
            else:
                return  # No execution method found

            # Process signals
            if signals:
                await self._process_signals(signals)

            # Update execution time metric
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics.execution_time = (
                self.metrics.execution_time * 0.9 + execution_time * 0.1
            )  # Exponential moving average

        except Exception as e:
            logger.error(f"Error executing strategy cycle for {self.instance_id}: {e}")
            raise

    async def _process_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Process incoming market data."""
        if not self.strategy_object:
            return

        try:
            # Call strategy's on_data method if available
            if hasattr(self.strategy_object, 'on_data'):
                if asyncio.iscoroutinefunction(self.strategy_object.on_data):
                    await self.strategy_object.on_data(symbol, data)
                else:
                    self.strategy_object.on_data(symbol, data)

        except Exception as e:
            logger.error(f"Error processing market data for {self.instance_id}: {e}")
            raise

    async def _process_signals(self, signals: List[Dict[str, Any]]) -> None:
        """Process trading signals generated by the strategy."""
        if not signals:
            return

        self.metrics.total_signals += len(signals)
        self.last_signal_time = datetime.now()

        logger.info(f"Processing {len(signals)} signals from strategy {self.instance_id}")

        # Here you would typically send signals to the execution engine
        # This is a simplified version - in reality you'd integrate with the trading engine
        for signal in signals:
            try:
                logger.info(f"Signal generated: {signal}")
                # Process each signal...

            except Exception as e:
                logger.error(f"Error processing signal: {e}")

    def _is_trading_time(self) -> bool:
        """Check if it's currently trading time based on configuration."""
        now = datetime.now()

        # Check trading days
        if now.weekday() not in self.config.trading_days:
            return False

        # Check trading hours
        if self.config.start_time and self.config.end_time:
            try:
                start_hour, start_minute = map(int, self.config.start_time.split(':'))
                end_hour, end_minute = map(int, self.config.end_time.split(':'))

                start_time = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                end_time = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

                if start_time <= now <= end_time:
                    return True
                else:
                    return False
            except (ValueError, AttributeError):
                logger.warning(f"Invalid time format in strategy config: {self.config.start_time}-{self.config.end_time}")

        return True  # Default to always trading if no time restrictions


class StrategyManager(BaseService):
    """
    Strategy Manager for coordinating multiple trading strategies.

    This manager handles strategy lifecycle, execution coordination,
    and performance monitoring.
    """

    def __init__(self, settings: Settings):
        """Initialize the strategy manager."""
        super().__init__(settings)
        self._instances: Dict[str, StrategyInstance] = {}
        self._strategy_classes: Dict[str, Type] = {}

    async def initialize(self) -> None:
        """Initialize the strategy manager."""
        logger.info("Initializing strategy manager...")

        # Discover and register strategy classes
        await self._discover_strategies()

        logger.info("Strategy manager initialized")

    async def create_strategy_instance(
        self,
        config_dict: Dict[str, Any],
    ) -> str:
        """
        Create a new strategy instance.

        Args:
            config_dict: Strategy configuration dictionary

        Returns:
            Strategy instance ID
        """
        try:
            # Validate and create config
            config = StrategyConfig(**config_dict)

            # Check if strategy class exists
            if config.strategy_class not in self._strategy_classes:
                raise ValidationError(f"Strategy class not found: {config.strategy_class}")

            # Create instance
            instance_id = str(uuid4())
            strategy_class = self._strategy_classes[config.strategy_class]

            instance = StrategyInstance(instance_id, config, strategy_class)

            # Initialize the instance
            await instance.initialize()

            # Store instance
            self._instances[instance_id] = instance

            # Start execution
            await instance.start()

            logger.info(f"Created and started strategy instance: {instance_id}")
            return instance_id

        except Exception as e:
            logger.error(f"Failed to create strategy instance: {e}")
            raise

    async def get_strategy_instance(self, instance_id: str) -> Optional[StrategyInstance]:
        """Get strategy instance by ID."""
        return self._instances.get(instance_id)

    async def stop_strategy_instance(self, instance_id: str) -> None:
        """Stop a strategy instance."""
        if instance_id not in self._instances:
            raise ValidationError(f"Strategy instance not found: {instance_id}")

        instance = self._instances[instance_id]
        await instance.stop()

        logger.info(f"Stopped strategy instance: {instance_id}")

    async def pause_strategy_instance(self, instance_id: str) -> None:
        """Pause a strategy instance."""
        if instance_id not in self._instances:
            raise ValidationError(f"Strategy instance not found: {instance_id}")

        await self._instances[instance_id].pause()

    async def resume_strategy_instance(self, instance_id: str) -> None:
        """Resume a strategy instance."""
        if instance_id not in self._instances:
            raise ValidationError(f"Strategy instance not found: {instance_id}")

        await self._instances[instance_id].resume()

    async def stop_all_strategies(self) -> None:
        """Stop all strategy instances."""
        logger.info("Stopping all strategies...")

        tasks = []
        for instance in self._instances.values():
            tasks.append(instance.stop())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All strategies stopped")

    async def pause_all_strategies(self) -> None:
        """Pause all strategy instances."""
        logger.info("Pausing all strategies...")

        for instance in self._instances.values():
            try:
                await instance.pause()
            except Exception as e:
                logger.error(f"Error pausing strategy {instance.instance_id}: {e}")

    async def resume_all_strategies(self) -> None:
        """Resume all strategy instances."""
        logger.info("Resuming all strategies...")

        for instance in self._instances.values():
            try:
                await instance.resume()
            except Exception as e:
                logger.error(f"Error resuming strategy {instance.instance_id}: {e}")

    async def update_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update market data for all strategies."""
        for instance in self._instances.values():
            if symbol in instance.config.symbols:
                try:
                    await instance.update_market_data(symbol, data)
                except Exception as e:
                    logger.error(f"Error updating market data for strategy {instance.instance_id}: {e}")

    async def get_all_strategies_status(self) -> Dict[str, Any]:
        """Get status of all strategies."""
        strategies = {}

        for instance_id, instance in self._instances.items():
            try:
                strategies[instance_id] = await instance.get_status()
            except Exception as e:
                logger.error(f"Error getting status for strategy {instance_id}: {e}")
                strategies[instance_id] = {
                    "instance_id": instance_id,
                    "status": "error",
                    "error": str(e)
                }

        return {
            "total_strategies": len(self._instances),
            "running_strategies": len([s for s in strategies.values() if s.get("status") == "running"]),
            "strategies": strategies
        }

    async def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get overall strategy statistics."""
        total = len(self._instances)
        running = len([i for i in self._instances.values() if i.status == StrategyStatus.RUNNING])
        paused = len([i for i in self._instances.values() if i.status == StrategyStatus.PAUSED])
        error = len([i for i in self._instances.values() if i.status == StrategyStatus.ERROR])

        total_signals = sum(i.metrics.total_signals for i in self._instances.values())
        total_trades = sum(i.metrics.executed_trades for i in self._instances.values())
        total_pnl = sum(i.metrics.total_pnl for i in self._instances.values())

        return {
            "active_strategies": running,
            "total_strategies": total,
            "paused_strategies": paused,
            "error_strategies": error,
            "total_signals": total_signals,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        healthy = True
        details = {}

        for instance_id, instance in self._instances.items():
            instance_healthy = instance.status in [StrategyStatus.RUNNING, StrategyStatus.PAUSED]
            if not instance_healthy:
                healthy = False

            details[instance_id] = {
                "healthy": instance_healthy,
                "status": instance.status.value,
                "error_count": instance.error_count,
            }

        return {
            "healthy": healthy,
            "total_strategies": len(self._instances),
            "strategy_details": details,
        }

    async def cleanup(self) -> None:
        """Cleanup strategy manager resources."""
        logger.info("Cleaning up strategy manager...")
        await self.stop_all_strategies()
        self._instances.clear()
        logger.info("Strategy manager cleaned up")

    async def _discover_strategies(self) -> None:
        """Discover and register available strategy classes."""
        logger.info("Discovering strategy classes...")

        # This is a simplified implementation
        # In a real system, you might scan specific directories or packages
        # for strategy classes and register them automatically

        # For now, we'll register some basic strategy classes
        # You would extend this to auto-discover strategies from your strategy modules

        try:
            # Example strategy registration
            # You would implement actual strategy discovery logic here
            logger.info("Strategy discovery completed")

        except Exception as e:
            logger.error(f"Error during strategy discovery: {e}")

    def register_strategy_class(self, class_name: str, strategy_class: Type) -> None:
        """Register a strategy class."""
        self._strategy_classes[class_name] = strategy_class
        logger.info(f"Registered strategy class: {class_name}")

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy classes."""
        return list(self._strategy_classes.keys())