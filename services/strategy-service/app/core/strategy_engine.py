"""
Strategy Engine - Manages strategy execution and lifecycle
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum
import asyncio

from .position_manager import PositionManager, PositionType
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy status"""
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class StrategyType(Enum):
    """Strategy types"""
    MA_CROSSOVER = "MA_CROSSOVER"
    RSI = "RSI"
    CUSTOM = "CUSTOM"


class BaseStrategy:
    """Base class for all trading strategies"""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        symbol: str,
        timeframe: str,
        position_manager: PositionManager,
        risk_manager: RiskManager
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.status = StrategyStatus.STOPPED
        self.created_at = datetime.now()
        self.started_at = None
        self.stopped_at = None
        self.parameters = {}
        self.signals_count = 0
        self.trades_count = 0

    def start(self):
        """Start the strategy"""
        self.status = StrategyStatus.RUNNING
        self.started_at = datetime.now()
        logger.info(f"Strategy {self.strategy_id} ({self.name}) started")

    def stop(self):
        """Stop the strategy"""
        self.status = StrategyStatus.STOPPED
        self.stopped_at = datetime.now()
        logger.info(f"Strategy {self.strategy_id} ({self.name}) stopped")

    def pause(self):
        """Pause the strategy"""
        self.status = StrategyStatus.PAUSED
        logger.info(f"Strategy {self.strategy_id} ({self.name}) paused")

    def resume(self):
        """Resume the strategy"""
        self.status = StrategyStatus.RUNNING
        logger.info(f"Strategy {self.strategy_id} ({self.name}) resumed")

    async def on_tick(self, tick_data: dict):
        """Called on each price tick (override in subclass)"""
        pass

    async def on_bar(self, bar_data: dict):
        """Called on each new bar/candle (override in subclass)"""
        pass

    def get_status(self) -> dict:
        """Get strategy status"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "signals_count": self.signals_count,
            "trades_count": self.trades_count,
            "parameters": self.parameters
        }


class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover Strategy"""

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        fast_period: int = 10,
        slow_period: int = 20,
        stop_loss_pips: float = 50.0,
        take_profit_pips: float = 100.0
    ):
        super().__init__(
            strategy_id=strategy_id,
            name="MA Crossover",
            symbol=symbol,
            timeframe=timeframe,
            position_manager=position_manager,
            risk_manager=risk_manager
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.price_history = []
        self.last_signal = None

        self.parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "stop_loss_pips": stop_loss_pips,
            "take_profit_pips": take_profit_pips
        }

    def calculate_ma(self, prices: list, period: int) -> Optional[float]:
        """Calculate simple moving average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    async def on_bar(self, bar_data: dict):
        """Process new bar data"""
        if self.status != StrategyStatus.RUNNING:
            return

        close_price = bar_data.get('close')
        if not close_price:
            return

        # Update price history
        self.price_history.append(close_price)
        if len(self.price_history) > self.slow_period + 10:
            self.price_history.pop(0)

        # Calculate MAs
        fast_ma = self.calculate_ma(self.price_history, self.fast_period)
        slow_ma = self.calculate_ma(self.price_history, self.slow_period)

        if fast_ma is None or slow_ma is None:
            return

        # Detect crossover
        signal = None
        if fast_ma > slow_ma and self.last_signal != "BUY":
            signal = "BUY"
            self.signals_count += 1
            logger.info(f"MA Crossover BUY signal: Fast MA {fast_ma:.5f} > Slow MA {slow_ma:.5f}")

        elif fast_ma < slow_ma and self.last_signal != "SELL":
            signal = "SELL"
            self.signals_count += 1
            logger.info(f"MA Crossover SELL signal: Fast MA {fast_ma:.5f} < Slow MA {slow_ma:.5f}")

        if signal:
            self.last_signal = signal
            await self.execute_trade(signal, close_price)

    async def execute_trade(self, signal: str, current_price: float):
        """Execute trade based on signal"""
        # Check if we can open a position
        open_positions = self.position_manager.get_open_positions_count(strategy_id=self.strategy_id)

        # Calculate SL and TP
        if signal == "BUY":
            stop_loss = current_price - (self.stop_loss_pips / 10000)
            take_profit = current_price + (self.take_profit_pips / 10000)
            position_type = PositionType.BUY
        else:
            stop_loss = current_price + (self.stop_loss_pips / 10000)
            take_profit = current_price - (self.take_profit_pips / 10000)
            position_type = PositionType.SELL

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.symbol, current_price, stop_loss
        )

        # Check if position can be opened
        can_open, reason = self.risk_manager.can_open_position(
            open_positions, self.symbol, position_size
        )

        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return

        # Open position
        position = self.position_manager.open_position(
            symbol=self.symbol,
            position_type=position_type,
            volume=position_size,
            open_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=self.strategy_id
        )

        self.trades_count += 1
        logger.info(f"Trade executed: {position.position_id}")


class RSIStrategy(BaseStrategy):
    """RSI-based Strategy"""

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        rsi_period: int = 14,
        oversold_level: float = 30.0,
        overbought_level: float = 70.0,
        stop_loss_pips: float = 50.0,
        take_profit_pips: float = 100.0
    ):
        super().__init__(
            strategy_id=strategy_id,
            name="RSI Strategy",
            symbol=symbol,
            timeframe=timeframe,
            position_manager=position_manager,
            risk_manager=risk_manager
        )
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.price_history = []
        self.last_signal = None

        self.parameters = {
            "rsi_period": rsi_period,
            "oversold_level": oversold_level,
            "overbought_level": overbought_level,
            "stop_loss_pips": stop_loss_pips,
            "take_profit_pips": take_profit_pips
        }

    def calculate_rsi(self, prices: list, period: int) -> Optional[float]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return None

        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(change, 0) for change in changes[-period:]]
        losses = [abs(min(change, 0)) for change in changes[-period:]]

        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def on_bar(self, bar_data: dict):
        """Process new bar data"""
        if self.status != StrategyStatus.RUNNING:
            return

        close_price = bar_data.get('close')
        if not close_price:
            return

        # Update price history
        self.price_history.append(close_price)
        if len(self.price_history) > self.rsi_period + 20:
            self.price_history.pop(0)

        # Calculate RSI
        rsi = self.calculate_rsi(self.price_history, self.rsi_period)

        if rsi is None:
            return

        # Detect signals
        signal = None
        if rsi < self.oversold_level and self.last_signal != "BUY":
            signal = "BUY"
            self.signals_count += 1
            logger.info(f"RSI BUY signal: RSI {rsi:.2f} < {self.oversold_level} (oversold)")

        elif rsi > self.overbought_level and self.last_signal != "SELL":
            signal = "SELL"
            self.signals_count += 1
            logger.info(f"RSI SELL signal: RSI {rsi:.2f} > {self.overbought_level} (overbought)")

        if signal:
            self.last_signal = signal
            await self.execute_trade(signal, close_price)

    async def execute_trade(self, signal: str, current_price: float):
        """Execute trade based on signal"""
        # Check if we can open a position
        open_positions = self.position_manager.get_open_positions_count(strategy_id=self.strategy_id)

        # Calculate SL and TP
        if signal == "BUY":
            stop_loss = current_price - (self.stop_loss_pips / 10000)
            take_profit = current_price + (self.take_profit_pips / 10000)
            position_type = PositionType.BUY
        else:
            stop_loss = current_price + (self.stop_loss_pips / 10000)
            take_profit = current_price - (self.take_profit_pips / 10000)
            position_type = PositionType.SELL

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.symbol, current_price, stop_loss
        )

        # Check if position can be opened
        can_open, reason = self.risk_manager.can_open_position(
            open_positions, self.symbol, position_size
        )

        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return

        # Open position
        position = self.position_manager.open_position(
            symbol=self.symbol,
            position_type=position_type,
            volume=position_size,
            open_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=self.strategy_id
        )

        self.trades_count += 1
        logger.info(f"Trade executed: {position.position_id}")


class StrategyEngine:
    """Manages all trading strategies"""

    def __init__(self, position_manager: PositionManager, risk_manager: RiskManager):
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_counter = 0
        self.mock_mode = True
        logger.info("Strategy Engine initialized in mock mode")

    def _generate_strategy_id(self) -> str:
        """Generate unique strategy ID"""
        self.strategy_counter += 1
        return f"STRAT_{self.strategy_counter:04d}"

    def create_strategy(
        self,
        strategy_type: str,
        symbol: str,
        timeframe: str,
        parameters: dict
    ) -> BaseStrategy:
        """Create a new strategy"""
        strategy_id = self._generate_strategy_id()

        if strategy_type == "MA_CROSSOVER":
            strategy = MovingAverageCrossover(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                position_manager=self.position_manager,
                risk_manager=self.risk_manager,
                fast_period=parameters.get('fast_period', 10),
                slow_period=parameters.get('slow_period', 20),
                stop_loss_pips=parameters.get('stop_loss_pips', 50.0),
                take_profit_pips=parameters.get('take_profit_pips', 100.0)
            )
        elif strategy_type == "RSI":
            strategy = RSIStrategy(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                position_manager=self.position_manager,
                risk_manager=self.risk_manager,
                rsi_period=parameters.get('rsi_period', 14),
                oversold_level=parameters.get('oversold_level', 30.0),
                overbought_level=parameters.get('overbought_level', 70.0),
                stop_loss_pips=parameters.get('stop_loss_pips', 50.0),
                take_profit_pips=parameters.get('take_profit_pips', 100.0)
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        self.strategies[strategy_id] = strategy
        logger.info(f"Strategy created: {strategy_id} ({strategy_type})")

        return strategy

    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Get a strategy by ID"""
        return self.strategies.get(strategy_id)

    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all strategies"""
        return list(self.strategies.values())

    def start_strategy(self, strategy_id: str) -> bool:
        """Start a strategy"""
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            logger.error(f"Strategy {strategy_id} not found")
            return False

        strategy.start()
        return True

    def stop_strategy(self, strategy_id: str, close_positions: bool = True) -> bool:
        """Stop a strategy"""
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            logger.error(f"Strategy {strategy_id} not found")
            return False

        strategy.stop()

        # Close all positions for this strategy if requested
        if close_positions:
            # In mock mode, use dummy price
            current_prices = {strategy.symbol: 1.1000}
            closed_count = self.position_manager.close_positions_by_strategy(
                strategy_id, current_prices
            )
            logger.info(f"Closed {closed_count} positions for strategy {strategy_id}")

        return True

    def pause_strategy(self, strategy_id: str) -> bool:
        """Pause a strategy"""
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            logger.error(f"Strategy {strategy_id} not found")
            return False

        strategy.pause()
        return True

    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume a strategy"""
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            logger.error(f"Strategy {strategy_id} not found")
            return False

        strategy.resume()
        return True

    def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy"""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False

        # Stop strategy first
        self.stop_strategy(strategy_id, close_positions=True)

        # Delete strategy
        del self.strategies[strategy_id]
        logger.info(f"Strategy {strategy_id} deleted")

        return True

    async def update_strategies(self, bar_data: dict):
        """Update all running strategies with new bar data"""
        for strategy in self.strategies.values():
            if strategy.status == StrategyStatus.RUNNING:
                await strategy.on_bar(bar_data)

    def get_statistics(self) -> dict:
        """Get strategy engine statistics"""
        running_count = sum(1 for s in self.strategies.values() if s.status == StrategyStatus.RUNNING)
        stopped_count = sum(1 for s in self.strategies.values() if s.status == StrategyStatus.STOPPED)
        paused_count = sum(1 for s in self.strategies.values() if s.status == StrategyStatus.PAUSED)

        return {
            "total_strategies": len(self.strategies),
            "running_strategies": running_count,
            "stopped_strategies": stopped_count,
            "paused_strategies": paused_count,
            "mock_mode": self.mock_mode
        }
