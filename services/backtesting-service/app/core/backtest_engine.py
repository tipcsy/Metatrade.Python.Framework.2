"""
Backtest Engine - Main orchestrator for backtesting

This module coordinates all backtesting components:
- Fetches historical data from Data Service
- Runs strategy simulation using Time Machine
- Manages positions using Position Simulator
- Calculates performance using Performance Calculator
- Stores results for later retrieval
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import uuid
import json
import requests

from .time_machine import TimeMachine, BarBuilder
from .position_simulator import PositionSimulator, PositionType, ExitReason
from .performance_calculator import PerformanceCalculator
from ..models.schemas import (
    BacktestStatus,
    StrategyType,
    BacktestRequest,
    BacktestResult,
    PerformanceMetrics,
    TradeRecord,
    EquityPoint
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the entire backtest process.
    """

    def __init__(self, data_service_url: str = "http://localhost:5001"):
        """
        Initialize backtest engine.

        Args:
            data_service_url: URL of the Data Service
        """
        self.data_service_url = data_service_url
        self.active_backtests: Dict[str, Dict[str, Any]] = {}
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        self.cancelled_backtests: set = set()

        logger.info(f"BacktestEngine initialized (Data Service: {data_service_url})")

    async def run_backtest(self, request: BacktestRequest) -> str:
        """
        Run a complete backtest.

        Args:
            request: Backtest request parameters

        Returns:
            Backtest ID
        """
        backtest_id = str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(
            f"Starting backtest {backtest_id}: "
            f"{request.strategy_type.value} on {request.symbol} {request.timeframe}"
        )

        # Initialize backtest record
        backtest_record = {
            "backtest_id": backtest_id,
            "status": BacktestStatus.RUNNING,
            "strategy_type": request.strategy_type,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "from_time": datetime.fromtimestamp(request.from_time),
            "to_time": datetime.fromtimestamp(request.to_time),
            "parameters": request.parameters,
            "initial_balance": request.initial_balance,
            "commission": request.commission,
            "spread_pips": request.spread_pips,
            "slippage_pips": request.slippage_pips,
            "position_size": request.position_size,
            "created_at": start_time,
            "started_at": start_time,
            "completed_at": None,
            "execution_time_seconds": None,
            "performance": None,
            "trades": None,
            "equity_curve": None,
            "error_message": None
        }

        self.active_backtests[backtest_id] = backtest_record

        # Run backtest asynchronously
        asyncio.create_task(self._execute_backtest(backtest_id, request))

        return backtest_id

    async def _execute_backtest(self, backtest_id: str, request: BacktestRequest):
        """
        Execute backtest in background.

        Args:
            backtest_id: Backtest ID
            request: Backtest parameters
        """
        try:
            # Step 1: Fetch historical data
            logger.info(f"[{backtest_id}] Fetching historical data...")
            bars = await self._fetch_historical_data(
                request.symbol,
                request.timeframe,
                request.from_time,
                request.to_time
            )

            if not bars:
                raise ValueError("No historical data available for the specified period")

            logger.info(f"[{backtest_id}] Loaded {len(bars)} bars")

            # Step 2: Initialize components
            time_machine = TimeMachine(bars)

            position_simulator = PositionSimulator(
                initial_balance=request.initial_balance,
                commission=request.commission,
                spread_pips=request.spread_pips,
                slippage_pips=request.slippage_pips
            )

            # Step 3: Initialize strategy
            strategy = self._create_strategy(
                request.strategy_type,
                request.parameters,
                position_simulator
            )

            # Step 4: Run simulation
            logger.info(f"[{backtest_id}] Running simulation...")

            equity_curve = []

            async def on_bar(bar: Dict[str, Any], bar_index: int):
                """Process each bar"""
                # Update open positions (check SL/TP)
                position_simulator.update_positions(bar['time'], bar)

                # Execute strategy logic
                await strategy.on_bar(bar)

                # Record equity curve
                current_equity = position_simulator.get_equity(bar['close'])
                running_max = max([eq['equity'] for eq in equity_curve] + [request.initial_balance])
                drawdown = running_max - current_equity
                drawdown_percent = (drawdown / running_max * 100) if running_max > 0 else 0.0

                equity_curve.append({
                    'timestamp': datetime.fromtimestamp(bar['time']),
                    'balance': position_simulator.balance,
                    'equity': current_equity,
                    'drawdown': drawdown,
                    'drawdown_percent': drawdown_percent
                })

            # Run time machine simulation
            await time_machine.run_simulation(on_bar)

            # Step 5: Close all remaining positions
            if len(bars) > 0:
                last_bar = bars[-1]
                position_simulator.close_all_positions(
                    last_bar['time'],
                    last_bar['close'],
                    ExitReason.END_OF_DATA
                )

            # Step 6: Calculate performance metrics
            logger.info(f"[{backtest_id}] Calculating performance metrics...")

            trades = position_simulator.get_closed_trades()

            performance_calc = PerformanceCalculator(
                initial_balance=request.initial_balance,
                trades=trades,
                equity_curve=equity_curve
            )

            metrics = performance_calc.calculate_all_metrics()

            # Step 7: Update backtest record
            end_time = datetime.now()
            execution_time = (end_time - self.active_backtests[backtest_id]['started_at']).total_seconds()

            self.active_backtests[backtest_id].update({
                "status": BacktestStatus.COMPLETED,
                "completed_at": end_time,
                "execution_time_seconds": execution_time,
                "performance": PerformanceMetrics(**metrics),
                "trades": [TradeRecord(**trade) for trade in trades],
                "equity_curve": [EquityPoint(**point) for point in equity_curve]
            })

            logger.info(
                f"[{backtest_id}] Backtest completed in {execution_time:.2f}s: "
                f"Trades={metrics['total_trades']}, "
                f"Net Profit=${metrics['net_profit']:.2f}, "
                f"Win Rate={metrics['win_rate']:.1f}%"
            )

        except Exception as e:
            logger.error(f"[{backtest_id}] Backtest failed: {str(e)}", exc_info=True)

            self.active_backtests[backtest_id].update({
                "status": BacktestStatus.FAILED,
                "completed_at": datetime.now(),
                "error_message": str(e)
            })

    async def _fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        from_time: int,
        to_time: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical OHLC data from Data Service.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            from_time: Start timestamp
            to_time: End timestamp

        Returns:
            List of OHLC bars
        """
        try:
            url = f"{self.data_service_url}/ohlc/query"
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "from_time": from_time,
                "to_time": to_time,
                "limit": 100000  # Large limit for backtesting
            }

            # Use asyncio to run synchronous request
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=30)
            )

            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                raise ValueError(f"Data Service error: {data.get('error', 'Unknown error')}")

            bars_raw = data['data']['bars']

            # Convert to standardized format
            bars = BarBuilder.from_dict_list(bars_raw)

            return bars

        except requests.RequestException as e:
            logger.error(f"Failed to fetch data from Data Service: {e}")
            raise ValueError(f"Failed to fetch historical data: {str(e)}")

    def _create_strategy(
        self,
        strategy_type: StrategyType,
        parameters: Dict[str, Any],
        position_simulator: PositionSimulator
    ):
        """
        Create strategy instance.

        Args:
            strategy_type: Type of strategy
            parameters: Strategy parameters
            position_simulator: Position simulator instance

        Returns:
            Strategy instance
        """
        if strategy_type == StrategyType.MA_CROSSOVER:
            return MACrossoverBacktestStrategy(parameters, position_simulator)
        elif strategy_type == StrategyType.RSI:
            return RSIBacktestStrategy(parameters, position_simulator)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

    def get_backtest(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        Get backtest results.

        Args:
            backtest_id: Backtest ID

        Returns:
            Backtest result or None if not found
        """
        if backtest_id not in self.active_backtests:
            return None

        record = self.active_backtests[backtest_id]

        return BacktestResult(**record)

    def delete_backtest(self, backtest_id: str) -> bool:
        """
        Delete backtest results.

        Args:
            backtest_id: Backtest ID

        Returns:
            True if deleted, False if not found
        """
        if backtest_id in self.active_backtests:
            del self.active_backtests[backtest_id]
            logger.info(f"Backtest {backtest_id} deleted")
            return True

        return False

    def list_backtests(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        strategy_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all backtests with filtering.

        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            status: Filter by status
            strategy_type: Filter by strategy type

        Returns:
            List of backtest summaries
        """
        summaries = []

        for backtest_id, record in self.active_backtests.items():
            # Apply filters
            if status and record['status'] != status:
                continue
            if strategy_type and record['strategy_type'] != strategy_type:
                continue

            summary = {
                "backtest_id": backtest_id,
                "status": record['status'],
                "strategy_type": record['strategy_type'],
                "symbol": record['symbol'],
                "timeframe": record['timeframe'],
                "created_at": record['created_at']
            }

            if record['performance']:
                summary.update({
                    "net_profit": record['performance'].net_profit,
                    "total_trades": record['performance'].total_trades,
                    "win_rate": record['performance'].win_rate
                })

            summaries.append(summary)

        # Sort by created_at descending
        summaries.sort(key=lambda x: x['created_at'], reverse=True)

        # Apply pagination
        return summaries[offset:offset + limit]

    def stop_backtest(self, backtest_id: str) -> bool:
        """
        Stop a running backtest.

        Args:
            backtest_id: Backtest ID

        Returns:
            True if stopped, False if not found or not running
        """
        if backtest_id not in self.active_backtests:
            return False

        record = self.active_backtests[backtest_id]

        if record['status'] != BacktestStatus.RUNNING:
            return False

        # Mark as cancelled
        self.cancelled_backtests.add(backtest_id)
        record['status'] = BacktestStatus.CANCELLED
        record['completed_at'] = datetime.now()

        logger.info(f"Backtest {backtest_id} stopped")
        return True

    async def run_optimization(self, request) -> str:
        """
        Run parameter optimization.

        Args:
            request: Optimization request

        Returns:
            Optimization ID
        """
        optimization_id = str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(
            f"Starting optimization {optimization_id}: "
            f"{request.strategy_type.value} on {request.symbol} {request.timeframe}"
        )

        # Initialize optimization record
        optimization_record = {
            "optimization_id": optimization_id,
            "status": BacktestStatus.RUNNING,
            "strategy_type": request.strategy_type,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "optimization_metric": request.optimization_metric,
            "parameter_ranges": request.parameter_ranges,
            "best_parameters": None,
            "best_metric_value": None,
            "all_results": [],
            "total_combinations": 0,
            "completed_combinations": 0,
            "walk_forward_results": None,
            "created_at": start_time,
            "started_at": start_time,
            "completed_at": None,
            "execution_time_seconds": None,
            "error_message": None
        }

        self.active_optimizations[optimization_id] = optimization_record

        # Run optimization asynchronously
        asyncio.create_task(self._execute_optimization(optimization_id, request))

        return optimization_id

    async def _execute_optimization(self, optimization_id: str, request):
        """
        Execute optimization (placeholder for now).

        Args:
            optimization_id: Optimization ID
            request: Optimization request
        """
        record = self.active_optimizations[optimization_id]

        try:
            # TODO: Implement parameter grid generation and optimization
            # For now, just mark as completed
            record['status'] = BacktestStatus.COMPLETED
            record['completed_at'] = datetime.now()
            record['execution_time_seconds'] = (
                record['completed_at'] - record['started_at']
            ).total_seconds()

            logger.info(f"Optimization {optimization_id} completed (placeholder)")

        except Exception as e:
            logger.error(f"Optimization {optimization_id} failed: {e}", exc_info=True)
            record['status'] = BacktestStatus.FAILED
            record['error_message'] = str(e)
            record['completed_at'] = datetime.now()

    def get_optimization(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """
        Get optimization results.

        Args:
            optimization_id: Optimization ID

        Returns:
            Optimization result or None if not found
        """
        if optimization_id not in self.active_optimizations:
            return None

        return self.active_optimizations[optimization_id]


# Strategy implementations for backtesting

class BaseBacktestStrategy:
    """Base class for backtest strategies"""

    def __init__(self, parameters: Dict[str, Any], position_simulator: PositionSimulator):
        self.parameters = parameters
        self.position_simulator = position_simulator
        self.price_history: List[float] = []

    async def on_bar(self, bar: Dict[str, Any]):
        """Override in subclass"""
        pass


class MACrossoverBacktestStrategy(BaseBacktestStrategy):
    """Moving Average Crossover strategy for backtesting"""

    def __init__(self, parameters: Dict[str, Any], position_simulator: PositionSimulator):
        super().__init__(parameters, position_simulator)

        self.fast_period = parameters.get('fast_period', 10)
        self.slow_period = parameters.get('slow_period', 20)
        self.stop_loss_pips = parameters.get('stop_loss_pips', 50.0)
        self.take_profit_pips = parameters.get('take_profit_pips', 100.0)
        self.position_size = parameters.get('position_size', 0.01)

        self.last_signal = None

    def calculate_ma(self, period: int) -> Optional[float]:
        """Calculate simple moving average"""
        if len(self.price_history) < period:
            return None
        return sum(self.price_history[-period:]) / period

    async def on_bar(self, bar: Dict[str, Any]):
        """Process bar and generate signals"""
        close_price = bar['close']

        # Update price history
        self.price_history.append(close_price)
        if len(self.price_history) > self.slow_period + 10:
            self.price_history.pop(0)

        # Calculate MAs
        fast_ma = self.calculate_ma(self.fast_period)
        slow_ma = self.calculate_ma(self.slow_period)

        if fast_ma is None or slow_ma is None:
            return

        # Detect crossover
        signal = None

        if fast_ma > slow_ma and self.last_signal != "BUY":
            signal = "BUY"
        elif fast_ma < slow_ma and self.last_signal != "SELL":
            signal = "SELL"

        if signal:
            # Close opposite positions first
            if self.position_simulator.get_open_positions_count() > 0:
                self.position_simulator.close_all_positions(
                    bar['time'],
                    close_price,
                    ExitReason.SIGNAL
                )

            # Open new position
            position_type = PositionType.BUY if signal == "BUY" else PositionType.SELL

            if position_type == PositionType.BUY:
                stop_loss = close_price - (self.stop_loss_pips * 0.0001)
                take_profit = close_price + (self.take_profit_pips * 0.0001)
            else:
                stop_loss = close_price + (self.stop_loss_pips * 0.0001)
                take_profit = close_price - (self.take_profit_pips * 0.0001)

            self.position_simulator.open_position(
                position_type=position_type,
                entry_time=bar['time'],
                market_price=close_price,
                position_size=self.position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self.last_signal = signal


class RSIBacktestStrategy(BaseBacktestStrategy):
    """RSI strategy for backtesting"""

    def __init__(self, parameters: Dict[str, Any], position_simulator: PositionSimulator):
        super().__init__(parameters, position_simulator)

        self.rsi_period = parameters.get('rsi_period', 14)
        self.oversold_level = parameters.get('oversold_level', 30.0)
        self.overbought_level = parameters.get('overbought_level', 70.0)
        self.stop_loss_pips = parameters.get('stop_loss_pips', 50.0)
        self.take_profit_pips = parameters.get('take_profit_pips', 100.0)
        self.position_size = parameters.get('position_size', 0.01)

        self.last_signal = None
        self.gains: List[float] = []
        self.losses: List[float] = []

    def calculate_rsi(self) -> Optional[float]:
        """Calculate RSI"""
        if len(self.price_history) < self.rsi_period + 1:
            return None

        # Calculate price changes
        changes = []
        for i in range(1, len(self.price_history)):
            change = self.price_history[i] - self.price_history[i-1]
            changes.append(change)

        # Separate gains and losses
        recent_changes = changes[-(self.rsi_period):]
        avg_gain = sum([c for c in recent_changes if c > 0]) / self.rsi_period
        avg_loss = abs(sum([c for c in recent_changes if c < 0])) / self.rsi_period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def on_bar(self, bar: Dict[str, Any]):
        """Process bar and generate signals"""
        close_price = bar['close']

        # Update price history
        self.price_history.append(close_price)
        if len(self.price_history) > self.rsi_period + 50:
            self.price_history.pop(0)

        # Calculate RSI
        rsi = self.calculate_rsi()

        if rsi is None:
            return

        # Generate signals
        signal = None

        if rsi < self.oversold_level and self.last_signal != "BUY":
            signal = "BUY"
        elif rsi > self.overbought_level and self.last_signal != "SELL":
            signal = "SELL"

        if signal:
            # Close opposite positions first
            if self.position_simulator.get_open_positions_count() > 0:
                self.position_simulator.close_all_positions(
                    bar['time'],
                    close_price,
                    ExitReason.SIGNAL
                )

            # Open new position
            position_type = PositionType.BUY if signal == "BUY" else PositionType.SELL

            if position_type == PositionType.BUY:
                stop_loss = close_price - (self.stop_loss_pips * 0.0001)
                take_profit = close_price + (self.take_profit_pips * 0.0001)
            else:
                stop_loss = close_price + (self.stop_loss_pips * 0.0001)
                take_profit = close_price - (self.take_profit_pips * 0.0001)

            self.position_simulator.open_position(
                position_type=position_type,
                entry_time=bar['time'],
                market_price=close_price,
                position_size=self.position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self.last_signal = signal
