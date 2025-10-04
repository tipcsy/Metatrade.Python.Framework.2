"""
Position Simulator - Simulates position opening/closing with realistic conditions

This module handles virtual position management during backtesting, including:
- Realistic fill price simulation with bid/ask spread
- Slippage modeling
- Commission calculation
- Stop-loss and take-profit execution
- Position tracking and P&L calculation
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class PositionType(str, Enum):
    """Position type"""
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(str, Enum):
    """Position status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(str, Enum):
    """Reason for position exit"""
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    SIGNAL = "SIGNAL"
    END_OF_DATA = "EOD"
    MANUAL = "MANUAL"


class Position:
    """
    Represents a single simulated trading position.
    """

    def __init__(
        self,
        position_id: str,
        position_type: PositionType,
        entry_time: int,
        entry_price: float,
        position_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        commission: float = 0.0
    ):
        self.position_id = position_id
        self.position_type = position_type
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission = commission

        self.status = PositionStatus.OPEN
        self.exit_time: Optional[int] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[ExitReason] = None

        self.swap = 0.0  # Overnight swap fees
        self.profit: Optional[float] = None
        self.profit_pips: Optional[float] = None

        # MAE/MFE tracking
        self.mae = 0.0  # Maximum Adverse Excursion (worst unrealized loss)
        self.mfe = 0.0  # Maximum Favorable Excursion (best unrealized profit)
        self.peak_profit = 0.0
        self.peak_loss = 0.0

    def update_mae_mfe(self, current_price: float, pip_value: float):
        """
        Update MAE and MFE based on current price.

        Args:
            current_price: Current market price
            pip_value: Value of one pip
        """
        # Calculate unrealized profit
        if self.position_type == PositionType.BUY:
            unrealized_pips = (current_price - self.entry_price) / pip_value
        else:
            unrealized_pips = (self.entry_price - current_price) / pip_value

        unrealized_profit = unrealized_pips * pip_value * self.position_size * 100000  # Assuming standard lot

        # Update MAE (maximum adverse excursion)
        if unrealized_profit < self.peak_loss:
            self.peak_loss = unrealized_profit
            self.mae = abs(unrealized_profit)

        # Update MFE (maximum favorable excursion)
        if unrealized_profit > self.peak_profit:
            self.peak_profit = unrealized_profit
            self.mfe = unrealized_profit

    def close(
        self,
        exit_time: int,
        exit_price: float,
        exit_reason: ExitReason,
        pip_value: float,
        exit_commission: float = 0.0
    ):
        """
        Close the position and calculate final P&L.

        Args:
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
            pip_value: Value of one pip
            exit_commission: Additional commission on exit
        """
        self.status = PositionStatus.CLOSED
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate profit in pips
        if self.position_type == PositionType.BUY:
            self.profit_pips = (exit_price - self.entry_price) / pip_value
        else:
            self.profit_pips = (self.entry_price - exit_price) / pip_value

        # Calculate profit in currency (assuming standard lot = 100,000 units)
        # For simplicity, using standard lot calculation
        # In real implementation, this should be symbol-specific
        point_value = self.position_size * 100000 * pip_value
        self.profit = self.profit_pips * point_value

        # Subtract commissions and swap
        total_commission = self.commission + exit_commission
        self.profit -= (total_commission + self.swap)

        logger.debug(
            f"Position {self.position_id} closed: "
            f"{self.position_type.value} {self.position_size} lots @ {exit_price:.5f}, "
            f"Profit: ${self.profit:.2f} ({self.profit_pips:.1f} pips), "
            f"Reason: {exit_reason.value}"
        )

    def get_duration_seconds(self) -> Optional[int]:
        """Get position duration in seconds"""
        if self.exit_time is None:
            return None
        return self.exit_time - self.entry_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            "trade_id": self.position_id,
            "entry_time": datetime.fromtimestamp(self.entry_time),
            "exit_time": datetime.fromtimestamp(self.exit_time) if self.exit_time else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_type": self.position_type.value,
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "commission": self.commission,
            "swap": self.swap,
            "profit": self.profit,
            "profit_pips": self.profit_pips,
            "mae": self.mae,
            "mfe": self.mfe,
            "duration_seconds": self.get_duration_seconds(),
            "exit_reason": self.exit_reason.value if self.exit_reason else None
        }


class PositionSimulator:
    """
    Simulates position management during backtesting.

    Handles position opening, monitoring, and closing with realistic
    market conditions including spread, slippage, and commission.
    """

    def __init__(
        self,
        initial_balance: float,
        commission: float = 0.0,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
        pip_value: float = 0.0001  # Standard for most pairs, 0.01 for JPY pairs
    ):
        """
        Initialize position simulator.

        Args:
            initial_balance: Starting account balance
            commission: Commission per trade (USD)
            spread_pips: Spread in pips
            slippage_pips: Slippage in pips (for market orders)
            pip_value: Value of one pip (0.0001 for most pairs, 0.01 for JPY)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.pip_value = pip_value

        self.positions: Dict[str, Position] = {}
        self.open_positions: List[str] = []
        self.closed_positions: List[str] = []

        self.equity = initial_balance
        self.peak_balance = initial_balance

        logger.info(
            f"PositionSimulator initialized: "
            f"Balance=${initial_balance:.2f}, "
            f"Commission=${commission:.2f}, "
            f"Spread={spread_pips} pips, "
            f"Slippage={slippage_pips} pips"
        )

    def _calculate_fill_price(
        self,
        position_type: PositionType,
        market_price: float,
        is_entry: bool
    ) -> float:
        """
        Calculate realistic fill price with spread and slippage.

        For entry:
        - BUY: Use ask price + slippage
        - SELL: Use bid price - slippage

        For exit:
        - BUY close: Use bid price - slippage
        - SELL close: Use ask price + slippage

        Args:
            position_type: Type of position
            market_price: Current market price (mid price)
            is_entry: True for entry, False for exit

        Returns:
            Realistic fill price
        """
        spread = self.spread_pips * self.pip_value
        slippage = self.slippage_pips * self.pip_value

        if is_entry:
            if position_type == PositionType.BUY:
                # Buy at ask + slippage
                return market_price + (spread / 2) + slippage
            else:
                # Sell at bid - slippage
                return market_price - (spread / 2) - slippage
        else:
            if position_type == PositionType.BUY:
                # Close buy at bid - slippage
                return market_price - (spread / 2) - slippage
            else:
                # Close sell at ask + slippage
                return market_price + (spread / 2) + slippage

    def open_position(
        self,
        position_type: PositionType,
        entry_time: int,
        market_price: float,
        position_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[str]:
        """
        Open a new position with realistic fill price.

        Args:
            position_type: BUY or SELL
            entry_time: Entry timestamp
            market_price: Current market price (mid price)
            position_size: Position size in lots
            stop_loss: Stop-loss price
            take_profit: Take-profit price

        Returns:
            Position ID or None if insufficient balance
        """
        # Calculate realistic entry price
        entry_price = self._calculate_fill_price(position_type, market_price, is_entry=True)

        # Check if we have enough balance for commission
        if self.balance < self.commission:
            logger.warning(f"Insufficient balance for commission: ${self.balance:.2f} < ${self.commission:.2f}")
            return None

        # Generate position ID
        position_id = str(uuid.uuid4())

        # Create position
        position = Position(
            position_id=position_id,
            position_type=position_type,
            entry_time=entry_time,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=self.commission
        )

        # Deduct commission from balance
        self.balance -= self.commission

        # Store position
        self.positions[position_id] = position
        self.open_positions.append(position_id)

        logger.info(
            f"Position opened: {position_type.value} {position_size} lots @ {entry_price:.5f}, "
            f"SL={stop_loss:.5f if stop_loss else 'None'}, "
            f"TP={take_profit:.5f if take_profit else 'None'}"
        )

        return position_id

    def update_positions(self, current_time: int, bar: Dict[str, Any]):
        """
        Update all open positions based on current bar.

        Checks for:
        - Stop-loss hits
        - Take-profit hits
        - Updates MAE/MFE

        Args:
            current_time: Current timestamp
            bar: Current OHLC bar
        """
        positions_to_close = []

        for position_id in self.open_positions[:]:  # Copy to avoid modification during iteration
            position = self.positions[position_id]

            # Update MAE/MFE based on bar extremes
            position.update_mae_mfe(bar['high'], self.pip_value)
            position.update_mae_mfe(bar['low'], self.pip_value)

            # Check stop-loss and take-profit
            if position.position_type == PositionType.BUY:
                # Check stop-loss hit (price went down to SL)
                if position.stop_loss and bar['low'] <= position.stop_loss:
                    positions_to_close.append((position_id, position.stop_loss, ExitReason.STOP_LOSS))

                # Check take-profit hit (price went up to TP)
                elif position.take_profit and bar['high'] >= position.take_profit:
                    positions_to_close.append((position_id, position.take_profit, ExitReason.TAKE_PROFIT))

            else:  # SELL
                # Check stop-loss hit (price went up to SL)
                if position.stop_loss and bar['high'] >= position.stop_loss:
                    positions_to_close.append((position_id, position.stop_loss, ExitReason.STOP_LOSS))

                # Check take-profit hit (price went down to TP)
                elif position.take_profit and bar['low'] <= position.take_profit:
                    positions_to_close.append((position_id, position.take_profit, ExitReason.TAKE_PROFIT))

        # Close positions that hit SL or TP
        for position_id, exit_price, exit_reason in positions_to_close:
            self.close_position(position_id, current_time, exit_price, exit_reason)

    def close_position(
        self,
        position_id: str,
        exit_time: int,
        market_price: float,
        exit_reason: ExitReason
    ) -> bool:
        """
        Close an open position.

        Args:
            position_id: Position ID
            exit_time: Exit timestamp
            market_price: Market price at exit
            exit_reason: Reason for closing

        Returns:
            True if successful
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return False

        position = self.positions[position_id]

        if position.status == PositionStatus.CLOSED:
            logger.warning(f"Position {position_id} already closed")
            return False

        # Calculate realistic exit price (unless it's SL/TP which use exact levels)
        if exit_reason in [ExitReason.STOP_LOSS, ExitReason.TAKE_PROFIT]:
            exit_price = market_price  # Use the exact SL/TP level
        else:
            exit_price = self._calculate_fill_price(position.position_type, market_price, is_entry=False)

        # Close position and calculate P&L
        position.close(
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pip_value=self.pip_value,
            exit_commission=self.commission
        )

        # Update balance with profit/loss
        self.balance += position.profit

        # Track peak balance
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Move from open to closed
        self.open_positions.remove(position_id)
        self.closed_positions.append(position_id)

        return True

    def close_all_positions(
        self,
        exit_time: int,
        market_price: float,
        exit_reason: ExitReason = ExitReason.END_OF_DATA
    ):
        """
        Close all open positions.

        Args:
            exit_time: Exit timestamp
            market_price: Market price
            exit_reason: Reason for closing
        """
        for position_id in self.open_positions[:]:
            self.close_position(position_id, exit_time, market_price, exit_reason)

    def get_equity(self, current_price: float) -> float:
        """
        Calculate current equity (balance + unrealized P&L).

        Args:
            current_price: Current market price

        Returns:
            Current equity
        """
        unrealized_pnl = 0.0

        for position_id in self.open_positions:
            position = self.positions[position_id]

            if position.position_type == PositionType.BUY:
                unrealized_pips = (current_price - position.entry_price) / self.pip_value
            else:
                unrealized_pips = (position.entry_price - current_price) / self.pip_value

            point_value = position.position_size * 100000 * self.pip_value
            unrealized_pnl += unrealized_pips * point_value

        self.equity = self.balance + unrealized_pnl
        return self.equity

    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get all closed trades"""
        return [
            self.positions[position_id].to_dict()
            for position_id in self.closed_positions
        ]

    def get_open_positions_count(self) -> int:
        """Get number of open positions"""
        return len(self.open_positions)

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulator statistics"""
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.balance,
            "current_equity": self.equity,
            "peak_balance": self.peak_balance,
            "total_trades": len(self.closed_positions),
            "open_positions": len(self.open_positions)
        }
