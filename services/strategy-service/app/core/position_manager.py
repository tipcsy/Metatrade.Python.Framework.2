"""
Position Manager - Handles position tracking and management
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position types"""
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(Enum):
    """Position status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


class Position:
    """Represents a trading position"""

    def __init__(
        self,
        position_id: str,
        symbol: str,
        position_type: PositionType,
        volume: float,
        open_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_id: Optional[str] = None
    ):
        self.position_id = position_id
        self.symbol = symbol
        self.position_type = position_type
        self.volume = volume
        self.open_price = open_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy_id = strategy_id
        self.status = PositionStatus.OPEN
        self.open_time = datetime.now()
        self.close_time = None
        self.close_price = None
        self.profit = 0.0
        self.commission = 0.0
        self.swap = 0.0

    def update_current_price(self, current_price: float):
        """Update profit based on current price"""
        if self.status != PositionStatus.OPEN:
            return

        # Calculate profit in pips (simplified for major forex pairs)
        if self.position_type == PositionType.BUY:
            pip_diff = (current_price - self.open_price) * 10000
        else:
            pip_diff = (self.open_price - current_price) * 10000

        # Calculate monetary profit (simplified: $10 per pip per lot)
        self.profit = pip_diff * 10 * self.volume

    def close(self, close_price: float):
        """Close the position"""
        self.status = PositionStatus.CLOSED
        self.close_time = datetime.now()
        self.close_price = close_price
        self.update_current_price(close_price)
        logger.info(f"Position {self.position_id} closed at {close_price}, profit: {self.profit:.2f}")

    def check_sl_tp(self, current_price: float) -> Optional[str]:
        """Check if stop loss or take profit is hit"""
        if self.status != PositionStatus.OPEN:
            return None

        # Check stop loss
        if self.stop_loss:
            if self.position_type == PositionType.BUY and current_price <= self.stop_loss:
                return "STOP_LOSS"
            elif self.position_type == PositionType.SELL and current_price >= self.stop_loss:
                return "STOP_LOSS"

        # Check take profit
        if self.take_profit:
            if self.position_type == PositionType.BUY and current_price >= self.take_profit:
                return "TAKE_PROFIT"
            elif self.position_type == PositionType.SELL and current_price <= self.take_profit:
                return "TAKE_PROFIT"

        return None

    def to_dict(self) -> dict:
        """Convert position to dictionary"""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "type": self.position_type.value,
            "volume": self.volume,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_id": self.strategy_id,
            "status": self.status.value,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "profit": round(self.profit, 2),
            "commission": round(self.commission, 2),
            "swap": round(self.swap, 2)
        }


class PositionManager:
    """Manages all trading positions"""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.position_counter = 0
        self.mock_mode = True  # Running in mock mode without MT5
        logger.info("Position Manager initialized in mock mode")

    def _generate_position_id(self) -> str:
        """Generate unique position ID"""
        self.position_counter += 1
        return f"POS_{self.position_counter:06d}"

    def open_position(
        self,
        symbol: str,
        position_type: PositionType,
        volume: float,
        open_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_id: Optional[str] = None
    ) -> Position:
        """Open a new position"""
        position_id = self._generate_position_id()

        position = Position(
            position_id=position_id,
            symbol=symbol,
            position_type=position_type,
            volume=volume,
            open_price=open_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=strategy_id
        )

        self.positions[position_id] = position
        logger.info(
            f"Position opened: {position_id} | {symbol} | {position_type.value} | "
            f"Vol: {volume} | Price: {open_price} | SL: {stop_loss} | TP: {take_profit}"
        )

        return position

    def close_position(self, position_id: str, close_price: float) -> bool:
        """Close a position"""
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return False

        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            logger.warning(f"Position {position_id} is not open")
            return False

        position.close(close_price)
        return True

    def close_all_positions(self, current_prices: Dict[str, float]) -> int:
        """Close all open positions"""
        closed_count = 0
        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                if position.symbol in current_prices:
                    position.close(current_prices[position.symbol])
                    closed_count += 1

        logger.info(f"Closed {closed_count} positions")
        return closed_count

    def close_positions_by_strategy(self, strategy_id: str, current_prices: Dict[str, float]) -> int:
        """Close all positions for a specific strategy"""
        closed_count = 0
        for position in self.positions.values():
            if position.strategy_id == strategy_id and position.status == PositionStatus.OPEN:
                if position.symbol in current_prices:
                    position.close(current_prices[position.symbol])
                    closed_count += 1

        logger.info(f"Closed {closed_count} positions for strategy {strategy_id}")
        return closed_count

    def update_positions(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Update all positions with current prices and check SL/TP
        Returns list of position IDs that were closed
        """
        closed_positions = []

        for position in self.positions.values():
            if position.status != PositionStatus.OPEN:
                continue

            if position.symbol not in current_prices:
                continue

            current_price = current_prices[position.symbol]
            position.update_current_price(current_price)

            # Check if SL or TP is hit
            sl_tp_result = position.check_sl_tp(current_price)
            if sl_tp_result:
                position.close(current_price)
                closed_positions.append(position.position_id)
                logger.info(f"Position {position.position_id} closed by {sl_tp_result}")

        return closed_positions

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position"""
        return self.positions.get(position_id)

    def get_open_positions(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[Position]:
        """Get all open positions, optionally filtered by symbol or strategy"""
        positions = [
            pos for pos in self.positions.values()
            if pos.status == PositionStatus.OPEN
        ]

        if symbol:
            positions = [pos for pos in positions if pos.symbol == symbol]

        if strategy_id:
            positions = [pos for pos in positions if pos.strategy_id == strategy_id]

        return positions

    def get_closed_positions(self, strategy_id: Optional[str] = None) -> List[Position]:
        """Get all closed positions, optionally filtered by strategy"""
        positions = [
            pos for pos in self.positions.values()
            if pos.status == PositionStatus.CLOSED
        ]

        if strategy_id:
            positions = [pos for pos in positions if pos.strategy_id == strategy_id]

        return positions

    def get_total_profit(self, strategy_id: Optional[str] = None) -> float:
        """Calculate total profit across all positions"""
        positions = self.positions.values()

        if strategy_id:
            positions = [pos for pos in positions if pos.strategy_id == strategy_id]

        total = sum(pos.profit for pos in positions)
        return round(total, 2)

    def get_open_positions_count(self, strategy_id: Optional[str] = None) -> int:
        """Get count of open positions"""
        return len(self.get_open_positions(strategy_id=strategy_id))

    def get_exposure(self, symbol: str) -> Dict[str, float]:
        """Get total exposure for a symbol"""
        buy_volume = 0.0
        sell_volume = 0.0

        for position in self.get_open_positions(symbol=symbol):
            if position.position_type == PositionType.BUY:
                buy_volume += position.volume
            else:
                sell_volume += position.volume

        return {
            "symbol": symbol,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "net_volume": buy_volume - sell_volume
        }

    def get_statistics(self) -> dict:
        """Get position statistics"""
        open_positions = self.get_open_positions()
        closed_positions = self.get_closed_positions()

        # Calculate win rate
        winning_trades = len([pos for pos in closed_positions if pos.profit > 0])
        total_closed = len(closed_positions)
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0

        # Calculate average profit
        avg_profit = sum(pos.profit for pos in closed_positions) / total_closed if total_closed > 0 else 0

        return {
            "total_positions": len(self.positions),
            "open_positions": len(open_positions),
            "closed_positions": len(closed_positions),
            "total_profit": self.get_total_profit(),
            "winning_trades": winning_trades,
            "losing_trades": total_closed - winning_trades,
            "win_rate": round(win_rate, 2),
            "average_profit": round(avg_profit, 2)
        }
