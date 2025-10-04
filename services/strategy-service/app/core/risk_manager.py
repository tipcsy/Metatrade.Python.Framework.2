"""
Risk Manager - Handles risk calculations and position sizing
"""

import logging
from typing import Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLimitType(Enum):
    """Risk limit types"""
    MAX_POSITIONS = "MAX_POSITIONS"
    MAX_RISK_PER_TRADE = "MAX_RISK_PER_TRADE"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"


class RiskManager:
    """Manages trading risk and position sizing"""

    def __init__(
        self,
        account_balance: float = 10000.0,
        max_risk_per_trade: float = 2.0,  # Percentage of account
        max_positions: int = 5,
        max_daily_loss: float = 5.0,  # Percentage of account
        max_drawdown: float = 20.0  # Percentage of account
    ):
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.daily_loss = 0.0
        self.peak_balance = account_balance
        self.risk_limits_active = True

        logger.info(
            f"Risk Manager initialized | Balance: ${account_balance} | "
            f"Max Risk/Trade: {max_risk_per_trade}% | Max Positions: {max_positions}"
        )

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_percentage: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk parameters

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            risk_percentage: Risk percentage (uses max_risk_per_trade if not provided)

        Returns:
            Position size in lots
        """
        if risk_percentage is None:
            risk_percentage = self.max_risk_per_trade

        # Calculate risk amount in account currency
        risk_amount = self.account_balance * (risk_percentage / 100)

        # Calculate stop loss distance in pips
        sl_distance_pips = abs(entry_price - stop_loss) * 10000

        if sl_distance_pips == 0:
            logger.warning("Stop loss distance is zero, using minimum position size")
            return 0.01

        # Calculate position size (simplified: $10 per pip per lot)
        pip_value = 10.0  # USD per pip for 1 lot
        position_size = risk_amount / (sl_distance_pips * pip_value)

        # Round to 2 decimal places and enforce minimum
        position_size = max(0.01, round(position_size, 2))

        logger.info(
            f"Position sizing for {symbol} | Entry: {entry_price} | SL: {stop_loss} | "
            f"Risk: {risk_percentage}% (${risk_amount:.2f}) | Size: {position_size} lots"
        )

        return position_size

    def can_open_position(
        self,
        current_open_positions: int,
        symbol: str,
        position_size: float
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a new position can be opened based on risk limits

        Returns:
            (can_open, reason) - Boolean and optional reason if cannot open
        """
        if not self.risk_limits_active:
            return True, None

        # Check max positions limit
        if current_open_positions >= self.max_positions:
            reason = f"Maximum positions limit reached ({self.max_positions})"
            logger.warning(reason)
            return False, reason

        # Check daily loss limit
        daily_loss_pct = (self.daily_loss / self.account_balance) * 100
        if daily_loss_pct >= self.max_daily_loss:
            reason = f"Daily loss limit reached ({daily_loss_pct:.2f}% >= {self.max_daily_loss}%)"
            logger.warning(reason)
            return False, reason

        # Check drawdown limit
        current_drawdown = ((self.peak_balance - self.account_balance) / self.peak_balance) * 100
        if current_drawdown >= self.max_drawdown:
            reason = f"Drawdown limit reached ({current_drawdown:.2f}% >= {self.max_drawdown}%)"
            logger.warning(reason)
            return False, reason

        # Check minimum position size
        if position_size < 0.01:
            reason = "Position size too small (< 0.01 lots)"
            logger.warning(reason)
            return False, reason

        return True, None

    def update_account_balance(self, new_balance: float):
        """Update account balance and track peak for drawdown calculation"""
        old_balance = self.account_balance
        self.account_balance = new_balance

        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Update daily loss
        balance_change = new_balance - old_balance
        if balance_change < 0:
            self.daily_loss += abs(balance_change)

        logger.info(
            f"Balance updated: ${old_balance:.2f} -> ${new_balance:.2f} | "
            f"Peak: ${self.peak_balance:.2f} | Daily Loss: ${self.daily_loss:.2f}"
        )

    def reset_daily_loss(self):
        """Reset daily loss counter (should be called at start of each trading day)"""
        logger.info(f"Resetting daily loss from ${self.daily_loss:.2f} to $0.00")
        self.daily_loss = 0.0

    def get_max_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Get maximum allowed position size based on risk limits"""
        return self.calculate_position_size(symbol, entry_price, stop_loss, self.max_risk_per_trade)

    def validate_stop_loss(
        self,
        entry_price: float,
        stop_loss: float,
        position_type: str,
        min_sl_pips: float = 10.0,
        max_sl_pips: float = 500.0
    ) -> tuple[bool, Optional[str]]:
        """
        Validate stop loss distance

        Returns:
            (is_valid, reason) - Boolean and optional reason if invalid
        """
        sl_distance_pips = abs(entry_price - stop_loss) * 10000

        if sl_distance_pips < min_sl_pips:
            reason = f"Stop loss too tight ({sl_distance_pips:.1f} pips < {min_sl_pips} pips)"
            return False, reason

        if sl_distance_pips > max_sl_pips:
            reason = f"Stop loss too wide ({sl_distance_pips:.1f} pips > {max_sl_pips} pips)"
            return False, reason

        # Check if SL is in correct direction
        if position_type == "BUY" and stop_loss >= entry_price:
            return False, "Buy stop loss must be below entry price"
        elif position_type == "SELL" and stop_loss <= entry_price:
            return False, "Sell stop loss must be above entry price"

        return True, None

    def validate_take_profit(
        self,
        entry_price: float,
        take_profit: float,
        position_type: str,
        min_tp_pips: float = 10.0
    ) -> tuple[bool, Optional[str]]:
        """
        Validate take profit distance

        Returns:
            (is_valid, reason) - Boolean and optional reason if invalid
        """
        tp_distance_pips = abs(entry_price - take_profit) * 10000

        if tp_distance_pips < min_tp_pips:
            reason = f"Take profit too close ({tp_distance_pips:.1f} pips < {min_tp_pips} pips)"
            return False, reason

        # Check if TP is in correct direction
        if position_type == "BUY" and take_profit <= entry_price:
            return False, "Buy take profit must be above entry price"
        elif position_type == "SELL" and take_profit >= entry_price:
            return False, "Sell take profit must be below entry price"

        return True, None

    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """Calculate risk/reward ratio"""
        risk_pips = abs(entry_price - stop_loss) * 10000
        reward_pips = abs(take_profit - entry_price) * 10000

        if risk_pips == 0:
            return 0.0

        return round(reward_pips / risk_pips, 2)

    def get_risk_status(self) -> dict:
        """Get current risk status"""
        current_drawdown = ((self.peak_balance - self.account_balance) / self.peak_balance) * 100
        daily_loss_pct = (self.daily_loss / self.account_balance) * 100

        return {
            "account_balance": round(self.account_balance, 2),
            "initial_balance": round(self.initial_balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "current_drawdown_pct": round(current_drawdown, 2),
            "max_drawdown_pct": self.max_drawdown,
            "daily_loss": round(self.daily_loss, 2),
            "daily_loss_pct": round(daily_loss_pct, 2),
            "max_daily_loss_pct": self.max_daily_loss,
            "max_positions": self.max_positions,
            "max_risk_per_trade_pct": self.max_risk_per_trade,
            "risk_limits_active": self.risk_limits_active,
            "profit_loss": round(self.account_balance - self.initial_balance, 2)
        }

    def enable_risk_limits(self):
        """Enable risk limit checks"""
        self.risk_limits_active = True
        logger.info("Risk limits enabled")

    def disable_risk_limits(self):
        """Disable risk limit checks (use with caution!)"""
        self.risk_limits_active = False
        logger.warning("Risk limits DISABLED - trading without protection!")

    def calculate_correlation_risk(
        self,
        symbols: list[str],
        correlation_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Calculate correlation risk between symbols (placeholder for future implementation)

        Args:
            symbols: List of symbols to check
            correlation_threshold: Correlation coefficient threshold

        Returns:
            Dictionary of symbol pairs and their correlation coefficients
        """
        # This would require historical price data to calculate actual correlations
        # For now, return empty dict (to be implemented when data service integration is ready)
        logger.info(f"Correlation risk check requested for symbols: {symbols}")
        return {}
