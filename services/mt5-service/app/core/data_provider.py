"""
MT5 Data Provider

Provides all data retrieval operations from MT5:
- Tick data retrieval
- OHLC (rates) data retrieval
- Symbol information
- Account information
"""

import MetaTrader5 as mt5
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from app.core.mt5_manager import get_mt5_manager

logger = logging.getLogger(__name__)


# Timeframe mapping: string to MT5 constant
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}


class DataProvider:
    """
    Handles all data retrieval operations from MT5.

    This class wraps MT5 data functions with proper error handling
    and connection checking.
    """

    def __init__(self):
        self.mt5_manager = get_mt5_manager()

    def get_ticks(self, symbol: str, from_timestamp: int, to_timestamp: int,
                  flags: int = mt5.COPY_TICKS_ALL) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Retrieve tick data from MT5.

        Args:
            symbol: Symbol name (e.g., "EURUSD")
            from_timestamp: Start timestamp (Unix timestamp in seconds)
            to_timestamp: End timestamp (Unix timestamp in seconds)
            flags: Copy flags (COPY_TICKS_ALL, COPY_TICKS_INFO, COPY_TICKS_TRADE)

        Returns:
            Tuple of (success, ticks_list, error_message)
            - success: True if operation succeeded
            - ticks_list: List of tick dictionaries
            - error_message: Error description if failed
        """
        # Check connection
        if not self.mt5_manager.is_connected():
            return False, None, "MT5 not connected"

        try:
            # Convert Unix timestamps to datetime
            from_date = datetime.fromtimestamp(from_timestamp)
            to_date = datetime.fromtimestamp(to_timestamp)

            logger.info(f"Fetching ticks for {symbol} from {from_date} to {to_date}")

            # Get ticks from MT5
            ticks = mt5.copy_ticks_range(symbol, from_date, to_date, flags)

            if ticks is None or len(ticks) == 0:
                error = mt5.last_error()
                logger.warning(f"No ticks retrieved for {symbol}: {error}")
                return False, None, f"No ticks available: {error}"

            # Convert to list of dictionaries
            ticks_list = []
            for tick in ticks:
                ticks_list.append({
                    "time": int(tick.time),  # Unix timestamp
                    "time_msc": int(tick.time_msc),  # Unix timestamp in milliseconds
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "last": float(tick.last),
                    "volume": int(tick.volume),
                    "flags": int(tick.flags)
                })

            logger.info(f"Retrieved {len(ticks_list)} ticks for {symbol}")
            return True, ticks_list, None

        except Exception as e:
            logger.error(f"Error fetching ticks for {symbol}: {e}")
            return False, None, str(e)

    def get_rates(self, symbol: str, timeframe: str, from_timestamp: int,
                  to_timestamp: int) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Retrieve OHLC (rates) data from MT5.

        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: Timeframe string (M1, M5, H1, etc.)
            from_timestamp: Start timestamp (Unix timestamp in seconds)
            to_timestamp: End timestamp (Unix timestamp in seconds)

        Returns:
            Tuple of (success, rates_list, error_message)
            - success: True if operation succeeded
            - rates_list: List of OHLC bar dictionaries
            - error_message: Error description if failed
        """
        # Check connection
        if not self.mt5_manager.is_connected():
            return False, None, "MT5 not connected"

        # Validate and convert timeframe
        if timeframe not in TIMEFRAME_MAP:
            return False, None, f"Invalid timeframe: {timeframe}. Must be one of {list(TIMEFRAME_MAP.keys())}"

        mt5_timeframe = TIMEFRAME_MAP[timeframe]

        try:
            # Convert Unix timestamps to datetime
            from_date = datetime.fromtimestamp(from_timestamp)
            to_date = datetime.fromtimestamp(to_timestamp)

            logger.info(f"Fetching {timeframe} rates for {symbol} from {from_date} to {to_date}")

            # Get rates from MT5
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date, to_date)

            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.warning(f"No rates retrieved for {symbol}: {error}")
                return False, None, f"No rates available: {error}"

            # Convert to list of dictionaries
            rates_list = []
            for rate in rates:
                rates_list.append({
                    "time": int(rate[0]),  # Unix timestamp
                    "open": float(rate[1]),
                    "high": float(rate[2]),
                    "low": float(rate[3]),
                    "close": float(rate[4]),
                    "tick_volume": int(rate[5]),
                    "spread": int(rate[6]),
                    "real_volume": int(rate[7])
                })

            logger.info(f"Retrieved {len(rates_list)} {timeframe} bars for {symbol}")
            return True, rates_list, None

        except Exception as e:
            logger.error(f"Error fetching rates for {symbol}: {e}")
            return False, None, str(e)

    def get_symbol_info(self, symbol: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Get symbol information from MT5.

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            Tuple of (success, symbol_info_dict, error_message)
            - success: True if operation succeeded
            - symbol_info_dict: Dictionary with symbol properties
            - error_message: Error description if failed
        """
        # Check connection
        if not self.mt5_manager.is_connected():
            return False, None, "MT5 not connected"

        try:
            logger.info(f"Fetching symbol info for {symbol}")

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                error = mt5.last_error()
                logger.warning(f"Symbol {symbol} not found: {error}")
                return False, None, f"Symbol not found: {error}"

            # Convert to dictionary
            info_dict = {
                "name": symbol_info.name,
                "description": symbol_info.description,
                "point": symbol_info.point,
                "digits": symbol_info.digits,
                "spread": symbol_info.spread,
                "trade_contract_size": symbol_info.trade_contract_size,
                "trade_tick_size": symbol_info.trade_tick_size,
                "trade_tick_value": symbol_info.trade_tick_value,
                "volume_min": symbol_info.volume_min,
                "volume_max": symbol_info.volume_max,
                "volume_step": symbol_info.volume_step,
                "currency_base": symbol_info.currency_base,
                "currency_profit": symbol_info.currency_profit,
                "currency_margin": symbol_info.currency_margin,
                "trade_mode": symbol_info.trade_mode,
                "trade_stops_level": symbol_info.trade_stops_level,
                "visible": symbol_info.visible,
                "select": symbol_info.select
            }

            logger.info(f"Symbol info retrieved for {symbol}")
            return True, info_dict, None

        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return False, None, str(e)

    def get_symbol_tick(self, symbol: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Get current tick for a symbol.

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            Tuple of (success, tick_dict, error_message)
        """
        # Check connection
        if not self.mt5_manager.is_connected():
            return False, None, "MT5 not connected"

        try:
            tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                error = mt5.last_error()
                return False, None, f"Failed to get tick: {error}"

            tick_dict = {
                "time": int(tick.time),
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "last": float(tick.last),
                "volume": int(tick.volume)
            }

            return True, tick_dict, None

        except Exception as e:
            logger.error(f"Error fetching tick for {symbol}: {e}")
            return False, None, str(e)

    def get_account_info(self) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Get MT5 account information.

        Returns:
            Tuple of (success, account_info_dict, error_message)
            - success: True if operation succeeded
            - account_info_dict: Dictionary with account properties
            - error_message: Error description if failed
        """
        # Check connection
        if not self.mt5_manager.is_connected():
            return False, None, "MT5 not connected"

        try:
            logger.info("Fetching account info")

            # Get account info
            account_info = mt5.account_info()

            if account_info is None:
                error = mt5.last_error()
                logger.warning(f"Failed to get account info: {error}")
                return False, None, f"Failed to get account info: {error}"

            # Convert to dictionary
            info_dict = {
                "login": account_info.login,
                "trade_mode": account_info.trade_mode,
                "leverage": account_info.leverage,
                "limit_orders": account_info.limit_orders,
                "margin_so_mode": account_info.margin_so_mode,
                "trade_allowed": account_info.trade_allowed,
                "trade_expert": account_info.trade_expert,
                "margin_mode": account_info.margin_mode,
                "currency_digits": account_info.currency_digits,
                "balance": account_info.balance,
                "credit": account_info.credit,
                "profit": account_info.profit,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "margin_so_call": account_info.margin_so_call,
                "margin_so_so": account_info.margin_so_so,
                "margin_initial": account_info.margin_initial,
                "margin_maintenance": account_info.margin_maintenance,
                "assets": account_info.assets,
                "liabilities": account_info.liabilities,
                "commission_blocked": account_info.commission_blocked,
                "name": account_info.name,
                "server": account_info.server,
                "currency": account_info.currency,
                "company": account_info.company
            }

            logger.info("Account info retrieved")
            return True, info_dict, None

        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return False, None, str(e)

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol exists and is available.

        Args:
            symbol: Symbol name

        Returns:
            True if symbol is valid, False otherwise
        """
        if not self.mt5_manager.is_connected():
            return False

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False

            # Check if symbol is visible and can be selected
            if not symbol_info.visible:
                # Try to enable symbol in Market Watch
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Symbol {symbol} cannot be selected")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
