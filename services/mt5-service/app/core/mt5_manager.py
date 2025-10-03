"""
MT5 Connection Manager

Handles all MetaTrader 5 connection lifecycle:
- Connection initialization
- Connection health monitoring
- Automatic reconnection
- Graceful shutdown
"""

import MetaTrader5 as mt5
import logging
import time
from typing import Optional, Dict
from threading import Thread, Event

logger = logging.getLogger(__name__)


class MT5Manager:
    """
    Manages MT5 Terminal connection with automatic reconnection.

    This is the SINGLE point of contact with the MetaTrader 5 Terminal.
    All MT5 operations must check connection status before execution.
    """

    def __init__(self):
        self._connected = False
        self._terminal_info: Optional[Dict] = None
        self._reconnect_thread: Optional[Thread] = None
        self._stop_reconnect = Event()
        self._reconnect_interval = 5  # seconds

    def initialize(self, account: Optional[int] = None,
                   password: Optional[str] = None,
                   server: Optional[str] = None) -> bool:
        """
        Initialize connection to MT5 Terminal.

        Args:
            account: MT5 account number (optional)
            password: MT5 account password (optional)
            server: Broker server name (optional)

        Returns:
            True if connected, False otherwise
        """
        try:
            logger.info("Initializing MT5 connection...")

            # Initialize MT5 connection
            if account and password and server:
                if not mt5.initialize(
                    login=account,
                    password=password,
                    server=server
                ):
                    error = mt5.last_error()
                    logger.error(f"MT5 initialization failed with account: {error}")
                    return False
            else:
                # Initialize without login (uses last logged account)
                if not mt5.initialize():
                    error = mt5.last_error()
                    logger.error(f"MT5 initialization failed: {error}")
                    return False

            # Verify connection by getting terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("Failed to get terminal info after initialization")
                mt5.shutdown()
                return False

            self._terminal_info = terminal_info._asdict()
            self._connected = True

            logger.info(f"MT5 connected successfully")
            logger.info(f"Terminal: {self._terminal_info.get('name')} "
                       f"Build: {self._terminal_info.get('build')}")

            return True

        except Exception as e:
            logger.error(f"Exception during MT5 initialization: {e}")
            self._connected = False
            return False

    def shutdown(self):
        """
        Gracefully shutdown MT5 connection.

        This MUST be called when the service stops to release MT5 resources.
        """
        try:
            logger.info("Shutting down MT5 connection...")

            # Stop reconnection thread if running
            if self._reconnect_thread and self._reconnect_thread.is_alive():
                self._stop_reconnect.set()
                self._reconnect_thread.join(timeout=2)

            # Shutdown MT5
            mt5.shutdown()
            self._connected = False
            self._terminal_info = None

            logger.info("MT5 shutdown completed")

        except Exception as e:
            logger.error(f"Error during MT5 shutdown: {e}")

    def is_connected(self) -> bool:
        """
        Check if MT5 is connected.

        This is a LIGHTWEIGHT check that returns cached status.
        For deep health check, use check_connection().

        Returns:
            True if connected, False otherwise
        """
        return self._connected

    def check_connection(self) -> bool:
        """
        Perform deep connection check by querying MT5 terminal.

        This actually calls MT5 to verify the connection is alive.
        Use this for periodic health checks, not for every operation.

        Returns:
            True if connection is alive, False otherwise
        """
        try:
            if not self._connected:
                return False

            # Try to get terminal info to verify connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("Connection check failed: terminal_info() returned None")
                self._connected = False
                return False

            # Update cached terminal info
            self._terminal_info = terminal_info._asdict()
            return True

        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            self._connected = False
            return False

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5.

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info("Attempting to reconnect to MT5...")

        # Shutdown existing connection
        try:
            mt5.shutdown()
        except:
            pass

        # Re-initialize
        return self.initialize()

    def start_auto_reconnect(self):
        """
        Start automatic reconnection thread.

        This thread will monitor connection and automatically reconnect
        every 5 seconds if connection is lost.
        """
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            logger.warning("Auto-reconnect thread already running")
            return

        self._stop_reconnect.clear()
        self._reconnect_thread = Thread(target=self._auto_reconnect_loop, daemon=True)
        self._reconnect_thread.start()
        logger.info("Auto-reconnect thread started")

    def stop_auto_reconnect(self):
        """
        Stop automatic reconnection thread.
        """
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            logger.info("Stopping auto-reconnect thread...")
            self._stop_reconnect.set()
            self._reconnect_thread.join(timeout=2)
            logger.info("Auto-reconnect thread stopped")

    def _auto_reconnect_loop(self):
        """
        Background thread that monitors connection and reconnects if needed.

        This runs continuously and checks connection every 30 seconds.
        If connection is lost, it attempts to reconnect every 5 seconds.
        """
        logger.info("Auto-reconnect loop started")

        while not self._stop_reconnect.is_set():
            try:
                # Check connection every 30 seconds
                if self._stop_reconnect.wait(timeout=30):
                    break

                # Perform deep connection check
                if not self.check_connection():
                    logger.warning("Connection lost, attempting to reconnect...")

                    # Retry with exponential backoff
                    attempt = 1
                    max_attempts = 3

                    while attempt <= max_attempts and not self._stop_reconnect.is_set():
                        if self.reconnect():
                            logger.info(f"Reconnection successful on attempt {attempt}")
                            break

                        logger.warning(f"Reconnection attempt {attempt}/{max_attempts} failed")
                        attempt += 1

                        # Wait before next retry (exponential backoff)
                        if attempt <= max_attempts:
                            wait_time = min(self._reconnect_interval * (2 ** (attempt - 1)), 60)
                            if self._stop_reconnect.wait(timeout=wait_time):
                                break

                    if attempt > max_attempts:
                        logger.error("Failed to reconnect after maximum attempts")

            except Exception as e:
                logger.error(f"Error in auto-reconnect loop: {e}")

        logger.info("Auto-reconnect loop stopped")

    def get_terminal_info(self) -> Optional[Dict]:
        """
        Get MT5 terminal information.

        Returns cached terminal info. Call check_connection() first
        to refresh the cache.

        Returns:
            Dictionary with terminal info or None if not connected
        """
        return self._terminal_info

    def get_last_error(self) -> tuple:
        """
        Get last MT5 error code and description.

        Returns:
            Tuple of (error_code, error_description)
        """
        error = mt5.last_error()
        return error


# Global instance
_mt5_manager: Optional[MT5Manager] = None


def get_mt5_manager() -> MT5Manager:
    """
    Get the global MT5Manager instance.

    This ensures we have a single MT5 connection across the service.

    Returns:
        MT5Manager instance
    """
    global _mt5_manager

    if _mt5_manager is None:
        _mt5_manager = MT5Manager()

    return _mt5_manager
