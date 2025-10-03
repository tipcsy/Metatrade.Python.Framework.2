"""
Database Manager for Data Service

Handles all database operations including:
- Database path management
- Table creation
- Batch data insertion
- Data querying with proper indexing
"""

import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from app.database.schema import (
    get_tick_table_sql,
    get_ohlc_table_sql,
    get_performance_pragmas
)

logger = logging.getLogger(__name__)

# Base database directory
BASE_DB_DIR = "/home/tipcsy/Metatrade.Python.Framework.2/database"


class DatabaseManager:
    """
    Manages all database operations for tick and OHLC data

    Features:
    - Symbol-based partitioned databases
    - Batch insert operations for performance
    - Connection pooling
    - Automatic table and index creation
    """

    def __init__(self, base_dir: str = BASE_DB_DIR):
        self.base_dir = base_dir
        self._ensure_base_directory()

    def _ensure_base_directory(self):
        """Ensure base database directory exists"""
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"Database base directory: {self.base_dir}")

    def get_tick_db_path(self, symbol: str, year: int, month: int) -> str:
        """
        Get path to tick database file

        Format: database/{YEAR}/{SYMBOL}_ticks_{MONTH}.db
        Example: database/2025/EURUSD_ticks_01.db

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            year: Year (e.g., 2025)
            month: Month (1-12)

        Returns:
            Full path to database file
        """
        year_dir = os.path.join(self.base_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        month_str = f"{month:02d}"
        filename = f"{symbol}_ticks_{month_str}.db"
        return os.path.join(year_dir, filename)

    def get_ohlc_db_path(self, symbol: str, year: int) -> str:
        """
        Get path to OHLC database file

        Format: database/{YEAR}/{SYMBOL}_ohlc.db
        Example: database/2025/EURUSD_ohlc.db

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            year: Year (e.g., 2025)

        Returns:
            Full path to database file
        """
        year_dir = os.path.join(self.base_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        filename = f"{symbol}_ohlc.db"
        return os.path.join(year_dir, filename)

    @contextmanager
    def get_connection(self, db_path: str):
        """
        Context manager for database connections

        Args:
            db_path: Path to database file

        Yields:
            sqlite3.Connection object
        """
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            # Apply performance pragmas
            cursor = conn.cursor()
            for pragma in get_performance_pragmas():
                cursor.execute(pragma)

            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def create_tick_table(self, db_path: str) -> bool:
        """
        Create tick table and indexes if not exists

        Args:
            db_path: Path to database file

        Returns:
            True if successful
        """
        try:
            with self.get_connection(db_path) as conn:
                cursor = conn.cursor()

                for sql in get_tick_table_sql():
                    cursor.execute(sql)

                conn.commit()
                logger.info(f"Tick table created: {db_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to create tick table: {e}")
            return False

    def create_ohlc_table(self, db_path: str) -> bool:
        """
        Create OHLC table and indexes if not exists

        Args:
            db_path: Path to database file

        Returns:
            True if successful
        """
        try:
            with self.get_connection(db_path) as conn:
                cursor = conn.cursor()

                for sql in get_ohlc_table_sql():
                    cursor.execute(sql)

                conn.commit()
                logger.info(f"OHLC table created: {db_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to create OHLC table: {e}")
            return False

    def save_ticks_batch(self, symbol: str, ticks: List[Dict[str, Any]]) -> int:
        """
        Save ticks in batch (optimized for performance)

        Batch size recommendation: 1000 ticks per batch

        Args:
            symbol: Trading symbol
            ticks: List of tick dictionaries with keys:
                   - timestamp (int): Unix timestamp in milliseconds
                   - date_readable (str): Date in YYYY-MM-DD format
                   - bid (float): Bid price
                   - ask (float): Ask price
                   - last (float): Last price
                   - volume (int): Volume
                   - flags (int): MT5 flags

        Returns:
            Number of ticks saved
        """
        if not ticks:
            return 0

        try:
            # Determine year and month from first tick
            first_timestamp = ticks[0]['timestamp']
            dt = datetime.fromtimestamp(first_timestamp / 1000)
            year = dt.year
            month = dt.month

            db_path = self.get_tick_db_path(symbol, year, month)
            self.create_tick_table(db_path)

            with self.get_connection(db_path) as conn:
                cursor = conn.cursor()

                # Prepare data for batch insert
                data = [
                    (
                        symbol,
                        tick['timestamp'],
                        tick['date_readable'],
                        tick['bid'],
                        tick['ask'],
                        tick['last'],
                        tick['volume'],
                        tick['flags']
                    )
                    for tick in ticks
                ]

                # Batch insert with transaction
                cursor.executemany(
                    """
                    INSERT INTO ticks
                    (symbol, timestamp, date_readable, bid, ask, last, volume, flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    data
                )

                conn.commit()
                logger.info(f"Saved {len(ticks)} ticks for {symbol} to {db_path}")
                return len(ticks)

        except Exception as e:
            logger.error(f"Failed to save ticks batch: {e}")
            return 0

    def save_ohlc_batch(self, symbol: str, timeframe: str, bars: List[Dict[str, Any]]) -> int:
        """
        Save OHLC bars in batch

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., 'M1', 'M5', 'H1')
            bars: List of bar dictionaries with keys:
                  - timestamp (int): Bar start time (Unix ms)
                  - open (float): Open price
                  - high (float): High price
                  - low (float): Low price
                  - close (float): Close price
                  - tick_volume (int): Tick volume
                  - spread (int): Spread
                  - real_volume (int): Real volume
                  - is_closed (int): 1 if closed, 0 if forming

        Returns:
            Number of bars saved
        """
        if not bars:
            return 0

        try:
            # Determine year from first bar
            first_timestamp = bars[0]['timestamp']
            dt = datetime.fromtimestamp(first_timestamp / 1000)
            year = dt.year

            db_path = self.get_ohlc_db_path(symbol, year)
            self.create_ohlc_table(db_path)

            with self.get_connection(db_path) as conn:
                cursor = conn.cursor()

                # Prepare data for batch insert
                data = [
                    (
                        symbol,
                        timeframe,
                        bar['timestamp'],
                        bar['open'],
                        bar['high'],
                        bar['low'],
                        bar['close'],
                        bar['tick_volume'],
                        bar.get('spread'),
                        bar.get('real_volume'),
                        bar['is_closed']
                    )
                    for bar in bars
                ]

                # Batch insert with transaction
                cursor.executemany(
                    """
                    INSERT INTO ohlc_data
                    (symbol, timeframe, timestamp, open, high, low, close,
                     tick_volume, spread, real_volume, is_closed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    data
                )

                conn.commit()
                logger.info(f"Saved {len(bars)} bars for {symbol} {timeframe} to {db_path}")
                return len(bars)

        except Exception as e:
            logger.error(f"Failed to save OHLC batch: {e}")
            return 0

    def get_ticks(
        self,
        symbol: str,
        from_time: int,
        to_time: int,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Query ticks from database

        Args:
            symbol: Trading symbol
            from_time: Start timestamp (Unix ms)
            to_time: End timestamp (Unix ms)
            limit: Maximum number of results

        Returns:
            List of tick dictionaries
        """
        ticks = []

        try:
            # Determine year/month range
            from_dt = datetime.fromtimestamp(from_time / 1000)
            to_dt = datetime.fromtimestamp(to_time / 1000)

            # Query across all relevant month databases
            current_dt = from_dt.replace(day=1)

            while current_dt <= to_dt:
                year = current_dt.year
                month = current_dt.month

                db_path = self.get_tick_db_path(symbol, year, month)

                if os.path.exists(db_path):
                    with self.get_connection(db_path) as conn:
                        cursor = conn.cursor()

                        cursor.execute(
                            """
                            SELECT timestamp, date_readable, bid, ask, last, volume, flags
                            FROM ticks
                            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                            ORDER BY timestamp ASC
                            LIMIT ?
                            """,
                            (symbol, from_time, to_time, limit)
                        )

                        for row in cursor.fetchall():
                            ticks.append({
                                'timestamp': row[0],
                                'date_readable': row[1],
                                'bid': row[2],
                                'ask': row[3],
                                'last': row[4],
                                'volume': row[5],
                                'flags': row[6]
                            })

                # Move to next month
                if current_dt.month == 12:
                    current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
                else:
                    current_dt = current_dt.replace(month=current_dt.month + 1)

                # Check limit
                if len(ticks) >= limit:
                    break

            logger.info(f"Retrieved {len(ticks)} ticks for {symbol}")
            return ticks[:limit]

        except Exception as e:
            logger.error(f"Failed to get ticks: {e}")
            return []

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        from_time: int,
        to_time: int,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Query OHLC bars from database

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., 'M1', 'M5')
            from_time: Start timestamp (Unix ms)
            to_time: End timestamp (Unix ms)
            limit: Maximum number of results

        Returns:
            List of OHLC bar dictionaries
        """
        bars = []

        try:
            from_dt = datetime.fromtimestamp(from_time / 1000)
            to_dt = datetime.fromtimestamp(to_time / 1000)

            # Query across all relevant year databases
            for year in range(from_dt.year, to_dt.year + 1):
                db_path = self.get_ohlc_db_path(symbol, year)

                if os.path.exists(db_path):
                    with self.get_connection(db_path) as conn:
                        cursor = conn.cursor()

                        cursor.execute(
                            """
                            SELECT timestamp, open, high, low, close,
                                   tick_volume, spread, real_volume, is_closed
                            FROM ohlc_data
                            WHERE symbol = ? AND timeframe = ?
                              AND timestamp >= ? AND timestamp <= ?
                            ORDER BY timestamp ASC
                            LIMIT ?
                            """,
                            (symbol, timeframe, from_time, to_time, limit)
                        )

                        for row in cursor.fetchall():
                            bars.append({
                                'timestamp': row[0],
                                'open': row[1],
                                'high': row[2],
                                'low': row[3],
                                'close': row[4],
                                'tick_volume': row[5],
                                'spread': row[6],
                                'real_volume': row[7],
                                'is_closed': row[8]
                            })

                if len(bars) >= limit:
                    break

            logger.info(f"Retrieved {len(bars)} bars for {symbol} {timeframe}")
            return bars[:limit]

        except Exception as e:
            logger.error(f"Failed to get OHLC: {e}")
            return []

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with statistics about all databases
        """
        stats = {
            'tick_databases': [],
            'ohlc_databases': [],
            'total_size_mb': 0
        }

        try:
            # Scan database directory
            for year_dir in os.listdir(self.base_dir):
                year_path = os.path.join(self.base_dir, year_dir)

                if not os.path.isdir(year_path):
                    continue

                for db_file in os.listdir(year_path):
                    db_path = os.path.join(year_path, db_file)

                    if not db_file.endswith('.db'):
                        continue

                    file_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
                    stats['total_size_mb'] += file_size

                    # Parse filename
                    if '_ticks_' in db_file:
                        parts = db_file.replace('.db', '').split('_ticks_')
                        symbol = parts[0]
                        month = parts[1]

                        # Get tick count
                        with self.get_connection(db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM ticks")
                            count = cursor.fetchone()[0]

                        stats['tick_databases'].append({
                            'symbol': symbol,
                            'year': year_dir,
                            'month': month,
                            'tick_count': count,
                            'file_size_mb': round(file_size, 2)
                        })

                    elif '_ohlc.db' in db_file:
                        symbol = db_file.replace('_ohlc.db', '')

                        # Get bar count per timeframe
                        with self.get_connection(db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT timeframe, COUNT(*) FROM ohlc_data GROUP BY timeframe"
                            )
                            timeframes = {row[0]: row[1] for row in cursor.fetchall()}

                        stats['ohlc_databases'].append({
                            'symbol': symbol,
                            'year': year_dir,
                            'timeframes': timeframes,
                            'file_size_mb': round(file_size, 2)
                        })

            stats['total_size_mb'] = round(stats['total_size_mb'], 2)

        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")

        return stats
