"""
Database schema definitions for Data Service

This module contains all SQL schema definitions for:
- Tick databases (partitioned by symbol and month)
- OHLC databases (partitioned by symbol and year)
- Completeness monitoring database
"""

# Tick database schema
TICK_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    date_readable TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    last REAL NOT NULL,
    volume INTEGER NOT NULL,
    flags INTEGER NOT NULL
);
"""

TICK_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_timestamp ON ticks(timestamp);
"""

TICK_SYMBOL_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON ticks(symbol, timestamp);
"""

# OHLC database schema
OHLC_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlc_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    tick_volume INTEGER NOT NULL,
    spread INTEGER,
    real_volume INTEGER,
    is_closed INTEGER NOT NULL
);
"""

OHLC_INDEX = """
CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp
ON ohlc_data(symbol, timeframe, timestamp);
"""

# Completeness database schema
TICK_COMPLETENESS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS tick_data_completeness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date_readable TEXT NOT NULL,
    status TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    first_timestamp INTEGER,
    last_timestamp INTEGER,
    last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date_readable)
);
"""

OHLC_COMPLETENESS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlc_data_completeness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    date_readable TEXT NOT NULL,
    status TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    expected_records INTEGER,
    last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, date_readable)
);
"""

# SQLite pragmas for performance optimization
SQLITE_PRAGMAS = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;
"""

def get_tick_table_sql() -> list[str]:
    """Get SQL statements to create tick table and indexes"""
    return [
        TICK_TABLE_SCHEMA,
        TICK_TIMESTAMP_INDEX,
        TICK_SYMBOL_TIMESTAMP_INDEX
    ]

def get_ohlc_table_sql() -> list[str]:
    """Get SQL statements to create OHLC table and indexes"""
    return [
        OHLC_TABLE_SCHEMA,
        OHLC_INDEX
    ]

def get_completeness_tables_sql() -> list[str]:
    """Get SQL statements to create completeness tables"""
    return [
        TICK_COMPLETENESS_TABLE_SCHEMA,
        OHLC_COMPLETENESS_TABLE_SCHEMA
    ]

def get_performance_pragmas() -> list[str]:
    """Get SQLite performance optimization pragmas"""
    return [
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = NORMAL",
        "PRAGMA cache_size = 10000",
        "PRAGMA temp_store = MEMORY"
    ]
