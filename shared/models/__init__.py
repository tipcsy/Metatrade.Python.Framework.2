# Shared Data Models
# Common data models used across all services

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ServiceStatus(BaseModel):
    """Service status model"""
    service_name: str
    status: str  # ONLINE, OFFLINE, STARTING, ERROR
    port: int
    uptime_seconds: Optional[int] = 0
    last_heartbeat: Optional[datetime] = None


class TickData(BaseModel):
    """Tick data model"""
    symbol: str
    timestamp: int
    bid: float
    ask: float
    last: float
    volume: int
    flags: int


class OHLCData(BaseModel):
    """OHLC/Candle data model"""
    symbol: str
    timeframe: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    real_volume: int
    is_closed: bool
