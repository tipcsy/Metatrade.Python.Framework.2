from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class ServiceStatus(str, Enum):
    """Service állapot típusok"""
    ONLINE = "online"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


class ServiceConfig(BaseModel):
    """Service konfiguráció modell"""
    name: str
    port: int
    path: str
    command: str
    auto_start: bool = False
    restart_on_failure: bool = True
    health_endpoint: str = "/health"


class ServiceInfo(BaseModel):
    """Service információ modell (runtime állapot)"""
    name: str
    status: ServiceStatus = ServiceStatus.OFFLINE
    port: int
    pid: Optional[int] = None
    uptime: Optional[int] = None  # másodpercekben
    last_check: Optional[datetime] = None
    error: Optional[str] = None
    restart_count: int = 0
    config: ServiceConfig


class ServiceStatusResponse(BaseModel):
    """Service státusz válasz modell"""
    name: str
    status: str
    port: int
    uptime: Optional[int] = None
    last_check: Optional[str] = None
    error: Optional[str] = None


class ServiceActionResponse(BaseModel):
    """Service művelet válasz modell"""
    success: bool
    message: str
    service: Optional[str] = None
