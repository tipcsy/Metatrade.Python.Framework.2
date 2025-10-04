from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# This will be set by main.py
get_db_manager = None


class TickBatch(BaseModel):
    """Batch of tick data to save"""
    symbol: str
    ticks: List[Dict[str, Any]]


class OHLCBatch(BaseModel):
    """Batch of OHLC data to save"""
    symbol: str
    timeframe: str
    bars: List[Dict[str, Any]]


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-service",
        "port": 5001
    }


@router.get("/statistics")
async def get_statistics():
    """Get database statistics"""
    try:
        db = get_db_manager()
        if not db:
            raise HTTPException(status_code=500, detail="DatabaseManager not initialized")

        stats = db.get_database_statistics()

        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ticks/save")
async def save_ticks(batch: TickBatch):
    """Save tick data batch"""
    try:
        db = get_db_manager()
        if not db:
            raise HTTPException(status_code=500, detail="DatabaseManager not initialized")

        count = db.save_ticks_batch(batch.symbol, batch.ticks)

        return {
            "success": True,
            "message": f"Saved {count} ticks for {batch.symbol}"
        }
    except Exception as e:
        logger.error(f"Error saving ticks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ohlc/save")
async def save_ohlc(batch: OHLCBatch):
    """Save OHLC data batch"""
    try:
        db = get_db_manager()
        if not db:
            raise HTTPException(status_code=500, detail="DatabaseManager not initialized")

        count = db.save_ohlc_batch(batch.symbol, batch.timeframe, batch.bars)

        return {
            "success": True,
            "message": f"Saved {count} bars for {batch.symbol} {batch.timeframe}"
        }
    except Exception as e:
        logger.error(f"Error saving OHLC: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ticks/query")
async def query_ticks(symbol: str, from_time: int, to_time: int, limit: int = 10000):
    """Query tick data"""
    try:
        db = get_db_manager()
        if not db:
            raise HTTPException(status_code=500, detail="DatabaseManager not initialized")

        ticks = db.get_ticks(symbol, from_time, to_time, limit)

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "count": len(ticks),
                "ticks": ticks
            }
        }
    except Exception as e:
        logger.error(f"Error querying ticks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ohlc/query")
async def query_ohlc(
    symbol: str,
    timeframe: str,
    from_time: int,
    to_time: int,
    limit: int = 10000
):
    """Query OHLC data"""
    try:
        db = get_db_manager()
        if not db:
            raise HTTPException(status_code=500, detail="DatabaseManager not initialized")

        bars = db.get_ohlc(symbol, timeframe, from_time, to_time, limit)

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(bars),
                "bars": bars
            }
        }
    except Exception as e:
        logger.error(f"Error querying OHLC: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
