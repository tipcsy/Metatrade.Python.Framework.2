from fastapi import APIRouter, HTTPException
from app.core.mt5_manager import get_mt5_manager
import MetaTrader5 as mt5
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    manager = get_mt5_manager()

    return {
        "status": "healthy",
        "service": "mt5-service",
        "port": 5002,
        "mt5_connected": manager.is_connected()
    }


@router.get("/connection/status")
async def get_connection_status():
    """Get MT5 connection status"""
    try:
        manager = get_mt5_manager()

        is_connected = manager.check_connection()
        terminal_info = manager.get_terminal_info()

        return {
            "success": True,
            "data": {
                "connected": is_connected,
                "terminal_info": terminal_info
            }
        }
    except Exception as e:
        logger.error(f"Error getting connection status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connection/reconnect")
async def reconnect_mt5():
    """Manually trigger MT5 reconnection"""
    try:
        manager = get_mt5_manager()

        success = manager.reconnect()

        if success:
            return {
                "success": True,
                "message": "MT5 reconnected successfully"
            }
        else:
            error = manager.get_last_error()
            return {
                "success": False,
                "message": f"Reconnection failed: {error}"
            }
    except Exception as e:
        logger.error(f"Error reconnecting MT5: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account/info")
async def get_account_info():
    """Get MT5 account information"""
    try:
        manager = get_mt5_manager()

        if not manager.is_connected():
            raise HTTPException(status_code=503, detail="MT5 not connected")

        account_info = mt5.account_info()
        if account_info is None:
            error = mt5.last_error()
            raise HTTPException(status_code=500, detail=f"Failed to get account info: {error}")

        return {
            "success": True,
            "data": {
                "login": account_info.login,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "profit": account_info.profit,
                "leverage": account_info.leverage,
                "currency": account_info.currency,
                "server": account_info.server,
                "company": account_info.company
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_symbols():
    """Get available trading symbols"""
    try:
        manager = get_mt5_manager()

        if not manager.is_connected():
            raise HTTPException(status_code=503, detail="MT5 not connected")

        symbols = mt5.symbols_get()
        if symbols is None:
            error = mt5.last_error()
            raise HTTPException(status_code=500, detail=f"Failed to get symbols: {error}")

        symbol_list = []
        for symbol in symbols:
            symbol_list.append({
                "name": symbol.name,
                "description": symbol.description,
                "path": symbol.path,
                "visible": symbol.visible,
                "digits": symbol.digits,
                "spread": symbol.spread
            })

        return {
            "success": True,
            "data": {
                "count": len(symbol_list),
                "symbols": symbol_list
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting symbols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
