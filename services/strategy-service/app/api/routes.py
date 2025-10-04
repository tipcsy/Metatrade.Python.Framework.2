from fastapi import APIRouter, HTTPException, Request
import logging
from typing import List, Optional

from app.models import (
    StrategyCreateRequest,
    StrategyResponse,
    PositionResponse,
    ClosePositionRequest,
    RiskStatusResponse,
    PositionStatisticsResponse,
    StrategyStatisticsResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "strategy-service",
        "port": 5003
    }


@router.get("/strategies", response_model=List[StrategyResponse])
async def get_strategies(request: Request):
    """Get all strategies"""
    logger.info("Getting all strategies")

    strategy_engine = request.app.state.strategy_engine
    strategies = strategy_engine.get_all_strategies()

    return [StrategyResponse(**s.get_status()) for s in strategies]


@router.post("/strategies", response_model=StrategyResponse)
async def create_strategy(strategy_request: StrategyCreateRequest, request: Request):
    """Create a new strategy"""
    logger.info(f"Creating strategy: {strategy_request.strategy_type} for {strategy_request.symbol}")

    try:
        strategy_engine = request.app.state.strategy_engine

        strategy = strategy_engine.create_strategy(
            strategy_type=strategy_request.strategy_type,
            symbol=strategy_request.symbol,
            timeframe=strategy_request.timeframe,
            parameters=strategy_request.parameters
        )

        return StrategyResponse(**strategy.get_status())

    except ValueError as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/strategies/statistics", response_model=StrategyStatisticsResponse)
async def get_strategy_statistics(request: Request):
    """Get strategy engine statistics"""
    logger.info("Getting strategy statistics")

    strategy_engine = request.app.state.strategy_engine
    stats = strategy_engine.get_statistics()

    return StrategyStatisticsResponse(**stats)


@router.get("/strategies/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: str, request: Request):
    """Get a specific strategy"""
    logger.info(f"Getting strategy {strategy_id}")

    strategy_engine = request.app.state.strategy_engine
    strategy = strategy_engine.get_strategy(strategy_id)

    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    return StrategyResponse(**strategy.get_status())


@router.post("/strategies/{strategy_id}/start")
async def start_strategy(strategy_id: str, request: Request):
    """Start a strategy"""
    logger.info(f"Starting strategy {strategy_id}")

    strategy_engine = request.app.state.strategy_engine
    success = strategy_engine.start_strategy(strategy_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    return {
        "success": True,
        "message": f"Strategy {strategy_id} started",
        "strategy_id": strategy_id
    }


@router.post("/strategies/{strategy_id}/stop")
async def stop_strategy(strategy_id: str, close_positions: bool = True, request: Request = None):
    """Stop a strategy"""
    logger.info(f"Stopping strategy {strategy_id} (close_positions={close_positions})")

    strategy_engine = request.app.state.strategy_engine
    success = strategy_engine.stop_strategy(strategy_id, close_positions=close_positions)

    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    return {
        "success": True,
        "message": f"Strategy {strategy_id} stopped",
        "strategy_id": strategy_id,
        "positions_closed": close_positions
    }


@router.post("/strategies/{strategy_id}/pause")
async def pause_strategy(strategy_id: str, request: Request):
    """Pause a strategy"""
    logger.info(f"Pausing strategy {strategy_id}")

    strategy_engine = request.app.state.strategy_engine
    success = strategy_engine.pause_strategy(strategy_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    return {
        "success": True,
        "message": f"Strategy {strategy_id} paused",
        "strategy_id": strategy_id
    }


@router.post("/strategies/{strategy_id}/resume")
async def resume_strategy(strategy_id: str, request: Request):
    """Resume a paused strategy"""
    logger.info(f"Resuming strategy {strategy_id}")

    strategy_engine = request.app.state.strategy_engine
    success = strategy_engine.resume_strategy(strategy_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    return {
        "success": True,
        "message": f"Strategy {strategy_id} resumed",
        "strategy_id": strategy_id
    }


@router.delete("/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str, request: Request):
    """Delete a strategy"""
    logger.info(f"Deleting strategy {strategy_id}")

    strategy_engine = request.app.state.strategy_engine
    success = strategy_engine.delete_strategy(strategy_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    return {
        "success": True,
        "message": f"Strategy {strategy_id} deleted",
        "strategy_id": strategy_id
    }


@router.get("/positions/statistics", response_model=PositionStatisticsResponse)
async def get_position_statistics(request: Request):
    """Get position statistics"""
    logger.info("Getting position statistics")

    position_manager = request.app.state.position_manager
    stats = position_manager.get_statistics()

    return PositionStatisticsResponse(**stats)


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    strategy_id: Optional[str] = None,
    request: Request = None
):
    """Get positions with optional filters"""
    logger.info(f"Getting positions (status={status}, symbol={symbol}, strategy_id={strategy_id})")

    position_manager = request.app.state.position_manager

    if status == "open":
        positions = position_manager.get_open_positions(symbol=symbol, strategy_id=strategy_id)
    elif status == "closed":
        positions = position_manager.get_closed_positions(strategy_id=strategy_id)
    else:
        # Get all positions
        positions = list(position_manager.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        if strategy_id:
            positions = [p for p in positions if p.strategy_id == strategy_id]

    return [PositionResponse(**p.to_dict()) for p in positions]


@router.get("/positions/{position_id}", response_model=PositionResponse)
async def get_position(position_id: str, request: Request):
    """Get a specific position"""
    logger.info(f"Getting position {position_id}")

    position_manager = request.app.state.position_manager
    position = position_manager.get_position(position_id)

    if not position:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

    return PositionResponse(**position.to_dict())


@router.post("/positions/close")
async def close_position(close_request: ClosePositionRequest, request: Request):
    """Close a position"""
    logger.info(f"Closing position {close_request.position_id}")

    position_manager = request.app.state.position_manager

    # Use provided close price or default to current price (mock mode)
    close_price = close_request.close_price if close_request.close_price else 1.1000

    success = position_manager.close_position(close_request.position_id, close_price)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to close position {close_request.position_id}"
        )

    return {
        "success": True,
        "message": f"Position {close_request.position_id} closed",
        "position_id": close_request.position_id,
        "close_price": close_price
    }


@router.post("/positions/close-all")
async def close_all_positions(strategy_id: Optional[str] = None, request: Request = None):
    """Close all open positions"""
    logger.info(f"Closing all positions (strategy_id={strategy_id})")

    position_manager = request.app.state.position_manager

    # Mock current prices
    current_prices = {"EURUSD": 1.1000, "GBPUSD": 1.2500, "USDJPY": 110.00}

    if strategy_id:
        closed_count = position_manager.close_positions_by_strategy(strategy_id, current_prices)
    else:
        closed_count = position_manager.close_all_positions(current_prices)

    return {
        "success": True,
        "message": f"Closed {closed_count} positions",
        "closed_count": closed_count
    }


@router.get("/risk/status", response_model=RiskStatusResponse)
async def get_risk_status(request: Request):
    """Get current risk status"""
    logger.info("Getting risk status")

    risk_manager = request.app.state.risk_manager
    status = risk_manager.get_risk_status()

    return RiskStatusResponse(**status)


@router.post("/risk/reset-daily-loss")
async def reset_daily_loss(request: Request):
    """Reset daily loss counter"""
    logger.info("Resetting daily loss")

    risk_manager = request.app.state.risk_manager
    risk_manager.reset_daily_loss()

    return {
        "success": True,
        "message": "Daily loss reset to 0"
    }


@router.post("/risk/update-balance")
async def update_balance(new_balance: float, request: Request):
    """Update account balance"""
    logger.info(f"Updating account balance to {new_balance}")

    risk_manager = request.app.state.risk_manager
    risk_manager.update_account_balance(new_balance)

    return {
        "success": True,
        "message": f"Account balance updated to ${new_balance:.2f}",
        "new_balance": new_balance
    }
