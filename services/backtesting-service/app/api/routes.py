"""
Backtesting Service API Routes

Complete REST API for backtesting and strategy optimization
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
import logging

from ..models.schemas import (
    BacktestRequest,
    BacktestResult,
    BacktestStatus,
    BacktestListItem,
    BacktestDeleteResponse,
    OptimizationRequest,
    OptimizationResult
)
from ..core.backtest_engine import BacktestEngine

router = APIRouter()
logger = logging.getLogger(__name__)

# Global backtest engine instance
backtest_engine = BacktestEngine(data_service_url="http://localhost:5001")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "backtesting-service",
        "port": 5006,
        "version": "1.0.0"
    }


@router.post("/backtest/start", response_model=dict)
async def start_backtest(request: BacktestRequest):
    """
    Start a new backtest.

    Args:
        request: Backtest configuration

    Returns:
        Backtest ID for tracking
    """
    try:
        logger.info(
            f"Backtest requested: {request.strategy_type.value} on "
            f"{request.symbol} {request.timeframe}"
        )

        backtest_id = await backtest_engine.run_backtest(request)

        return {
            "success": True,
            "data": {
                "backtest_id": backtest_id,
                "message": "Backtest started successfully"
            }
        }
    except Exception as e:
        logger.error(f"Failed to start backtest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")


@router.get("/backtest/{backtest_id}", response_model=dict)
async def get_backtest_result(backtest_id: str):
    """
    Get complete backtest result.

    Args:
        backtest_id: Backtest ID

    Returns:
        Complete backtest result with performance, trades, and equity curve
    """
    try:
        result = backtest_engine.get_backtest(backtest_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get backtest: {str(e)}")


@router.get("/backtest/{backtest_id}/status", response_model=dict)
async def get_backtest_status(backtest_id: str):
    """
    Get backtest status.

    Args:
        backtest_id: Backtest ID

    Returns:
        Current status of the backtest
    """
    try:
        result = backtest_engine.get_backtest(backtest_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        return {
            "success": True,
            "data": {
                "backtest_id": backtest_id,
                "status": result["status"],
                "progress": result.get("progress", 0),
                "error_message": result.get("error_message")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/backtests", response_model=dict)
async def list_backtests(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    strategy_type: Optional[str] = None
):
    """
    List all backtests with optional filtering.

    Args:
        limit: Maximum number of results
        offset: Offset for pagination
        status: Filter by status
        strategy_type: Filter by strategy type

    Returns:
        List of backtest summaries
    """
    try:
        backtests = backtest_engine.list_backtests(
            limit=limit,
            offset=offset,
            status=status,
            strategy_type=strategy_type
        )

        return {
            "success": True,
            "data": {
                "backtests": backtests,
                "total": len(backtests),
                "limit": limit,
                "offset": offset
            }
        }
    except Exception as e:
        logger.error(f"Failed to list backtests: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")


@router.delete("/backtest/{backtest_id}", response_model=BacktestDeleteResponse)
async def delete_backtest(backtest_id: str):
    """
    Delete a backtest.

    Args:
        backtest_id: Backtest ID

    Returns:
        Deletion confirmation
    """
    try:
        success = backtest_engine.delete_backtest(backtest_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        return BacktestDeleteResponse(
            success=True,
            message="Backtest deleted successfully",
            backtest_id=backtest_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete backtest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete backtest: {str(e)}")


@router.post("/backtest/{backtest_id}/stop", response_model=dict)
async def stop_backtest(backtest_id: str):
    """
    Stop a running backtest.

    Args:
        backtest_id: Backtest ID

    Returns:
        Stop confirmation
    """
    try:
        success = backtest_engine.stop_backtest(backtest_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Backtest {backtest_id} not found or not running"
            )

        return {
            "success": True,
            "data": {
                "backtest_id": backtest_id,
                "message": "Backtest stopped"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop backtest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop backtest: {str(e)}")


@router.post("/optimize/start", response_model=dict)
async def start_optimization(request: OptimizationRequest):
    """
    Start parameter optimization.

    Args:
        request: Optimization configuration

    Returns:
        Optimization ID for tracking
    """
    try:
        logger.info(
            f"Optimization requested: {request.strategy_type.value} on "
            f"{request.symbol} {request.timeframe}"
        )

        optimization_id = await backtest_engine.run_optimization(request)

        return {
            "success": True,
            "data": {
                "optimization_id": optimization_id,
                "message": "Optimization started successfully"
            }
        }
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")


@router.get("/optimize/{optimization_id}", response_model=dict)
async def get_optimization_result(optimization_id: str):
    """
    Get optimization result.

    Args:
        optimization_id: Optimization ID

    Returns:
        Complete optimization result with best parameters
    """
    try:
        result = backtest_engine.get_optimization(optimization_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization {optimization_id} not found"
            )

        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get optimization: {str(e)}")


@router.get("/optimize/{optimization_id}/status", response_model=dict)
async def get_optimization_status(optimization_id: str):
    """
    Get optimization status.

    Args:
        optimization_id: Optimization ID

    Returns:
        Current status and progress
    """
    try:
        result = backtest_engine.get_optimization(optimization_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization {optimization_id} not found"
            )

        return {
            "success": True,
            "data": {
                "optimization_id": optimization_id,
                "status": result["status"],
                "progress": result.get("completed_combinations", 0) / max(result.get("total_combinations", 1), 1) * 100,
                "completed_combinations": result.get("completed_combinations", 0),
                "total_combinations": result.get("total_combinations", 0),
                "error_message": result.get("error_message")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/strategies", response_model=dict)
async def list_available_strategies():
    """
    List available strategy types.

    Returns:
        List of available strategies with their parameters
    """
    strategies = [
        {
            "type": "MA_CROSSOVER",
            "name": "Moving Average Crossover",
            "description": "Simple MA crossover strategy",
            "parameters": {
                "fast_period": {"type": "int", "min": 5, "max": 50, "default": 10},
                "slow_period": {"type": "int", "min": 20, "max": 200, "default": 30}
            }
        },
        {
            "type": "RSI",
            "name": "RSI Strategy",
            "description": "RSI overbought/oversold strategy",
            "parameters": {
                "period": {"type": "int", "min": 5, "max": 30, "default": 14},
                "overbought": {"type": "float", "min": 60, "max": 90, "default": 70},
                "oversold": {"type": "float", "min": 10, "max": 40, "default": 30}
            }
        }
    ]

    return {
        "success": True,
        "data": {
            "strategies": strategies
        }
    }


@router.get("/metrics/definitions", response_model=dict)
async def get_metric_definitions():
    """
    Get definitions of all performance metrics.

    Returns:
        Dictionary of metric definitions and calculations
    """
    metrics = {
        "sharpe_ratio": {
            "name": "Sharpe Ratio",
            "description": "Risk-adjusted return (annualized return / std deviation)",
            "higher_is_better": True
        },
        "profit_factor": {
            "name": "Profit Factor",
            "description": "Gross profit / Gross loss",
            "higher_is_better": True
        },
        "max_drawdown": {
            "name": "Maximum Drawdown",
            "description": "Largest peak-to-trough decline",
            "higher_is_better": False
        },
        "win_rate": {
            "name": "Win Rate",
            "description": "Percentage of winning trades",
            "higher_is_better": True
        },
        "expectancy": {
            "name": "Expectancy",
            "description": "Average profit per trade",
            "higher_is_better": True
        }
    }

    return {
        "success": True,
        "data": {
            "metrics": metrics
        }
    }
