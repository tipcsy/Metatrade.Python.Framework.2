"""
Comprehensive Phase 5 Integration Tests
Tests the complete trading system integration including ML pipeline, risk management, and performance.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.trading.trading_engine import TradingEngine
from src.trading.order_manager import OrderManager
from src.trading.risk_manager import RiskManager
from src.trading.ml_pipeline import MLPipeline
from src.trading.portfolio_optimizer import PortfolioOptimizer
from src.trading.data_processor import DataProcessor
from src.trading.metrics_collector import MetricsCollector


class TestPhase5Integration:
    """Complete integration tests for Phase 5 trading system"""

    @pytest.fixture
    async def trading_system(self):
        """Setup complete trading system for testing"""
        # Mock database connections
        db_mock = AsyncMock()

        # Initialize components
        risk_manager = RiskManager(db_mock)
        order_manager = OrderManager(db_mock)
        ml_pipeline = MLPipeline(db_mock)
        data_processor = DataProcessor(db_mock)
        metrics_collector = MetricsCollector(db_mock)

        trading_engine = TradingEngine(
            db=db_mock,
            risk_manager=risk_manager,
            order_manager=order_manager,
            ml_pipeline=ml_pipeline
        )

        # Start all components
        await trading_engine.start()
        await data_processor.start()
        await metrics_collector.start()

        yield {
            'engine': trading_engine,
            'risk_manager': risk_manager,
            'order_manager': order_manager,
            'ml_pipeline': ml_pipeline,
            'data_processor': data_processor,
            'metrics_collector': metrics_collector
        }

        # Cleanup
        await trading_engine.stop()
        await data_processor.stop()
        await metrics_collector.stop()

    async def test_complete_trading_workflow(self, trading_system):
        """Test complete trading workflow from signal to execution"""
        engine = trading_system['engine']
        ml_pipeline = trading_system['ml_pipeline']

        # Generate market data
        market_data = {
            'symbol': 'EURUSD',
            'bid': Decimal('1.1050'),
            'ask': Decimal('1.1052'),
            'timestamp': datetime.now(timezone.utc),
            'volume': 1000000
        }

        # Mock ML prediction
        with patch.object(ml_pipeline, 'predict', return_value={'signal': 0.7, 'confidence': 0.85}):
            # Process market data and generate signals
            await engine.process_market_data(market_data)

            # Verify signal generation
            signals = await engine.get_active_signals()
            assert len(signals) > 0
            assert signals[0]['symbol'] == 'EURUSD'
            assert signals[0]['signal'] > 0.5

    async def test_risk_management_integration(self, trading_system):
        """Test risk management blocks dangerous trades"""
        engine = trading_system['engine']
        risk_manager = trading_system['risk_manager']

        # Create high-risk order
        order = {
            'symbol': 'EURUSD',
            'side': 'BUY',
            'quantity': Decimal('10000000'),  # Very large position
            'order_type': 'MARKET',
            'strategy_id': 'test_strategy'
        }

        # Should be blocked by risk management
        result = await engine.submit_order(order)
        assert result['status'] == 'REJECTED'
        assert 'risk' in result['reason'].lower()

    async def test_ml_pipeline_performance(self, trading_system):
        """Test ML pipeline meets performance requirements"""
        ml_pipeline = trading_system['ml_pipeline']

        # Create test features
        features = np.random.randn(100, 20)  # 100 samples, 20 features

        # Measure prediction time
        start_time = asyncio.get_event_loop().time()
        predictions = await ml_pipeline.batch_predict(features)
        end_time = asyncio.get_event_loop().time()

        # Verify performance requirement (<1ms per prediction)
        avg_time_per_prediction = (end_time - start_time) / len(features)
        assert avg_time_per_prediction < 0.001  # <1ms
        assert len(predictions) == len(features)

    async def test_order_execution_latency(self, trading_system):
        """Test order execution meets latency requirements"""
        engine = trading_system['engine']

        # Create simple market order
        order = {
            'symbol': 'EURUSD',
            'side': 'BUY',
            'quantity': Decimal('100000'),
            'order_type': 'MARKET',
            'strategy_id': 'test_strategy'
        }

        # Measure execution time
        start_time = asyncio.get_event_loop().time()
        result = await engine.submit_order(order)
        end_time = asyncio.get_event_loop().time()

        # Verify latency requirement (<100μs target, allowing 1ms for test environment)
        execution_time = (end_time - start_time) * 1000000  # Convert to microseconds
        assert execution_time < 1000  # <1ms in test environment
        assert result['status'] in ['ACCEPTED', 'FILLED']

    async def test_data_processing_throughput(self, trading_system):
        """Test data processing meets throughput requirements"""
        data_processor = trading_system['data_processor']

        # Generate high volume of market data
        market_data_batch = []
        for i in range(10000):
            market_data_batch.append({
                'symbol': f'SYMBOL{i % 100}',
                'bid': Decimal(f'{1.0 + (i % 1000) * 0.0001:.4f}'),
                'ask': Decimal(f'{1.0002 + (i % 1000) * 0.0001:.4f}'),
                'timestamp': datetime.now(timezone.utc),
                'volume': 1000000 + i
            })

        # Measure processing time
        start_time = asyncio.get_event_loop().time()
        await data_processor.process_batch(market_data_batch)
        end_time = asyncio.get_event_loop().time()

        # Verify throughput requirement (>1M ticks/second target)
        processing_time = end_time - start_time
        throughput = len(market_data_batch) / processing_time
        assert throughput > 100000  # >100K/sec in test environment

    async def test_portfolio_optimization(self, trading_system):
        """Test portfolio optimization algorithms"""
        optimizer = PortfolioOptimizer()

        # Create sample portfolio data
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        returns_data = pd.DataFrame(
            np.random.randn(252, len(symbols)) * 0.01,  # Daily returns for 1 year
            columns=symbols
        )

        # Test different optimization methods
        for method in ['markowitz', 'black_litterman', 'risk_parity']:
            weights = await optimizer.optimize(returns_data, method=method)

            # Verify portfolio constraints
            assert abs(sum(weights.values()) - 1.0) < 0.001  # Weights sum to 1
            assert all(w >= 0 for w in weights.values())  # No short positions
            assert len(weights) == len(symbols)

    async def test_metrics_collection(self, trading_system):
        """Test comprehensive metrics collection"""
        metrics_collector = trading_system['metrics_collector']

        # Collect various metrics
        await metrics_collector.record_trade_latency(50)  # 50 microseconds
        await metrics_collector.record_pnl('EURUSD', Decimal('150.50'))
        await metrics_collector.record_risk_metric('var_95', Decimal('10000'))

        # Get aggregated metrics
        metrics = await metrics_collector.get_metrics_summary()

        # Verify metrics structure
        assert 'performance' in metrics
        assert 'risk' in metrics
        assert 'execution' in metrics
        assert metrics['execution']['avg_latency_us'] > 0

    async def test_system_fault_tolerance(self, trading_system):
        """Test system handles component failures gracefully"""
        engine = trading_system['engine']

        # Simulate database connection failure
        with patch.object(engine.db, 'execute', side_effect=Exception("Database error")):
            order = {
                'symbol': 'EURUSD',
                'side': 'BUY',
                'quantity': Decimal('100000'),
                'order_type': 'MARKET',
                'strategy_id': 'test_strategy'
            }

            # System should handle error gracefully
            result = await engine.submit_order(order)
            assert result['status'] == 'ERROR'
            assert 'database' in result['reason'].lower()

    async def test_concurrent_order_processing(self, trading_system):
        """Test system handles concurrent orders correctly"""
        engine = trading_system['engine']

        # Create multiple concurrent orders
        orders = []
        for i in range(100):
            orders.append({
                'symbol': 'EURUSD',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': Decimal('100000'),
                'order_type': 'MARKET',
                'strategy_id': f'test_strategy_{i}'
            })

        # Submit all orders concurrently
        tasks = [engine.submit_order(order) for order in orders]
        results = await asyncio.gather(*tasks)

        # Verify all orders were processed
        assert len(results) == len(orders)
        successful_orders = [r for r in results if r['status'] in ['ACCEPTED', 'FILLED']]
        assert len(successful_orders) > 0

    async def test_ml_model_retraining(self, trading_system):
        """Test ML model retraining workflow"""
        ml_pipeline = trading_system['ml_pipeline']

        # Generate training data
        features = np.random.randn(10000, 20)
        targets = np.random.choice([0, 1], 10000)

        # Test model retraining
        result = await ml_pipeline.retrain_models(features, targets)

        # Verify retraining success
        assert result['status'] == 'SUCCESS'
        assert result['model_accuracy'] > 0.5
        assert 'training_time' in result

    def test_configuration_validation(self, trading_system):
        """Test system validates configuration correctly"""
        engine = trading_system['engine']

        # Test with invalid configuration
        invalid_config = {
            'max_position_size': -1000,  # Negative value
            'risk_limit': 'invalid_string',  # Should be numeric
        }

        with pytest.raises(ValueError):
            engine.validate_configuration(invalid_config)

        # Test with valid configuration
        valid_config = {
            'max_position_size': 1000000,
            'risk_limit': 0.02,
            'max_daily_trades': 1000
        }

        # Should not raise exception
        engine.validate_configuration(valid_config)


@pytest.mark.performance
class TestPhase5Performance:
    """Performance-specific tests for Phase 5"""

    async def test_latency_requirements(self):
        """Test all latency requirements are met"""
        # Order submission: <50μs target
        # Risk validation: <10μs target
        # Cache access: <100μs target
        # ML inference: <1ms target

        # This would be implemented with actual performance monitoring
        # in production environment
        pass

    async def test_throughput_requirements(self):
        """Test all throughput requirements are met"""
        # Orders: 1M/second target
        # Market data: 10M messages/second target
        # Database writes: 500K/second target

        # This would be implemented with load testing tools
        # in production environment
        pass


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=src.trading",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])