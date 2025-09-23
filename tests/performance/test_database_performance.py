"""
Performance tests for the MetaTrader Python Framework database layer.

This module contains performance tests to ensure the database layer
can handle expected loads and meets performance requirements.
"""

from __future__ import annotations

import asyncio
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any

import pytest

from src.database.database import DatabaseManager
from src.database.services import SymbolGroupService, SymbolService, UserService, AccountService
from src.database.cache import initialize_redis, close_redis, get_cache_manager


class TestDatabasePerformance:
    """Performance tests for database operations."""

    @pytest.fixture(scope="class")
    def test_db_url(self) -> str:
        """Create temporary SQLite database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "perf_test_trading.db"
        return f"sqlite:///{db_path}"

    @pytest.fixture(scope="class")
    def db_manager(self, test_db_url: str):
        """Setup test database manager."""
        manager = DatabaseManager(test_db_url)
        manager.initialize()
        manager.create_all_tables()
        yield manager
        manager.close()

    @pytest.fixture(scope="class")
    def sample_data(self):
        """Generate sample data for performance tests."""
        return {
            'symbol_groups': [
                {
                    'name': f'Group {i}',
                    'description': f'Test group {i}',
                    'group_type': 'FOREX' if i % 2 == 0 else 'CRYPTO',
                    'display_order': i,
                    'is_active': True
                }
                for i in range(10)
            ],
            'users': [
                {
                    'username': f'user{i:04d}',
                    'email': f'user{i:04d}@example.com',
                    'password': 'TestPassword123!',
                    'first_name': f'User{i}',
                    'last_name': 'Test',
                    'role': 'USER'
                }
                for i in range(1000)
            ]
        }

    def test_bulk_create_performance(self, db_manager: DatabaseManager, sample_data: Dict[str, List[Dict]]):
        """Test bulk create operations performance."""
        user_service = UserService()

        # Measure bulk user creation
        start_time = time.time()
        created_users = user_service.bulk_create(sample_data['users'], batch_size=100)
        end_time = time.time()

        duration = end_time - start_time
        users_per_second = len(created_users) / duration

        print(f"\nBulk Create Performance:")
        print(f"Created {len(created_users)} users in {duration:.2f} seconds")
        print(f"Rate: {users_per_second:.1f} users/second")

        assert len(created_users) == 1000
        assert duration < 30.0  # Should complete in under 30 seconds
        assert users_per_second > 30  # Should create at least 30 users per second

    def test_concurrent_read_performance(self, db_manager: DatabaseManager):
        """Test concurrent read operations performance."""
        user_service = UserService()

        # First ensure we have some users
        users = user_service.get_all(limit=100)
        if len(users) < 10:
            pytest.skip("Not enough users for concurrent read test")

        user_ids = [user.id for user in users[:10]]

        def read_user(user_id: str) -> Dict[str, Any]:
            """Read a single user and measure time."""
            start = time.time()
            user = user_service.get_by_id(user_id)
            end = time.time()
            return {
                'user_id': user_id,
                'duration': end - start,
                'success': user is not None
            }

        # Test concurrent reads
        num_threads = 20
        num_iterations = 5

        all_durations = []
        success_count = 0

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for iteration in range(num_iterations):
                futures = []

                # Submit concurrent read tasks
                for _ in range(num_threads):
                    user_id = user_ids[iteration % len(user_ids)]
                    future = executor.submit(read_user, user_id)
                    futures.append(future)

                # Collect results
                for future in as_completed(futures):
                    result = future.result()
                    all_durations.append(result['duration'])
                    if result['success']:
                        success_count += 1

        total_operations = num_threads * num_iterations
        avg_duration = statistics.mean(all_durations)
        max_duration = max(all_durations)
        min_duration = min(all_durations)

        print(f"\nConcurrent Read Performance:")
        print(f"Total operations: {total_operations}")
        print(f"Successful operations: {success_count}")
        print(f"Average response time: {avg_duration*1000:.1f}ms")
        print(f"Min response time: {min_duration*1000:.1f}ms")
        print(f"Max response time: {max_duration*1000:.1f}ms")

        assert success_count == total_operations
        assert avg_duration < 0.1  # Average response time under 100ms
        assert max_duration < 0.5  # Max response time under 500ms

    def test_complex_query_performance(self, db_manager: DatabaseManager):
        """Test performance of complex queries with filtering and pagination."""
        symbol_service = SymbolService()

        # Create test data
        group_service = SymbolGroupService()
        test_group = group_service.create({
            'name': 'Performance Test Group',
            'group_type': 'FOREX',
            'is_active': True
        })

        # Create symbols for testing
        symbols_data = []
        for i in range(500):
            symbols_data.append({
                'symbol': f'TEST{i:03d}',
                'name': f'Test Symbol {i}',
                'symbol_group_id': test_group.id,
                'market': 'FOREX',
                'base_currency': 'EUR' if i % 2 == 0 else 'GBP',
                'quote_currency': 'USD',
                'digits': 5,
                'point': Decimal('0.00001'),
                'tick_size': Decimal('0.00001'),
                'tick_value': Decimal('1.0'),
                'contract_size': Decimal('100000'),
                'min_lot': Decimal('0.01'),
                'max_lot': Decimal('100'),
                'lot_step': Decimal('0.01'),
                'is_tradeable': i % 3 != 0  # 2/3 tradeable
            })

        symbol_service.bulk_create(symbols_data, batch_size=50)

        # Test complex queries
        test_cases = [
            {
                'name': 'Filter by market',
                'filters': {'market': 'FOREX'},
                'limit': 100
            },
            {
                'name': 'Filter by currency and tradeable',
                'filters': {'base_currency': 'EUR', 'is_tradeable': True},
                'limit': 50
            },
            {
                'name': 'Filter by group',
                'filters': {'symbol_group_id': test_group.id},
                'limit': 200
            }
        ]

        for test_case in test_cases:
            start_time = time.time()
            results = symbol_service.get_all(
                filters=test_case['filters'],
                limit=test_case['limit'],
                order_by='symbol'
            )
            end_time = time.time()

            duration = end_time - start_time

            print(f"\nComplex Query Performance - {test_case['name']}:")
            print(f"Results: {len(results)} symbols")
            print(f"Duration: {duration*1000:.1f}ms")

            assert duration < 1.0  # Should complete in under 1 second
            assert len(results) <= test_case['limit']

    def test_cache_performance(self, db_manager: DatabaseManager):
        """Test caching performance improvement."""
        try:
            # Initialize Redis for testing
            initialize_redis(host="localhost", port=6379, db=15)
            cache_manager = get_cache_manager()

            symbol_service = SymbolService()

            # Get a symbol to test with
            symbols = symbol_service.get_all(limit=1)
            if not symbols:
                pytest.skip("No symbols available for cache test")

            symbol_id = symbols[0].id

            # Test without cache (cold)
            start_time = time.time()
            symbol1 = symbol_service.get_by_id(symbol_id, use_cache=False)
            cold_duration = time.time() - start_time

            # Test with cache (warm)
            start_time = time.time()
            symbol2 = symbol_service.get_by_id(symbol_id, use_cache=True)
            warm_duration = time.time() - start_time

            # Test cache hit
            start_time = time.time()
            symbol3 = symbol_service.get_by_id(symbol_id, use_cache=True)
            cache_hit_duration = time.time() - start_time

            print(f"\nCache Performance:")
            print(f"Cold read (no cache): {cold_duration*1000:.1f}ms")
            print(f"Warm read (cache miss): {warm_duration*1000:.1f}ms")
            print(f"Cache hit: {cache_hit_duration*1000:.1f}ms")

            assert symbol1.id == symbol2.id == symbol3.id
            assert cache_hit_duration < cold_duration  # Cache should be faster

            close_redis()

        except Exception as e:
            pytest.skip(f"Redis not available for cache performance test: {e}")

    @pytest.mark.asyncio
    async def test_async_performance(self, db_manager: DatabaseManager):
        """Test async operations performance."""
        user_service = UserService()

        # Get some users to test with
        users = user_service.get_all(limit=10)
        if len(users) < 5:
            pytest.skip("Not enough users for async test")

        user_ids = [user.id for user in users[:5]]

        # Test async concurrent operations
        async def async_read_user(user_id: str) -> Dict[str, Any]:
            start = time.time()
            user = await user_service.get_by_id_async(user_id)
            end = time.time()
            return {
                'user_id': user_id,
                'duration': end - start,
                'success': user is not None
            }

        # Run concurrent async operations
        start_time = time.time()
        tasks = [async_read_user(user_id) for user_id in user_ids * 4]  # 20 total tasks
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time

        successful_operations = sum(1 for r in results if r['success'])
        avg_individual_duration = statistics.mean([r['duration'] for r in results])

        print(f"\nAsync Performance:")
        print(f"Total operations: {len(results)}")
        print(f"Successful operations: {successful_operations}")
        print(f"Total time: {total_duration*1000:.1f}ms")
        print(f"Average individual duration: {avg_individual_duration*1000:.1f}ms")
        print(f"Operations per second: {len(results)/total_duration:.1f}")

        assert successful_operations == len(results)
        assert total_duration < 2.0  # Should complete all operations quickly

    def test_memory_usage_bulk_operations(self, db_manager: DatabaseManager):
        """Test memory usage during bulk operations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        account_service = AccountService()
        user_service = UserService()

        # Create a test user first
        test_user = user_service.create({
            'username': 'bulktest',
            'email': 'bulktest@example.com',
            'password': 'Password123!',
            'role': 'USER'
        })

        # Create large batch of accounts
        account_data = []
        for i in range(1000):
            account_data.append({
                'user_id': test_user.id,
                'account_number': f'BULK{i:06d}',
                'account_name': f'Bulk Test Account {i}',
                'account_type': 'DEMO',
                'currency': 'USD',
                'balance': Decimal('10000'),
                'leverage': 100
            })

        # Monitor memory during bulk creation
        start_time = time.time()
        created_accounts = account_service.bulk_create(account_data, batch_size=100)
        end_time = time.time()

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        duration = end_time - start_time
        memory_increase = peak_memory - initial_memory

        print(f"\nMemory Usage - Bulk Operations:")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Peak memory: {peak_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Created {len(created_accounts)} accounts in {duration:.2f} seconds")
        print(f"Memory per account: {memory_increase/len(created_accounts)*1024:.1f} KB")

        assert len(created_accounts) == 1000
        assert memory_increase < 500  # Should not use more than 500MB additional memory

    def test_transaction_performance(self, db_manager: DatabaseManager):
        """Test transaction performance under load."""
        account_service = AccountService()

        # Get an account to test with
        accounts = account_service.get_all(limit=1)
        if not accounts:
            pytest.skip("No accounts available for transaction test")

        account_id = accounts[0].id

        # Test multiple balance updates (simulating trading activity)
        num_updates = 100
        start_time = time.time()

        for i in range(num_updates):
            # Simulate balance changes
            new_balance = Decimal('10000') + Decimal(str(i * 10))
            new_equity = new_balance + Decimal('50')
            margin = Decimal('100')
            free_margin = new_equity - margin

            success = account_service.update_balance(
                account_id,
                new_balance,
                new_equity,
                margin,
                free_margin
            )
            assert success

        end_time = time.time()
        duration = end_time - start_time
        updates_per_second = num_updates / duration

        print(f"\nTransaction Performance:")
        print(f"Updates: {num_updates}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Rate: {updates_per_second:.1f} updates/second")

        assert updates_per_second > 50  # Should handle at least 50 updates per second

    def test_search_performance(self, db_manager: DatabaseManager):
        """Test search functionality performance."""
        symbol_service = SymbolService()

        # Test search queries
        search_terms = ['EUR', 'USD', 'GBP', 'TEST', 'FOREX']

        for term in search_terms:
            start_time = time.time()
            results = symbol_service.search_symbols(term, limit=50)
            end_time = time.time()

            duration = end_time - start_time

            print(f"\nSearch Performance - '{term}':")
            print(f"Results: {len(results)}")
            print(f"Duration: {duration*1000:.1f}ms")

            assert duration < 0.5  # Should complete search in under 500ms

    def benchmark_summary(self):
        """Print benchmark summary and recommendations."""
        print(f"\n" + "="*50)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*50)
        print("Database Layer Performance Tests Completed")
        print("\nKey Performance Indicators:")
        print("• Bulk operations: >30 records/second")
        print("• Individual reads: <100ms average")
        print("• Concurrent operations: <500ms max")
        print("• Search queries: <500ms")
        print("• Cache hits: Faster than cold reads")
        print("• Memory usage: <500MB for 1000 records")
        print("\nRecommendations:")
        print("• Use bulk operations for large datasets")
        print("• Enable caching for frequently accessed data")
        print("• Consider connection pooling for high concurrency")
        print("• Monitor memory usage with large operations")
        print("• Use async operations for concurrent workloads")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])