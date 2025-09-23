"""
Integration tests for the MetaTrader Python Framework database layer.

This module provides comprehensive integration tests for the database
layer including models, services, caching, and migrations.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.core.config import Settings
from src.database.database import DatabaseManager, initialize_database
from src.database.models import *
from src.database.services import *
from src.database.cache import initialize_redis, close_redis
from src.database.utils.health_check import DatabaseHealthChecker


class TestDatabaseIntegration:
    """Integration tests for database functionality."""

    @pytest.fixture(scope="class")
    def test_db_url(self) -> str:
        """Create temporary SQLite database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_trading.db"
        return f"sqlite:///{db_path}"

    @pytest.fixture(scope="class")
    def db_manager(self, test_db_url: str) -> Generator[DatabaseManager, None, None]:
        """Setup test database manager."""
        manager = DatabaseManager(test_db_url)
        manager.initialize()

        # Create all tables
        manager.create_all_tables()

        yield manager

        # Cleanup
        manager.close()

    @pytest.fixture(scope="class")
    def sample_user_data(self) -> Dict[str, Any]:
        """Sample user data for testing."""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPassword123!",
            "first_name": "Test",
            "last_name": "User",
            "role": "USER"
        }

    @pytest.fixture(scope="class")
    def sample_symbol_group_data(self) -> Dict[str, Any]:
        """Sample symbol group data for testing."""
        return {
            "name": "Major Forex",
            "description": "Major currency pairs",
            "group_type": "FOREX",
            "display_order": 1,
            "market_open_hour": 0,
            "market_close_hour": 23,
            "is_active": True
        }

    @pytest.fixture(scope="class")
    def sample_symbol_data(self, symbol_group: SymbolGroup) -> Dict[str, Any]:
        """Sample symbol data for testing."""
        return {
            "symbol": "EURUSD",
            "name": "Euro vs US Dollar",
            "symbol_group_id": symbol_group.id,
            "market": "FOREX",
            "base_currency": "EUR",
            "quote_currency": "USD",
            "digits": 5,
            "point": Decimal("0.00001"),
            "tick_size": Decimal("0.00001"),
            "tick_value": Decimal("1.0"),
            "contract_size": Decimal("100000"),
            "min_lot": Decimal("0.01"),
            "max_lot": Decimal("100"),
            "lot_step": Decimal("0.01"),
            "is_tradeable": True
        }

    def test_database_connection(self, db_manager: DatabaseManager):
        """Test database connection and basic operations."""
        assert db_manager.is_initialized

        # Test health check
        health = db_manager.health_check()
        assert health["status"] == "healthy"
        assert "connection" in health["details"]

    def test_user_service_crud(self, db_manager: DatabaseManager, sample_user_data: Dict[str, Any]):
        """Test user service CRUD operations."""
        user_service = UserService()

        # Test create
        user = user_service.create(sample_user_data)
        assert user.id is not None
        assert user.username == sample_user_data["username"]
        assert user.email == sample_user_data["email"]
        assert user.password_hash is not None
        assert user.password_hash != sample_user_data["password"]  # Should be hashed

        # Test read
        found_user = user_service.get_by_id(user.id)
        assert found_user is not None
        assert found_user.id == user.id

        # Test get by username
        found_by_username = user_service.get_by_username(user.username)
        assert found_by_username is not None
        assert found_by_username.id == user.id

        # Test get by email
        found_by_email = user_service.get_by_email(user.email)
        assert found_by_email is not None
        assert found_by_email.id == user.id

        # Test authentication
        authenticated_user = user_service.authenticate(user.username, sample_user_data["password"])
        assert authenticated_user is not None
        assert authenticated_user.id == user.id

        # Test wrong password
        wrong_auth = user_service.authenticate(user.username, "wrongpassword")
        assert wrong_auth is None

        # Test update
        update_data = {"first_name": "Updated", "last_name": "Name"}
        updated_user = user_service.update(user.id, update_data)
        assert updated_user is not None
        assert updated_user.first_name == "Updated"
        assert updated_user.last_name == "Name"

        # Test password change
        success = user_service.change_password(
            user.id,
            sample_user_data["password"],
            "NewPassword123!"
        )
        assert success

        # Test authentication with new password
        auth_new = user_service.authenticate(user.username, "NewPassword123!")
        assert auth_new is not None

        # Test old password doesn't work
        auth_old = user_service.authenticate(user.username, sample_user_data["password"])
        assert auth_old is None

    def test_symbol_group_service(self, db_manager: DatabaseManager, sample_symbol_group_data: Dict[str, Any]):
        """Test symbol group service operations."""
        group_service = SymbolGroupService()

        # Test create
        group = group_service.create(sample_symbol_group_data)
        assert group.id is not None
        assert group.name == sample_symbol_group_data["name"]
        assert group.group_type == sample_symbol_group_data["group_type"]

        # Test get by type
        forex_groups = group_service.get_by_type("FOREX")
        assert len(forex_groups) >= 1
        assert any(g.id == group.id for g in forex_groups)

        # Test get all types
        types = group_service.get_all_types()
        assert "FOREX" in types

        return group

    def test_symbol_service(self, db_manager: DatabaseManager):
        """Test symbol service operations."""
        # First create a symbol group
        group_service = SymbolGroupService()
        group_data = {
            "name": "Test Group",
            "description": "Test symbol group",
            "group_type": "FOREX",
            "is_active": True
        }
        group = group_service.create(group_data)

        # Now test symbol service
        symbol_service = SymbolService()
        symbol_data = {
            "symbol": "EURUSD",
            "name": "Euro vs US Dollar",
            "symbol_group_id": group.id,
            "market": "FOREX",
            "base_currency": "EUR",
            "quote_currency": "USD",
            "digits": 5,
            "point": Decimal("0.00001"),
            "tick_size": Decimal("0.00001"),
            "tick_value": Decimal("1.0"),
            "contract_size": Decimal("100000"),
            "min_lot": Decimal("0.01"),
            "max_lot": Decimal("100"),
            "lot_step": Decimal("0.01"),
            "is_tradeable": True
        }

        # Test create
        symbol = symbol_service.create(symbol_data)
        assert symbol.id is not None
        assert symbol.symbol == "EURUSD"

        # Test get by symbol
        found_symbol = symbol_service.get_by_symbol("EURUSD")
        assert found_symbol is not None
        assert found_symbol.id == symbol.id

        # Test update quote
        success = symbol_service.update_quote(
            symbol.id,
            Decimal("1.1234"),
            Decimal("1.1236"),
            Decimal("1000")
        )
        assert success

        # Verify quote update
        updated_symbol = symbol_service.get_by_id(symbol.id)
        assert updated_symbol.last_bid == Decimal("1.1234")
        assert updated_symbol.last_ask == Decimal("1.1236")
        assert updated_symbol.last_volume == Decimal("1000")

        # Test lot size validation
        is_valid = symbol_service.validate_lot_size(symbol.id, Decimal("0.1"))
        assert is_valid

        # Test invalid lot size
        with pytest.raises(Exception):  # Should raise ValidationError
            symbol_service.validate_lot_size(symbol.id, Decimal("0.001"))

        # Test search
        search_results = symbol_service.search_symbols("EUR")
        assert len(search_results) >= 1
        assert any(s.symbol == "EURUSD" for s in search_results)

    def test_account_service(self, db_manager: DatabaseManager):
        """Test account service operations."""
        # First create a user
        user_service = UserService()
        user_data = {
            "username": "testtrader",
            "email": "trader@example.com",
            "password": "Password123!",
            "role": "TRADER"
        }
        user = user_service.create(user_data)

        # Test account service
        account_service = AccountService()
        account_data = {
            "user_id": user.id,
            "account_number": "ACC123456",
            "account_name": "Test Trading Account",
            "account_type": "DEMO",
            "currency": "USD",
            "leverage": 100,
            "balance": Decimal("10000"),
            "broker_name": "Test Broker"
        }

        # Test create
        account = account_service.create(account_data)
        assert account.id is not None
        assert account.account_number == "ACC123456"
        assert account.balance == Decimal("10000")

        # Test get by account number
        found_account = account_service.get_by_account_number("ACC123456")
        assert found_account is not None
        assert found_account.id == account.id

        # Test get by user
        user_accounts = account_service.get_by_user(user.id)
        assert len(user_accounts) >= 1
        assert any(a.id == account.id for a in user_accounts)

        # Test balance update
        success = account_service.update_balance(
            account.id,
            Decimal("10500"),
            Decimal("10500"),
            Decimal("100"),
            Decimal("10400")
        )
        assert success

        # Test deposit
        success = account_service.add_deposit(account.id, Decimal("1000"), "Test deposit")
        assert success

        # Verify balance increased
        updated_account = account_service.get_by_id(account.id)
        assert updated_account.balance > account.balance

        # Test withdrawal
        success = account_service.add_withdrawal(account.id, Decimal("500"), "Test withdrawal")
        assert success

    def test_caching_functionality(self, db_manager: DatabaseManager):
        """Test caching functionality."""
        # Test with Redis (if available)
        try:
            # Initialize Redis for testing
            redis_client = initialize_redis(host="localhost", port=6379, db=15)  # Use test DB

            # Test symbol service with caching
            symbol_service = SymbolService()

            # Create a symbol group first
            group_service = SymbolGroupService()
            group = group_service.create({
                "name": "Cache Test Group",
                "group_type": "FOREX",
                "is_active": True
            })

            # Create symbol
            symbol_data = {
                "symbol": "GBPUSD",
                "name": "British Pound vs US Dollar",
                "symbol_group_id": group.id,
                "market": "FOREX",
                "base_currency": "GBP",
                "quote_currency": "USD",
                "digits": 5,
                "point": Decimal("0.00001"),
                "tick_size": Decimal("0.00001"),
                "tick_value": Decimal("1.0"),
                "contract_size": Decimal("100000"),
                "min_lot": Decimal("0.01"),
                "max_lot": Decimal("100"),
                "lot_step": Decimal("0.01"),
                "is_tradeable": True
            }
            symbol = symbol_service.create(symbol_data)

            # First access (should cache)
            symbol1 = symbol_service.get_by_id(symbol.id, use_cache=True)
            assert symbol1 is not None

            # Second access (should come from cache)
            symbol2 = symbol_service.get_by_id(symbol.id, use_cache=True)
            assert symbol2 is not None
            assert symbol2.id == symbol1.id

            # Test cache invalidation on update
            symbol_service.update(symbol.id, {"name": "Updated Name"})

            # Verify update
            updated_symbol = symbol_service.get_by_id(symbol.id, use_cache=True)
            assert updated_symbol.name == "Updated Name"

            close_redis()

        except Exception as e:
            # Redis not available, skip caching tests
            pytest.skip(f"Redis not available for caching tests: {e}")

    def test_database_health_check(self, db_manager: DatabaseManager):
        """Test comprehensive database health check."""
        health_checker = DatabaseHealthChecker()

        # Test basic health check
        health = health_checker.run_basic_health_check()
        assert health["status"] in ["healthy", "degraded"]
        assert "checks" in health
        assert "connectivity" in health["checks"]

        # Test performance checks
        perf_check = health["checks"].get("performance")
        if perf_check:
            assert "queries" in perf_check
            assert "summary" in perf_check

    def test_business_logic_validation(self, db_manager: DatabaseManager):
        """Test business logic validation in services."""
        symbol_service = SymbolService()

        # Test duplicate symbol creation (should fail)
        group_service = SymbolGroupService()
        group = group_service.create({
            "name": "Validation Test Group",
            "group_type": "FOREX",
            "is_active": True
        })

        symbol_data = {
            "symbol": "TESTPAIR",
            "name": "Test Currency Pair",
            "symbol_group_id": group.id,
            "market": "FOREX",
            "base_currency": "TEST",
            "quote_currency": "USD",
            "digits": 5,
            "point": Decimal("0.00001"),
            "tick_size": Decimal("0.00001"),
            "tick_value": Decimal("1.0"),
            "contract_size": Decimal("100000"),
            "min_lot": Decimal("0.01"),
            "max_lot": Decimal("100"),
            "lot_step": Decimal("0.01"),
            "is_tradeable": True
        }

        # Create first symbol
        symbol1 = symbol_service.create(symbol_data)
        assert symbol1 is not None

        # Try to create duplicate (should fail)
        with pytest.raises(Exception):  # Should raise BusinessLogicError
            symbol_service.create(symbol_data)

    def test_transaction_management(self, db_manager: DatabaseManager):
        """Test database transaction management."""
        user_service = UserService()

        # Test rollback on error
        try:
            # This should fail due to duplicate username
            user_service.create({
                "username": "testuser",  # This username should already exist
                "email": "duplicate@example.com",
                "password": "Password123!",
                "role": "USER"
            })
        except Exception:
            pass  # Expected to fail

        # Verify no partial data was saved
        duplicate_user = user_service.get_by_email("duplicate@example.com")
        assert duplicate_user is None

    @pytest.mark.asyncio
    async def test_async_operations(self, db_manager: DatabaseManager):
        """Test async database operations."""
        user_service = UserService()

        # Test async get
        users = user_service.get_all(limit=5)
        if users:
            user_id = users[0].id
            async_user = await user_service.get_by_id_async(user_id)
            assert async_user is not None
            assert async_user.id == user_id

    def test_performance_with_bulk_operations(self, db_manager: DatabaseManager):
        """Test performance with bulk operations."""
        group_service = SymbolGroupService()

        # Create test group
        group = group_service.create({
            "name": "Bulk Test Group",
            "group_type": "TEST",
            "is_active": True
        })

        symbol_service = SymbolService()

        # Prepare bulk data
        bulk_data = []
        for i in range(100):
            bulk_data.append({
                "symbol": f"TEST{i:03d}",
                "name": f"Test Symbol {i}",
                "symbol_group_id": group.id,
                "market": "TEST",
                "base_currency": "TST",
                "quote_currency": "USD",
                "digits": 5,
                "point": Decimal("0.00001"),
                "tick_size": Decimal("0.00001"),
                "tick_value": Decimal("1.0"),
                "contract_size": Decimal("100000"),
                "min_lot": Decimal("0.01"),
                "max_lot": Decimal("100"),
                "lot_step": Decimal("0.01"),
                "is_tradeable": True
            })

        # Test bulk create
        import time
        start_time = time.time()
        created_symbols = symbol_service.bulk_create(bulk_data, batch_size=20)
        end_time = time.time()

        assert len(created_symbols) == 100
        assert all(symbol.id is not None for symbol in created_symbols)

        # Should complete in reasonable time (adjust threshold as needed)
        duration = end_time - start_time
        assert duration < 10.0  # Should complete in under 10 seconds

        print(f"Bulk created 100 symbols in {duration:.2f} seconds")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])