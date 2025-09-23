"""
Unit tests for database services.

This module contains unit tests for all service classes including
business logic validation, caching behavior, and error handling.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict

from src.core.exceptions import ValidationError, NotFoundError, BusinessLogicError
from src.database.services.symbols import SymbolGroupService, SymbolService, SymbolSessionService
from src.database.services.accounts import UserService, AccountService, TransactionService
from src.database.models.symbols import SymbolGroup, Symbol, SymbolSession
from src.database.models.accounts import User, Account, Transaction


class TestSymbolGroupService:
    """Test cases for SymbolGroupService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        with patch('src.database.services.base.get_database_manager') as mock:
            yield mock.return_value

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager."""
        with patch('src.database.services.base.get_cache_manager') as mock:
            yield mock.return_value

    @pytest.fixture
    def service(self, mock_db_manager, mock_cache_manager):
        """Create SymbolGroupService instance with mocked dependencies."""
        return SymbolGroupService()

    def test_create_schema_property(self, service):
        """Test create_schema property returns correct schema."""
        from src.database.schemas.symbols import SymbolGroupCreateSchema
        assert service.create_schema == SymbolGroupCreateSchema

    def test_validate_create_data_success(self, service):
        """Test successful data validation for creation."""
        data = {
            "name": "Major Forex",
            "description": "Major currency pairs",
            "group_type": "FOREX",
            "display_order": 1,
            "is_active": True
        }

        validated = service._validate_create_data(data)
        assert validated["name"] == "Major Forex"
        assert validated["group_type"] == "FOREX"

    def test_validate_create_data_invalid(self, service):
        """Test validation failure for invalid data."""
        data = {
            "name": "",  # Invalid: empty name
            "group_type": "INVALID",  # Invalid: not in allowed types
        }

        with pytest.raises(ValidationError):
            service._validate_create_data(data)

    def test_business_rules_market_hours(self, service):
        """Test business rule validation for market hours."""
        group = SymbolGroup(
            name="Test Group",
            group_type="FOREX",
            market_open_hour=9,
            market_close_hour=9  # Same as open - should fail
        )

        with pytest.raises(BusinessLogicError, match="Market open and close hours cannot be the same"):
            service._apply_business_rules(group, 'create')

    @patch.object(SymbolGroupService, 'get_by_name_and_type')
    def test_business_rules_unique_name(self, mock_get, service):
        """Test business rule for unique name per type."""
        # Mock existing group
        existing_group = SymbolGroup(id="existing-id", name="Test Group", group_type="FOREX")
        mock_get.return_value = existing_group

        # New group with same name and type
        new_group = SymbolGroup(id="new-id", name="Test Group", group_type="FOREX")

        with pytest.raises(BusinessLogicError, match="already exists"):
            service._apply_business_rules(new_group, 'create')

    def test_get_by_type(self, service, mock_db_manager):
        """Test getting groups by type."""
        # Mock session and query
        mock_session = Mock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Mock query result
        mock_groups = [
            Mock(id="1", group_type="FOREX", display_order=1),
            Mock(id="2", group_type="FOREX", display_order=2)
        ]
        service.get_all = Mock(return_value=mock_groups)

        result = service.get_by_type("FOREX")

        assert len(result) == 2
        service.get_all.assert_called_once_with(
            filters={'group_type': 'FOREX', 'is_active': True},
            order_by='display_order'
        )

    def test_reorder_groups_success(self, service, mock_db_manager):
        """Test successful group reordering."""
        # Mock session
        mock_session = Mock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        group_orders = {"group1": 1, "group2": 2, "group3": 3}

        result = service.reorder_groups(group_orders)

        assert result is True
        assert mock_session.query.call_count == 3
        mock_session.commit.assert_called_once()


class TestSymbolService:
    """Test cases for SymbolService."""

    @pytest.fixture
    def service(self):
        """Create SymbolService instance."""
        with patch('src.database.services.base.get_database_manager'), \
             patch('src.database.services.base.get_cache_manager'):
            return SymbolService()

    def test_business_rules_lot_size_validation(self, service):
        """Test business rule validation for lot sizes."""
        symbol = Symbol(
            symbol="EURUSD",
            min_lot=Decimal("0.1"),
            max_lot=Decimal("0.05"),  # Invalid: max < min
            lot_step=Decimal("0.01")
        )

        with pytest.raises(BusinessLogicError, match="Minimum lot size must be less than maximum"):
            service._apply_business_rules(symbol, 'create')

    def test_business_rules_lot_step_validation(self, service):
        """Test business rule validation for lot step."""
        symbol = Symbol(
            symbol="EURUSD",
            min_lot=Decimal("0.01"),
            max_lot=Decimal("1.0"),
            lot_step=Decimal("0.1")  # Invalid: step > min_lot
        )

        with pytest.raises(BusinessLogicError, match="Lot step cannot be greater than minimum lot size"):
            service._apply_business_rules(symbol, 'create')

    def test_business_rules_margin_validation(self, service):
        """Test business rule validation for margin requirements."""
        symbol = Symbol(
            symbol="EURUSD",
            margin_initial=Decimal("1.0"),
            margin_maintenance=Decimal("2.0")  # Invalid: maintenance > initial
        )

        with pytest.raises(BusinessLogicError, match="Maintenance margin cannot exceed initial margin"):
            service._apply_business_rules(symbol, 'create')

    def test_business_rules_currency_validation(self, service):
        """Test business rule validation for currency pair."""
        symbol = Symbol(
            symbol="USDNEW",
            base_currency="USD",
            quote_currency="USD"  # Invalid: same currencies
        )

        with pytest.raises(BusinessLogicError, match="Base and quote currencies must be different"):
            service._apply_business_rules(symbol, 'create')

    def test_update_quote_validation(self, service):
        """Test quote update validation."""
        with pytest.raises(ValidationError, match="Bid and ask prices must be positive"):
            service.update_quote("symbol-id", Decimal("-1"), Decimal("1.1236"))

        with pytest.raises(ValidationError, match="Ask price cannot be less than bid price"):
            service.update_quote("symbol-id", Decimal("1.1236"), Decimal("1.1234"))

    @patch.object(SymbolService, 'get_by_id')
    def test_validate_lot_size_success(self, mock_get, service):
        """Test successful lot size validation."""
        # Mock symbol with validate_lot_size method
        mock_symbol = Mock()
        mock_symbol.validate_lot_size.return_value = True
        mock_get.return_value = mock_symbol

        result = service.validate_lot_size("symbol-id", Decimal("0.1"))

        assert result is True
        mock_symbol.validate_lot_size.assert_called_once_with(Decimal("0.1"))

    @patch.object(SymbolService, 'get_by_id')
    def test_validate_lot_size_failure(self, mock_get, service):
        """Test lot size validation failure."""
        # Mock symbol with validate_lot_size method
        mock_symbol = Mock()
        mock_symbol.validate_lot_size.return_value = False
        mock_symbol.symbol = "EURUSD"
        mock_symbol.min_lot = Decimal("0.01")
        mock_symbol.max_lot = Decimal("100")
        mock_symbol.lot_step = Decimal("0.01")
        mock_get.return_value = mock_symbol

        with pytest.raises(ValidationError, match="Invalid lot size"):
            service.validate_lot_size("symbol-id", Decimal("0.001"))

    def test_search_symbols(self, service):
        """Test symbol search functionality."""
        # Mock the database session and query
        with patch.object(service, '_db_manager') as mock_db_manager:
            mock_session = Mock()
            mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

            # Mock query chain
            mock_query = Mock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [Mock(symbol="EURUSD")]

            result = service.search_symbols("EUR", limit=10)

            assert len(result) == 1
            mock_session.query.assert_called_once()


class TestUserService:
    """Test cases for UserService."""

    @pytest.fixture
    def service(self):
        """Create UserService instance."""
        with patch('src.database.services.base.get_database_manager'), \
             patch('src.database.services.base.get_cache_manager'):
            return UserService()

    def test_hash_password(self, service):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = service._hash_password(password)

        assert hashed != password
        assert ':' in hashed  # Should contain salt separator

        # Verify password
        assert service._verify_password(password, hashed)
        assert not service._verify_password("wrongpassword", hashed)

    def test_create_with_password_hashing(self, service):
        """Test user creation with password hashing."""
        data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPassword123!",
            "role": "USER"
        }

        # Mock the parent create method
        with patch('src.database.services.base.BaseService.create') as mock_create:
            mock_user = Mock()
            mock_user.id = "user-id"
            mock_create.return_value = mock_user

            service.create(data)

            # Verify password was hashed and removed from data
            call_args = mock_create.call_args[0][0]
            assert 'password' not in call_args
            assert 'password_hash' in call_args
            assert call_args['password_hash'] != "TestPassword123!"

    def test_authenticate_success(self, service):
        """Test successful authentication."""
        # Mock password hash
        password_hash = service._hash_password("TestPassword123!")

        # Mock user
        mock_user = Mock()
        mock_user.is_active = True
        mock_user.password_hash = password_hash

        with patch.object(service, 'get_by_username', return_value=mock_user), \
             patch.object(service, '_update_login_stats') as mock_update:

            result = service.authenticate("testuser", "TestPassword123!")

            assert result == mock_user
            mock_update.assert_called_once_with(mock_user.id)

    def test_authenticate_inactive_user(self, service):
        """Test authentication with inactive user."""
        mock_user = Mock()
        mock_user.is_active = False

        with patch.object(service, 'get_by_username', return_value=mock_user):
            result = service.authenticate("testuser", "TestPassword123!")
            assert result is None

    def test_authenticate_wrong_password(self, service):
        """Test authentication with wrong password."""
        password_hash = service._hash_password("CorrectPassword123!")

        mock_user = Mock()
        mock_user.is_active = True
        mock_user.password_hash = password_hash

        with patch.object(service, 'get_by_username', return_value=mock_user):
            result = service.authenticate("testuser", "WrongPassword123!")
            assert result is None

    def test_change_password_success(self, service):
        """Test successful password change."""
        old_password = "OldPassword123!"
        new_password = "NewPassword123!"
        user_id = "user-id"

        # Mock user with current password
        mock_user = Mock()
        mock_user.password_hash = service._hash_password(old_password)

        with patch.object(service, 'get_by_id', return_value=mock_user), \
             patch.object(service, '_db_manager') as mock_db_manager:

            mock_session = Mock()
            mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

            result = service.change_password(user_id, old_password, new_password)

            assert result is True
            mock_session.query.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_change_password_wrong_current(self, service):
        """Test password change with wrong current password."""
        old_password = "OldPassword123!"
        wrong_password = "WrongPassword123!"
        new_password = "NewPassword123!"
        user_id = "user-id"

        # Mock user with current password
        mock_user = Mock()
        mock_user.password_hash = service._hash_password(old_password)

        with patch.object(service, 'get_by_id', return_value=mock_user):
            from src.core.exceptions import AuthenticationError
            with pytest.raises(AuthenticationError, match="Current password is invalid"):
                service.change_password(user_id, wrong_password, new_password)


class TestAccountService:
    """Test cases for AccountService."""

    @pytest.fixture
    def service(self):
        """Create AccountService instance."""
        with patch('src.database.services.base.get_database_manager'), \
             patch('src.database.services.base.get_cache_manager'):
            return AccountService()

    def test_business_rules_negative_balance(self, service):
        """Test business rule for negative balance."""
        account = Account(
            account_number="ACC123",
            balance=Decimal("-100")  # Invalid: negative balance
        )

        with pytest.raises(BusinessLogicError, match="Account balance cannot be negative"):
            service._apply_business_rules(account, 'create')

    def test_business_rules_leverage_limits(self, service):
        """Test business rule for leverage limits."""
        account = Account(
            account_number="ACC123",
            leverage=1500  # Invalid: exceeds maximum
        )

        with pytest.raises(BusinessLogicError, match="Leverage must be between 1:1 and 1000:1"):
            service._apply_business_rules(account, 'create')

    def test_update_balance_validation(self, service):
        """Test balance update validation."""
        with pytest.raises(ValidationError, match="Balance values cannot be negative"):
            service.update_balance(
                "account-id",
                Decimal("-100"),  # Invalid: negative balance
                Decimal("100"),
                Decimal("50"),
                Decimal("50")
            )

    def test_add_deposit_success(self, service):
        """Test successful deposit addition."""
        account_id = "account-id"
        amount = Decimal("1000")

        # Mock account
        mock_account = Mock()
        mock_account.account_number = "ACC123"
        mock_account.balance = Decimal("5000")
        mock_account.add_deposit = Mock()

        with patch.object(service, '_db_manager') as mock_db_manager, \
             patch('src.database.services.accounts.TransactionService') as mock_tx_service:

            mock_session = Mock()
            mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = mock_account

            # Mock transaction service
            mock_tx_instance = Mock()
            mock_tx_service.return_value = mock_tx_instance

            result = service.add_deposit(account_id, amount, "Test deposit")

            assert result is True
            mock_account.add_deposit.assert_called_once_with(amount)
            mock_tx_instance.create.assert_called_once()

    def test_add_deposit_invalid_amount(self, service):
        """Test deposit with invalid amount."""
        with pytest.raises(ValidationError, match="Deposit amount must be positive"):
            service.add_deposit("account-id", Decimal("-100"))

    def test_check_margin_requirements(self, service):
        """Test margin requirements check."""
        account_id = "account-id"
        required_margin = Decimal("1000")

        # Mock account
        mock_account = Mock()
        mock_account.can_open_position.return_value = True

        with patch.object(service, 'get_by_id', return_value=mock_account):
            result = service.check_margin_requirements(account_id, required_margin)

            assert result is True
            mock_account.can_open_position.assert_called_once_with(required_margin)


class TestCachingBehavior:
    """Test caching behavior in services."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        with patch('src.database.services.base.get_database_manager'), \
             patch('src.database.services.base.get_cache_manager'):
            service = SymbolService()

            cache_key = service._get_cache_key("test-id")
            assert cache_key == "test-id"

    def test_cached_service_functionality(self):
        """Test CachedService specific functionality."""
        with patch('src.database.services.base.get_database_manager'), \
             patch('src.database.services.base.get_cache_manager') as mock_cache:

            service = SymbolService()  # Inherits from CachedService

            # Test cache invalidation
            mock_cache.return_value.invalidate_pattern.return_value = 5
            result = service.invalidate_all_cache()

            assert result == 5
            mock_cache.return_value.invalidate_pattern.assert_called_once_with(service.cache_type)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])