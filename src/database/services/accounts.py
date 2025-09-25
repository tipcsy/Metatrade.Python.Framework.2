"""
Account and user management services for the MetaTrader Python Framework.

This module provides business logic for user and account management including
authentication, authorization, and account operations.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Type, Union

from sqlalchemy import and_, or_, func

from src.core.exceptions import (
    ValidationError,
    NotFoundError,
    BusinessLogicError,
    SecurityError,
)
from src.core.logging import get_logger
from src.database.models.accounts import User, Account, Transaction, AccountSettings
from src.database.schemas.accounts import (
    UserCreateSchema,
    UserUpdateSchema,
    UserPasswordChangeSchema,
    UserResponseSchema,
    AccountCreateSchema,
    AccountUpdateSchema,
    AccountResponseSchema,
    TransactionCreateSchema,
    TransactionResponseSchema,
    AccountSettingsCreateSchema,
    AccountSettingsUpdateSchema,
    AccountSettingsResponseSchema,
)
from .base import CachedService

logger = get_logger(__name__)


class UserService(CachedService[User, UserCreateSchema, UserUpdateSchema, UserResponseSchema]):
    """Service for user management."""

    def __init__(self):
        super().__init__(User, cache_ttl=1800)  # Cache for 30 minutes

    @property
    def create_schema(self) -> Type[UserCreateSchema]:
        return UserCreateSchema

    @property
    def update_schema(self) -> Type[UserUpdateSchema]:
        return UserUpdateSchema

    @property
    def response_schema(self) -> Type[UserResponseSchema]:
        return UserResponseSchema

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hash_part = password_hash.split(':')
            expected_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return hash_part == expected_hash
        except ValueError:
            return False

    def _apply_business_rules(self, instance: User, operation: str) -> None:
        """Apply business rules for users."""
        if operation in ('create', 'update'):
            # Validate username uniqueness
            if operation == 'create':
                existing_user = self.get_by_username(instance.username)
                if existing_user:
                    raise BusinessLogicError(f"Username '{instance.username}' already exists")

                existing_email = self.get_by_email(instance.email)
                if existing_email:
                    raise BusinessLogicError(f"Email '{instance.email}' already exists")

            # Validate role assignments
            if instance.role == 'ADMIN':
                # Only existing admins can create other admins
                # This would require current user context in a real application
                pass

    def create(self, data: Dict[str, Any], commit: bool = True) -> User:
        """
        Create new user with password hashing.

        Args:
            data: User creation data
            commit: Whether to commit transaction

        Returns:
            Created user instance
        """
        # Hash password before creation
        if 'password' in data:
            data['password_hash'] = self._hash_password(data.pop('password'))

        return super().create(data, commit)

    def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username

        Returns:
            User or None
        """
        cache_key = f"username:{username}"

        # Check cache first
        if self.cache_enabled:
            cached_data = self._cache_manager.get(self.cache_type, cache_key)
            if cached_data:
                return User(**cached_data)

        with self._db_manager.get_session() as session:
            user = session.query(User).filter(
                and_(
                    User.username == username,
                    User.deleted_at.is_(None)
                )
            ).first()

            if user and self.cache_enabled:
                self._cache_manager.set(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(user),
                    self.cache_ttl
                )

            return user

    def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.

        Args:
            email: Email address

        Returns:
            User or None
        """
        cache_key = f"email:{email.lower()}"

        # Check cache first
        if self.cache_enabled:
            cached_data = self._cache_manager.get(self.cache_type, cache_key)
            if cached_data:
                return User(**cached_data)

        with self._db_manager.get_session() as session:
            user = session.query(User).filter(
                and_(
                    User.email == email.lower(),
                    User.deleted_at.is_(None)
                )
            ).first()

            if user and self.cache_enabled:
                self._cache_manager.set(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(user),
                    self.cache_ttl
                )

            return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.

        Args:
            username: Username or email
            password: Password

        Returns:
            User if authentication successful, None otherwise
        """
        # Try to find user by username or email
        user = self.get_by_username(username)
        if not user:
            user = self.get_by_email(username)

        if not user:
            logger.warning(f"Authentication failed: user '{username}' not found")
            return None

        if not user.is_active:
            logger.warning(f"Authentication failed: user '{username}' is inactive")
            return None

        if not self._verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed: invalid password for user '{username}'")
            return None

        # Update login statistics
        try:
            self._update_login_stats(user.id)
        except Exception as e:
            logger.error(f"Failed to update login stats for user {user.id}: {e}")

        logger.info(f"User '{username}' authenticated successfully")
        return user

    def _update_login_stats(self, user_id: str) -> None:
        """Update user login statistics."""
        with self._db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.now(timezone.utc)
                user.login_count += 1
                session.commit()

                # Update cache
                if self.cache_enabled:
                    cache_key = self._get_cache_key(user_id)
                    self._cache_manager.set(
                        self.cache_type,
                        cache_key,
                        self._model_to_dict(user),
                        self.cache_ttl
                    )

    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            True if successful

        Raises:
            NotFoundError: If user not found
            SecurityError: If current password is invalid
        """
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError(f"User with ID {user_id} not found")

        if not self._verify_password(current_password, user.password_hash):
            raise SecurityError("Current password is invalid")

        try:
            new_password_hash = self._hash_password(new_password)

            with self._db_manager.get_session() as session:
                session.query(User).filter(User.id == user_id).update({
                    'password_hash': new_password_hash
                })
                session.commit()

            # Invalidate cache
            if self.cache_enabled:
                cache_key = self._get_cache_key(user_id)
                self._cache_manager.delete(self.cache_type, cache_key)

            logger.info(f"Password changed for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error changing password for user {user_id}: {e}")
            return False

    def get_by_role(self, role: str, active_only: bool = True) -> List[User]:
        """
        Get users by role.

        Args:
            role: User role
            active_only: Whether to return only active users

        Returns:
            List of users
        """
        filters = {'role': role.upper()}
        if active_only:
            filters['is_active'] = True

        return self.get_all(filters=filters, order_by='username')

    def activate_user(self, user_id: str) -> bool:
        """
        Activate user account.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        return self.update(user_id, {'is_active': True}) is not None

    def deactivate_user(self, user_id: str) -> bool:
        """
        Deactivate user account.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        return self.update(user_id, {'is_active': False}) is not None

    def verify_email(self, user_id: str) -> bool:
        """
        Mark user email as verified.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        return self.update(user_id, {'is_verified': True}) is not None


class AccountService(CachedService[Account, AccountCreateSchema, AccountUpdateSchema, AccountResponseSchema]):
    """Service for trading account management."""

    def __init__(self):
        super().__init__(Account, cache_ttl=300)  # Cache for 5 minutes

    @property
    def create_schema(self) -> Type[AccountCreateSchema]:
        return AccountCreateSchema

    @property
    def update_schema(self) -> Type[AccountUpdateSchema]:
        return AccountUpdateSchema

    @property
    def response_schema(self) -> Type[AccountResponseSchema]:
        return AccountResponseSchema

    def _apply_business_rules(self, instance: Account, operation: str) -> None:
        """Apply business rules for accounts."""
        if operation in ('create', 'update'):
            # Validate account number uniqueness
            if operation == 'create':
                existing = self.get_by_account_number(instance.account_number)
                if existing:
                    raise BusinessLogicError(f"Account number '{instance.account_number}' already exists")

            # Validate balance constraints
            if instance.balance < 0:
                raise BusinessLogicError("Account balance cannot be negative")

            if instance.equity < 0:
                raise BusinessLogicError("Account equity cannot be negative")

            # Validate leverage constraints
            if instance.leverage < 1 or instance.leverage > 1000:
                raise BusinessLogicError("Leverage must be between 1:1 and 1000:1")

    def get_by_account_number(self, account_number: str) -> Optional[Account]:
        """
        Get account by account number.

        Args:
            account_number: Account number

        Returns:
            Account or None
        """
        cache_key = f"account_number:{account_number}"

        # Check cache first
        if self.cache_enabled:
            cached_data = self._cache_manager.get(self.cache_type, cache_key)
            if cached_data:
                return Account(**cached_data)

        with self._db_manager.get_session() as session:
            account = session.query(Account).filter(
                and_(
                    Account.account_number == account_number,
                    Account.deleted_at.is_(None)
                )
            ).first()

            if account and self.cache_enabled:
                self._cache_manager.set(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(account),
                    self.cache_ttl
                )

            return account

    def get_by_user(self, user_id: str, active_only: bool = True) -> List[Account]:
        """
        Get accounts by user.

        Args:
            user_id: User ID
            active_only: Whether to return only active accounts

        Returns:
            List of accounts
        """
        filters = {'user_id': user_id}
        if active_only:
            filters['is_active'] = True

        return self.get_all(filters=filters, order_by='account_number')

    def get_by_type(self, account_type: str, active_only: bool = True) -> List[Account]:
        """
        Get accounts by type.

        Args:
            account_type: Account type
            active_only: Whether to return only active accounts

        Returns:
            List of accounts
        """
        filters = {'account_type': account_type.upper()}
        if active_only:
            filters['is_active'] = True

        return self.get_all(filters=filters, order_by='account_number')

    def update_balance(
        self,
        account_id: str,
        balance: Decimal,
        equity: Decimal,
        margin: Decimal,
        free_margin: Decimal
    ) -> bool:
        """
        Update account balance information.

        Args:
            account_id: Account ID
            balance: New balance
            equity: New equity
            margin: Used margin
            free_margin: Free margin

        Returns:
            True if successful

        Raises:
            NotFoundError: If account not found
            ValidationError: If values are invalid
        """
        if balance < 0 or equity < 0 or margin < 0 or free_margin < 0:
            raise ValidationError("Balance values cannot be negative")

        try:
            with self._db_manager.get_session() as session:
                account = session.query(Account).filter(Account.id == account_id).first()
                if not account:
                    raise NotFoundError(f"Account with ID {account_id} not found")

                account.update_balance(balance, equity, margin, free_margin)
                session.commit()

                # Update cache
                if self.cache_enabled:
                    cache_key = self._get_cache_key(account_id)
                    self._cache_manager.set(
                        self.cache_type,
                        cache_key,
                        self._model_to_dict(account),
                        self.cache_ttl
                    )

                logger.debug(f"Updated balance for account {account.account_number}")
                return True

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Error updating balance for account {account_id}: {e}")
            return False

    def add_deposit(self, account_id: str, amount: Decimal, comment: str = None) -> bool:
        """
        Add deposit to account.

        Args:
            account_id: Account ID
            amount: Deposit amount
            comment: Optional comment

        Returns:
            True if successful
        """
        if amount <= 0:
            raise ValidationError("Deposit amount must be positive")

        try:
            with self._db_manager.get_session() as session:
                account = session.query(Account).filter(Account.id == account_id).first()
                if not account:
                    raise NotFoundError(f"Account with ID {account_id} not found")

                balance_before = account.balance
                account.add_deposit(amount)
                balance_after = account.balance

                # Create transaction record
                transaction_service = TransactionService()
                transaction_data = {
                    'account_id': account_id,
                    'transaction_type': 'DEPOSIT',
                    'amount': amount,
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'comment': comment or 'Deposit'
                }
                transaction_service.create(transaction_data)

                session.commit()

                # Update cache
                if self.cache_enabled:
                    cache_key = self._get_cache_key(account_id)
                    self._cache_manager.set(
                        self.cache_type,
                        cache_key,
                        self._model_to_dict(account),
                        self.cache_ttl
                    )

                logger.info(f"Added deposit of {amount} to account {account.account_number}")
                return True

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Error adding deposit to account {account_id}: {e}")
            return False

    def add_withdrawal(self, account_id: str, amount: Decimal, comment: str = None) -> bool:
        """
        Add withdrawal from account.

        Args:
            account_id: Account ID
            amount: Withdrawal amount
            comment: Optional comment

        Returns:
            True if successful

        Raises:
            ValidationError: If insufficient funds
        """
        if amount <= 0:
            raise ValidationError("Withdrawal amount must be positive")

        try:
            with self._db_manager.get_session() as session:
                account = session.query(Account).filter(Account.id == account_id).first()
                if not account:
                    raise NotFoundError(f"Account with ID {account_id} not found")

                balance_before = account.balance
                success = account.add_withdrawal(amount)

                if not success:
                    raise ValidationError("Insufficient funds for withdrawal")

                balance_after = account.balance

                # Create transaction record
                transaction_service = TransactionService()
                transaction_data = {
                    'account_id': account_id,
                    'transaction_type': 'WITHDRAWAL',
                    'amount': -amount,  # Negative for withdrawal
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'comment': comment or 'Withdrawal'
                }
                transaction_service.create(transaction_data)

                session.commit()

                # Update cache
                if self.cache_enabled:
                    cache_key = self._get_cache_key(account_id)
                    self._cache_manager.set(
                        self.cache_type,
                        cache_key,
                        self._model_to_dict(account),
                        self.cache_ttl
                    )

                logger.info(f"Added withdrawal of {amount} from account {account.account_number}")
                return True

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Error adding withdrawal to account {account_id}: {e}")
            return False

    def check_margin_requirements(self, account_id: str, required_margin: Decimal) -> bool:
        """
        Check if account can meet margin requirements.

        Args:
            account_id: Account ID
            required_margin: Required margin amount

        Returns:
            True if requirements can be met

        Raises:
            NotFoundError: If account not found
        """
        account = self.get_by_id(account_id)
        if not account:
            raise NotFoundError(f"Account with ID {account_id} not found")

        return account.can_open_position(required_margin)

    def get_account_summary(self, user_id: str) -> Dict[str, Union[int, Decimal]]:
        """
        Get account summary for user.

        Args:
            user_id: User ID

        Returns:
            Account summary statistics
        """
        accounts = self.get_by_user(user_id, active_only=True)

        summary = {
            'total_accounts': len(accounts),
            'demo_accounts': len([a for a in accounts if a.account_type == 'DEMO']),
            'live_accounts': len([a for a in accounts if a.account_type == 'LIVE']),
            'total_balance': sum(a.balance for a in accounts),
            'total_equity': sum(a.equity for a in accounts),
            'total_margin': sum(a.margin for a in accounts),
            'currencies': list(set(a.currency for a in accounts))
        }

        return summary


class TransactionService(CachedService[Transaction, TransactionCreateSchema, TransactionCreateSchema, TransactionResponseSchema]):
    """Service for transaction management."""

    def __init__(self):
        super().__init__(Transaction, cache_ttl=600)  # Cache for 10 minutes

    @property
    def create_schema(self) -> Type[TransactionCreateSchema]:
        return TransactionCreateSchema

    @property
    def update_schema(self) -> Type[TransactionCreateSchema]:
        return TransactionCreateSchema

    @property
    def response_schema(self) -> Type[TransactionResponseSchema]:
        return TransactionResponseSchema

    def get_by_account(
        self,
        account_id: str,
        transaction_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Transaction]:
        """
        Get transactions by account.

        Args:
            account_id: Account ID
            transaction_type: Filter by transaction type
            limit: Maximum number of results

        Returns:
            List of transactions
        """
        filters = {'account_id': account_id}
        if transaction_type:
            filters['transaction_type'] = transaction_type.upper()

        return self.get_all(
            filters=filters,
            order_by='created_at',
            order_desc=True,
            limit=limit
        )

    def get_account_balance_history(
        self,
        account_id: str,
        days: int = 30
    ) -> List[Dict[str, Union[str, Decimal]]]:
        """
        Get account balance history.

        Args:
            account_id: Account ID
            days: Number of days to look back

        Returns:
            List of balance history entries
        """
        from datetime import timedelta

        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        with self._db_manager.get_session() as session:
            transactions = session.query(Transaction).filter(
                and_(
                    Transaction.account_id == account_id,
                    Transaction.created_at >= start_date
                )
            ).order_by(Transaction.created_at).all()

            history = []
            for transaction in transactions:
                history.append({
                    'date': transaction.created_at.isoformat(),
                    'balance': transaction.balance_after,
                    'transaction_type': transaction.transaction_type,
                    'amount': transaction.amount
                })

            return history


class AccountSettingsService(CachedService[AccountSettings, AccountSettingsCreateSchema, AccountSettingsUpdateSchema, AccountSettingsResponseSchema]):
    """Service for account settings management."""

    def __init__(self):
        super().__init__(AccountSettings, cache_ttl=1800)  # Cache for 30 minutes

    @property
    def create_schema(self) -> Type[AccountSettingsCreateSchema]:
        return AccountSettingsCreateSchema

    @property
    def update_schema(self) -> Type[AccountSettingsUpdateSchema]:
        return AccountSettingsUpdateSchema

    @property
    def response_schema(self) -> Type[AccountSettingsResponseSchema]:
        return AccountSettingsResponseSchema

    def get_by_account(self, account_id: str) -> Optional[AccountSettings]:
        """
        Get settings by account ID.

        Args:
            account_id: Account ID

        Returns:
            Account settings or None
        """
        cache_key = f"account:{account_id}"

        # Check cache first
        if self.cache_enabled:
            cached_data = self._cache_manager.get(self.cache_type, cache_key)
            if cached_data:
                return AccountSettings(**cached_data)

        with self._db_manager.get_session() as session:
            settings = session.query(AccountSettings).filter(
                AccountSettings.account_id == account_id
            ).first()

            if settings and self.cache_enabled:
                self._cache_manager.set(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(settings),
                    self.cache_ttl
                )

            return settings

    def get_or_create_for_account(self, account_id: str) -> AccountSettings:
        """
        Get settings for account or create with defaults.

        Args:
            account_id: Account ID

        Returns:
            Account settings
        """
        settings = self.get_by_account(account_id)
        if not settings:
            # Create default settings
            default_data = {
                'account_id': account_id,
                'default_lot_size': Decimal('0.1'),
                'risk_per_trade': Decimal('2.0'),
                'auto_stop_loss': False,
                'auto_take_profit': False,
                'email_notifications': True,
                'chart_theme': 'dark',
                'default_timeframe': 'H1'
            }
            settings = self.create(default_data)

        return settings