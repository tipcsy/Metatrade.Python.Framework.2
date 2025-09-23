"""
System configuration and audit models for the MetaTrader Python Framework.

This module defines database models for system configuration, audit trails,
monitoring, and operational data management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    DECIMAL,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import BaseModel
from src.database.models.mixins import ConfigurationMixin


class SystemConfiguration(BaseModel, ConfigurationMixin):
    """
    Model for system-wide configuration settings.

    Stores application configuration, feature flags,
    and system parameters with version control.
    """

    __tablename__ = "system_configurations"

    # Configuration classification
    config_category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Configuration category (TRADING, RISK, NOTIFICATION, etc.)"
    )

    config_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Configuration key/identifier"
    )

    # Value information
    config_value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Configuration value (can be JSON for complex values)"
    )

    value_type: Mapped[str] = mapped_column(
        String(20),
        default="STRING",
        nullable=False,
        doc="Value type (STRING, INTEGER, FLOAT, BOOLEAN, JSON)"
    )

    default_value: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Default value"
    )

    # Validation and constraints
    validation_rules: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Validation rules as JSON"
    )

    # Access control
    is_sensitive: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether this configuration contains sensitive data"
    )

    requires_restart: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether changing this config requires system restart"
    )

    # Environment specific
    environment: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        doc="Target environment (DEVELOPMENT, TESTING, PRODUCTION)"
    )

    __table_args__ = (
        UniqueConstraint('config_category', 'config_key', 'environment', name='uq_system_config'),
        Index('ix_system_config_category_active', 'config_category', 'is_active'),
    )

    def get_typed_value(self):
        """Get configuration value in its proper type."""
        if self.value_type == "INTEGER":
            return int(self.config_value)
        elif self.value_type == "FLOAT":
            return float(self.config_value)
        elif self.value_type == "BOOLEAN":
            return self.config_value.lower() in ('true', '1', 'yes', 'on')
        elif self.value_type == "JSON":
            import json
            return json.loads(self.config_value)
        else:
            return self.config_value

    def set_typed_value(self, value) -> None:
        """Set configuration value from its proper type."""
        if self.value_type == "JSON":
            import json
            self.config_value = json.dumps(value)
        else:
            self.config_value = str(value)

    def __repr__(self) -> str:
        return f"<SystemConfiguration(category='{self.config_category}', key='{self.config_key}')>"


class AuditLog(BaseModel):
    """
    Model for audit trail logging.

    Records all system activities, user actions, and data changes
    for compliance, security, and debugging purposes.
    """

    __tablename__ = "audit_logs"

    # Event classification
    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Event type (LOGIN, LOGOUT, TRADE, ORDER, CONFIG_CHANGE, etc.)"
    )

    event_category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Event category (AUTHENTICATION, TRADING, ADMINISTRATION, etc.)"
    )

    severity: Mapped[str] = mapped_column(
        String(20),
        default="INFO",
        nullable=False,
        index=True,
        doc="Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # User and session information
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        doc="User ID who performed the action"
    )

    session_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="Session ID"
    )

    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # IPv6 length
        nullable=True,
        index=True,
        doc="Client IP address"
    )

    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Client user agent"
    )

    # Event details
    event_description: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        doc="Human-readable event description"
    )

    event_data: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Additional event data as JSON"
    )

    # Resource information
    resource_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        doc="Type of resource affected (ACCOUNT, ORDER, POSITION, etc.)"
    )

    resource_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="ID of the affected resource"
    )

    # Change tracking
    old_values: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Previous values (for update operations) as JSON"
    )

    new_values: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="New values (for create/update operations) as JSON"
    )

    # Result and status
    operation_result: Mapped[str] = mapped_column(
        String(20),
        default="SUCCESS",
        nullable=False,
        index=True,
        doc="Operation result (SUCCESS, FAILURE, PARTIAL)"
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
        doc="Error message if operation failed"
    )

    # Performance tracking
    execution_time_ms: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Operation execution time in milliseconds"
    )

    __table_args__ = (
        Index('ix_audit_log_type_category', 'event_type', 'event_category'),
        Index('ix_audit_log_user_time', 'user_id', 'created_at'),
        Index('ix_audit_log_resource', 'resource_type', 'resource_id'),
        Index('ix_audit_log_severity_time', 'severity', 'created_at'),
        Index('ix_audit_log_result', 'operation_result'),
    )

    @classmethod
    def log_event(
        cls,
        event_type: str,
        event_category: str,
        description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        severity: str = "INFO",
        event_data: Optional[dict] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        old_values: Optional[dict] = None,
        new_values: Optional[dict] = None,
        operation_result: str = "SUCCESS",
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None
    ) -> 'AuditLog':
        """
        Create a new audit log entry.

        Args:
            event_type: Type of event
            event_category: Event category
            description: Human-readable description
            user_id: User performing the action
            session_id: Session ID
            ip_address: Client IP address
            severity: Event severity
            event_data: Additional event data
            resource_type: Type of affected resource
            resource_id: ID of affected resource
            old_values: Previous values (for updates)
            new_values: New values (for creates/updates)
            operation_result: Operation result
            error_message: Error message if failed
            execution_time_ms: Execution time

        Returns:
            Created audit log entry
        """
        import json

        log_entry = cls(
            event_type=event_type,
            event_category=event_category,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            event_description=description,
            event_data=json.dumps(event_data) if event_data else None,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=json.dumps(old_values) if old_values else None,
            new_values=json.dumps(new_values) if new_values else None,
            operation_result=operation_result,
            error_message=error_message,
            execution_time_ms=execution_time_ms
        )

        return log_entry

    def __repr__(self) -> str:
        return f"<AuditLog(type='{self.event_type}', result='{self.operation_result}')>"


class SystemMonitoring(BaseModel):
    """
    Model for system monitoring and health metrics.

    Tracks system performance, resource usage, and operational metrics
    for monitoring and alerting purposes.
    """

    __tablename__ = "system_monitoring"

    # Monitoring scope
    component: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="System component (DATABASE, API, TRADING_ENGINE, etc.)"
    )

    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Metric name (CPU_USAGE, MEMORY_USAGE, RESPONSE_TIME, etc.)"
    )

    # Metric value and metadata
    metric_value: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Metric value"
    )

    metric_unit: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Metric unit (PERCENT, MILLISECONDS, BYTES, COUNT, etc.)"
    )

    # Timestamp for the metric
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Metric timestamp"
    )

    # Additional context
    host_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="Host/server name where metric was collected"
    )

    process_id: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Process ID"
    )

    # Metric metadata
    tags: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Additional tags as JSON"
    )

    # Alert thresholds
    warning_threshold: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Warning threshold value"
    )

    critical_threshold: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Critical threshold value"
    )

    __table_args__ = (
        Index('ix_monitoring_component_metric', 'component', 'metric_name'),
        Index('ix_monitoring_timestamp', 'timestamp'),
        Index('ix_monitoring_host_time', 'host_name', 'timestamp'),
    )

    @property
    def is_warning(self) -> bool:
        """Check if metric value exceeds warning threshold."""
        return (self.warning_threshold is not None and
                self.metric_value >= self.warning_threshold)

    @property
    def is_critical(self) -> bool:
        """Check if metric value exceeds critical threshold."""
        return (self.critical_threshold is not None and
                self.metric_value >= self.critical_threshold)

    def __repr__(self) -> str:
        return f"<SystemMonitoring(component='{self.component}', metric='{self.metric_name}', value={self.metric_value})>"


class FeatureFlag(BaseModel):
    """
    Model for feature flags and toggles.

    Enables/disables application features dynamically without
    code deployment for controlled rollouts and A/B testing.
    """

    __tablename__ = "feature_flags"

    # Feature identification
    feature_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        doc="Unique feature name/identifier"
    )

    feature_description: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Feature description"
    )

    # Feature status
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether the feature is enabled"
    )

    # Environment and targeting
    environment: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        doc="Target environment (null for all environments)"
    )

    user_percentage: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=5, scale=2),
        nullable=True,
        doc="Percentage of users who should see this feature (0-100)"
    )

    # Feature lifecycle
    start_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Feature activation start date"
    )

    end_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Feature activation end date"
    )

    # Additional targeting rules
    targeting_rules: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Targeting rules as JSON (user segments, conditions, etc.)"
    )

    # Metadata
    created_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="User who created the feature flag"
    )

    __table_args__ = (
        Index('ix_feature_flag_env_enabled', 'environment', 'is_enabled'),
    )

    def is_active_for_environment(self, env: str) -> bool:
        """
        Check if feature is active for given environment.

        Args:
            env: Environment to check

        Returns:
            True if feature is active for the environment
        """
        if not self.is_enabled:
            return False

        # Check environment targeting
        if self.environment is not None and self.environment != env:
            return False

        # Check date range
        now = datetime.now(timezone.utc)
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False

        return True

    def is_active_for_user(self, user_id: str, env: str) -> bool:
        """
        Check if feature is active for given user and environment.

        Args:
            user_id: User ID to check
            env: Environment to check

        Returns:
            True if feature is active for the user
        """
        if not self.is_active_for_environment(env):
            return False

        # Check user percentage rollout
        if self.user_percentage is not None:
            # Simple hash-based user bucketing
            import hashlib
            user_hash = int(hashlib.md5(f"{self.feature_name}:{user_id}".encode()).hexdigest()[:8], 16)
            user_bucket = user_hash % 100
            if user_bucket >= float(self.user_percentage):
                return False

        # Additional targeting rules could be implemented here
        # based on the targeting_rules JSON field

        return True

    def __repr__(self) -> str:
        return f"<FeatureFlag(name='{self.feature_name}', enabled={self.is_enabled})>"


class SystemAlert(BaseModel):
    """
    Model for system alerts and notifications.

    Manages system alerts, their status, and resolution tracking
    for operational monitoring and incident management.
    """

    __tablename__ = "system_alerts"

    # Alert classification
    alert_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Alert type (PERFORMANCE, ERROR, SECURITY, MAINTENANCE, etc.)"
    )

    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        doc="Alert severity (LOW, MEDIUM, HIGH, CRITICAL)"
    )

    # Alert content
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        doc="Alert title/summary"
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Detailed alert message"
    )

    # Source information
    source_component: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Component that generated the alert"
    )

    source_host: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Host/server where alert originated"
    )

    # Alert status
    status: Mapped[str] = mapped_column(
        String(20),
        default="OPEN",
        nullable=False,
        index=True,
        doc="Alert status (OPEN, ACKNOWLEDGED, RESOLVED, CLOSED)"
    )

    # Alert timing
    first_occurred: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="When the alert first occurred"
    )

    last_occurred: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="When the alert last occurred"
    )

    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the alert was acknowledged"
    )

    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the alert was resolved"
    )

    # Alert count
    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        doc="Number of times this alert has occurred"
    )

    # Assignment and resolution
    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="User assigned to handle this alert"
    )

    resolution_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Notes about alert resolution"
    )

    # Notification tracking
    notifications_sent: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of notifications sent for this alert"
    )

    last_notification: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the last notification was sent"
    )

    __table_args__ = (
        Index('ix_alert_type_severity', 'alert_type', 'severity'),
        Index('ix_alert_status_time', 'status', 'first_occurred'),
        Index('ix_alert_component_status', 'source_component', 'status'),
    )

    def acknowledge(self, user_id: str) -> None:
        """
        Acknowledge the alert.

        Args:
            user_id: User acknowledging the alert
        """
        if self.status == "OPEN":
            self.status = "ACKNOWLEDGED"
            self.acknowledged_at = datetime.now(timezone.utc)
            self.assigned_to = user_id

    def resolve(self, user_id: str, resolution_notes: Optional[str] = None) -> None:
        """
        Resolve the alert.

        Args:
            user_id: User resolving the alert
            resolution_notes: Notes about the resolution
        """
        self.status = "RESOLVED"
        self.resolved_at = datetime.now(timezone.utc)
        self.assigned_to = user_id
        if resolution_notes:
            self.resolution_notes = resolution_notes

    def update_occurrence(self) -> None:
        """Update alert occurrence count and timestamp."""
        self.occurrence_count += 1
        self.last_occurred = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"<SystemAlert(type='{self.alert_type}', severity='{self.severity}', status='{self.status}')>"