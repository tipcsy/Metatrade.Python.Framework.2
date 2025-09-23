"""
Database health check utilities for the MetaTrader Python Framework.

This module provides comprehensive database health monitoring capabilities
including performance metrics, connection testing, and system diagnostics.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.core.logging import get_logger
from src.database.database import get_database_manager

logger = get_logger(__name__)


class DatabaseHealthChecker:
    """
    Comprehensive database health monitoring system.

    Provides detailed health checks, performance monitoring,
    and diagnostics for database operations.
    """

    def __init__(self):
        """Initialize the health checker."""
        self._db_manager = get_database_manager()
        self._test_queries = {
            'basic': "SELECT 1 as test",
            'datetime': "SELECT CURRENT_TIMESTAMP as now",
            'count': "SELECT COUNT(*) as total FROM information_schema.tables WHERE table_schema = 'public'",
        }

    def run_basic_health_check(self) -> Dict[str, Any]:
        """
        Run basic database connectivity and response time checks.

        Returns:
            Dict containing health check results
        """
        start_time = time.time()
        health_result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'unknown',
            'checks': {},
            'summary': {},
            'errors': []
        }

        try:
            # Test basic connectivity
            connectivity_result = self._test_connectivity()
            health_result['checks']['connectivity'] = connectivity_result

            # Test query performance
            performance_result = self._test_query_performance()
            health_result['checks']['performance'] = performance_result

            # Test connection pool status
            pool_result = self._check_connection_pool()
            health_result['checks']['connection_pool'] = pool_result

            # Determine overall status
            all_checks_passed = all(
                check.get('status') == 'healthy'
                for check in health_result['checks'].values()
            )

            health_result['status'] = 'healthy' if all_checks_passed else 'degraded'

            # Calculate summary metrics
            total_time = time.time() - start_time
            health_result['summary'] = {
                'total_check_time_ms': round(total_time * 1000, 2),
                'checks_passed': sum(1 for check in health_result['checks'].values() if check.get('status') == 'healthy'),
                'total_checks': len(health_result['checks']),
                'overall_status': health_result['status']
            }

        except Exception as e:
            health_result['status'] = 'unhealthy'
            health_result['errors'].append(f"Health check failed: {str(e)}")
            logger.error(f"Database health check failed: {e}")

        return health_result

    def _test_connectivity(self) -> Dict[str, Any]:
        """Test basic database connectivity."""
        result = {
            'name': 'connectivity',
            'status': 'unknown',
            'response_time_ms': None,
            'details': {},
            'errors': []
        }

        try:
            start_time = time.time()

            with self._db_manager.get_session() as session:
                db_result = session.execute(text(self._test_queries['basic']))
                row = db_result.fetchone()

                if row and row[0] == 1:
                    result['status'] = 'healthy'
                    result['details']['test_result'] = row[0]
                else:
                    result['status'] = 'unhealthy'
                    result['errors'].append('Unexpected test query result')

            result['response_time_ms'] = round((time.time() - start_time) * 1000, 2)

        except SQLAlchemyError as e:
            result['status'] = 'unhealthy'
            result['errors'].append(f"Database error: {str(e)}")
            logger.error(f"Database connectivity test failed: {e}")

        except Exception as e:
            result['status'] = 'unhealthy'
            result['errors'].append(f"Unexpected error: {str(e)}")
            logger.error(f"Connectivity test failed: {e}")

        return result

    def _test_query_performance(self) -> Dict[str, Any]:
        """Test database query performance with multiple test queries."""
        result = {
            'name': 'performance',
            'status': 'unknown',
            'queries': {},
            'summary': {},
            'errors': []
        }

        total_time = 0
        successful_queries = 0

        for query_name, query_sql in self._test_queries.items():
            query_result = self._execute_timed_query(query_name, query_sql)
            result['queries'][query_name] = query_result

            if query_result['status'] == 'success':
                successful_queries += 1
                total_time += query_result['execution_time_ms']
            else:
                result['errors'].extend(query_result.get('errors', []))

        # Calculate performance summary
        if successful_queries > 0:
            avg_response_time = total_time / successful_queries
            result['summary'] = {
                'successful_queries': successful_queries,
                'total_queries': len(self._test_queries),
                'average_response_time_ms': round(avg_response_time, 2),
                'total_execution_time_ms': round(total_time, 2)
            }

            # Determine status based on performance thresholds
            if avg_response_time < 100:  # Less than 100ms average
                result['status'] = 'healthy'
            elif avg_response_time < 500:  # Less than 500ms average
                result['status'] = 'degraded'
            else:
                result['status'] = 'unhealthy'
        else:
            result['status'] = 'unhealthy'
            result['errors'].append('No queries executed successfully')

        return result

    def _execute_timed_query(self, query_name: str, query_sql: str) -> Dict[str, Any]:
        """Execute a query and measure its performance."""
        query_result = {
            'name': query_name,
            'sql': query_sql,
            'status': 'unknown',
            'execution_time_ms': None,
            'row_count': None,
            'errors': []
        }

        try:
            start_time = time.time()

            with self._db_manager.get_session() as session:
                db_result = session.execute(text(query_sql))
                rows = db_result.fetchall()
                query_result['row_count'] = len(rows)

            query_result['execution_time_ms'] = round((time.time() - start_time) * 1000, 2)
            query_result['status'] = 'success'

        except SQLAlchemyError as e:
            query_result['status'] = 'failed'
            query_result['errors'].append(f"SQL error: {str(e)}")

        except Exception as e:
            query_result['status'] = 'failed'
            query_result['errors'].append(f"Unexpected error: {str(e)}")

        return query_result

    def _check_connection_pool(self) -> Dict[str, Any]:
        """Check connection pool health and statistics."""
        result = {
            'name': 'connection_pool',
            'status': 'unknown',
            'pool_info': {},
            'statistics': {},
            'errors': []
        }

        try:
            # Get connection pool statistics from database manager
            stats = self._db_manager.get_connection_stats()

            if 'pool_info' in stats and stats['pool_info']:
                pool_info = stats['pool_info']
                result['pool_info'] = pool_info

                # Analyze pool health
                checked_out = pool_info.get('checked_out', 0)
                size = pool_info.get('size', 0)
                overflow = pool_info.get('overflow', 0)
                invalid = pool_info.get('invalid', 0)

                # Calculate pool utilization
                if size > 0:
                    utilization = (checked_out / size) * 100
                    result['pool_info']['utilization_percent'] = round(utilization, 2)

                # Determine pool status
                if invalid > 0:
                    result['status'] = 'degraded'
                    result['errors'].append(f"Pool has {invalid} invalid connections")
                elif utilization > 90:
                    result['status'] = 'degraded'
                    result['errors'].append(f"High pool utilization: {utilization:.1f}%")
                else:
                    result['status'] = 'healthy'

            result['statistics'] = stats.get('pool_stats', {})

        except Exception as e:
            result['status'] = 'unhealthy'
            result['errors'].append(f"Pool check failed: {str(e)}")
            logger.error(f"Connection pool check failed: {e}")

        return result

    async def run_async_health_check(self) -> Dict[str, Any]:
        """
        Run asynchronous database health checks.

        Returns:
            Dict containing async health check results
        """
        start_time = time.time()
        health_result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'unknown',
            'checks': {},
            'summary': {},
            'errors': []
        }

        try:
            # Test async connectivity
            async_connectivity_result = await self._test_async_connectivity()
            health_result['checks']['async_connectivity'] = async_connectivity_result

            # Test concurrent operations
            concurrency_result = await self._test_concurrent_operations()
            health_result['checks']['concurrency'] = concurrency_result

            # Determine overall status
            all_checks_passed = all(
                check.get('status') == 'healthy'
                for check in health_result['checks'].values()
            )

            health_result['status'] = 'healthy' if all_checks_passed else 'degraded'

            # Calculate summary metrics
            total_time = time.time() - start_time
            health_result['summary'] = {
                'total_check_time_ms': round(total_time * 1000, 2),
                'checks_passed': sum(1 for check in health_result['checks'].values() if check.get('status') == 'healthy'),
                'total_checks': len(health_result['checks']),
                'overall_status': health_result['status']
            }

        except Exception as e:
            health_result['status'] = 'unhealthy'
            health_result['errors'].append(f"Async health check failed: {str(e)}")
            logger.error(f"Async database health check failed: {e}")

        return health_result

    async def _test_async_connectivity(self) -> Dict[str, Any]:
        """Test asynchronous database connectivity."""
        result = {
            'name': 'async_connectivity',
            'status': 'unknown',
            'response_time_ms': None,
            'details': {},
            'errors': []
        }

        try:
            start_time = time.time()

            async with self._db_manager.get_async_session() as session:
                db_result = await session.execute(text(self._test_queries['basic']))
                row = await db_result.fetchone()

                if row and row[0] == 1:
                    result['status'] = 'healthy'
                    result['details']['test_result'] = row[0]
                else:
                    result['status'] = 'unhealthy'
                    result['errors'].append('Unexpected test query result')

            result['response_time_ms'] = round((time.time() - start_time) * 1000, 2)

        except SQLAlchemyError as e:
            result['status'] = 'unhealthy'
            result['errors'].append(f"Async database error: {str(e)}")
            logger.error(f"Async connectivity test failed: {e}")

        except Exception as e:
            result['status'] = 'unhealthy'
            result['errors'].append(f"Unexpected async error: {str(e)}")
            logger.error(f"Async connectivity test failed: {e}")

        return result

    async def _test_concurrent_operations(self, num_concurrent: int = 5) -> Dict[str, Any]:
        """Test concurrent database operations."""
        result = {
            'name': 'concurrency',
            'status': 'unknown',
            'concurrent_operations': num_concurrent,
            'results': [],
            'summary': {},
            'errors': []
        }

        try:
            # Create multiple concurrent query tasks
            tasks = []
            for i in range(num_concurrent):
                task = self._async_query_task(f"task_{i}", self._test_queries['basic'])
                tasks.append(task)

            # Execute all tasks concurrently
            start_time = time.time()
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Process results
            successful_tasks = 0
            failed_tasks = 0

            for i, task_result in enumerate(task_results):
                if isinstance(task_result, Exception):
                    result['results'].append({
                        'task_id': f"task_{i}",
                        'status': 'failed',
                        'error': str(task_result)
                    })
                    failed_tasks += 1
                    result['errors'].append(f"Task {i} failed: {str(task_result)}")
                else:
                    result['results'].append(task_result)
                    if task_result.get('status') == 'success':
                        successful_tasks += 1
                    else:
                        failed_tasks += 1

            # Calculate summary
            result['summary'] = {
                'total_time_ms': round(total_time * 1000, 2),
                'successful_tasks': successful_tasks,
                'failed_tasks': failed_tasks,
                'success_rate_percent': round((successful_tasks / num_concurrent) * 100, 2)
            }

            # Determine status
            if successful_tasks == num_concurrent:
                result['status'] = 'healthy'
            elif successful_tasks > num_concurrent * 0.5:  # More than 50% success
                result['status'] = 'degraded'
            else:
                result['status'] = 'unhealthy'

        except Exception as e:
            result['status'] = 'unhealthy'
            result['errors'].append(f"Concurrency test failed: {str(e)}")
            logger.error(f"Concurrent operations test failed: {e}")

        return result

    async def _async_query_task(self, task_id: str, query_sql: str) -> Dict[str, Any]:
        """Execute a single async query task."""
        task_result = {
            'task_id': task_id,
            'status': 'unknown',
            'execution_time_ms': None,
            'error': None
        }

        try:
            start_time = time.time()

            async with self._db_manager.get_async_session() as session:
                db_result = await session.execute(text(query_sql))
                await db_result.fetchone()

            task_result['execution_time_ms'] = round((time.time() - start_time) * 1000, 2)
            task_result['status'] = 'success'

        except Exception as e:
            task_result['status'] = 'failed'
            task_result['error'] = str(e)

        return task_result

    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report combining all checks.

        Returns:
            Dict containing detailed health report
        """
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'report_type': 'comprehensive',
            'sync_health': self.run_basic_health_check(),
            'database_info': self._get_database_info(),
            'recommendations': []
        }

        # Add recommendations based on health check results
        sync_health = report['sync_health']

        # Check response times
        if 'performance' in sync_health['checks']:
            perf_check = sync_health['checks']['performance']
            if 'summary' in perf_check:
                avg_time = perf_check['summary'].get('average_response_time_ms', 0)
                if avg_time > 200:
                    report['recommendations'].append(
                        f"High average response time ({avg_time:.1f}ms). Consider optimizing queries or checking server resources."
                    )

        # Check connection pool utilization
        if 'connection_pool' in sync_health['checks']:
            pool_check = sync_health['checks']['connection_pool']
            if 'pool_info' in pool_check:
                utilization = pool_check['pool_info'].get('utilization_percent', 0)
                if utilization > 80:
                    report['recommendations'].append(
                        f"High connection pool utilization ({utilization:.1f}%). Consider increasing pool size."
                    )

        # Check for errors
        total_errors = len(sync_health.get('errors', []))
        for check in sync_health.get('checks', {}).values():
            total_errors += len(check.get('errors', []))

        if total_errors > 0:
            report['recommendations'].append(
                f"Found {total_errors} error(s). Review error details and consider investigating root causes."
            )

        # Overall status recommendation
        if sync_health['status'] != 'healthy':
            report['recommendations'].append(
                "Database health is not optimal. Review individual check results and address any issues found."
            )

        return report

    def _get_database_info(self) -> Dict[str, Any]:
        """Get basic database information."""
        info = {
            'engine_info': {},
            'connection_stats': {},
            'error': None
        }

        try:
            stats = self._db_manager.get_connection_stats()
            info['engine_info'] = stats.get('engine_info', {})
            info['connection_stats'] = stats.get('pool_stats', {})

        except Exception as e:
            info['error'] = str(e)
            logger.error(f"Failed to get database info: {e}")

        return info


# Convenience functions
def run_health_check() -> Dict[str, Any]:
    """Run a basic database health check."""
    checker = DatabaseHealthChecker()
    return checker.run_basic_health_check()


async def run_async_health_check() -> Dict[str, Any]:
    """Run an async database health check."""
    checker = DatabaseHealthChecker()
    return await checker.run_async_health_check()


def get_health_report() -> Dict[str, Any]:
    """Get a comprehensive database health report."""
    checker = DatabaseHealthChecker()
    return checker.get_comprehensive_health_report()