#!/usr/bin/env python3
"""
Test runner for the MetaTrader Python Framework database layer.

This script provides a comprehensive test runner for all database-related
tests including unit tests, integration tests, and performance tests.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pytest


def run_unit_tests(verbose: bool = False, pattern: Optional[str] = None) -> int:
    """
    Run unit tests.

    Args:
        verbose: Enable verbose output
        pattern: Test pattern to match

    Returns:
        Exit code
    """
    print("Running unit tests...")

    args = [
        "tests/unit/",
        "--tb=short",
        "--strict-markers",
        "--strict-config",
    ]

    if verbose:
        args.append("-v")

    if pattern:
        args.extend(["-k", pattern])

    return pytest.main(args)


def run_integration_tests(verbose: bool = False, pattern: Optional[str] = None) -> int:
    """
    Run integration tests.

    Args:
        verbose: Enable verbose output
        pattern: Test pattern to match

    Returns:
        Exit code
    """
    print("Running integration tests...")

    args = [
        "tests/test_database_integration.py",
        "--tb=short",
        "--strict-markers",
        "--strict-config",
    ]

    if verbose:
        args.append("-v")

    if pattern:
        args.extend(["-k", pattern])

    return pytest.main(args)


def run_performance_tests(verbose: bool = False, pattern: Optional[str] = None) -> int:
    """
    Run performance tests.

    Args:
        verbose: Enable verbose output
        pattern: Test pattern to match

    Returns:
        Exit code
    """
    print("Running performance tests...")

    args = [
        "tests/performance/",
        "--tb=short",
        "--strict-markers",
        "--strict-config",
        "-s",  # Disable output capturing for performance results
    ]

    if verbose:
        args.append("-v")

    if pattern:
        args.extend(["-k", pattern])

    return pytest.main(args)


def run_coverage_tests(html: bool = False) -> int:
    """
    Run tests with coverage reporting.

    Args:
        html: Generate HTML coverage report

    Returns:
        Exit code
    """
    print("Running tests with coverage...")

    try:
        import pytest_cov
    except ImportError:
        print("Error: pytest-cov is required for coverage tests")
        print("Install with: pip install pytest-cov")
        return 1

    args = [
        "tests/",
        "--cov=src/database",
        "--cov-report=term-missing",
        "--cov-fail-under=80",  # Require 80% coverage
        "--tb=short",
    ]

    if html:
        args.extend(["--cov-report=html:htmlcov"])
        print("HTML coverage report will be generated in htmlcov/")

    return pytest.main(args)


def run_all_tests(verbose: bool = False, fast: bool = False) -> int:
    """
    Run all test suites.

    Args:
        verbose: Enable verbose output
        fast: Skip slow tests

    Returns:
        Exit code
    """
    print("Running all test suites...")

    # Unit tests (always run)
    exit_code = run_unit_tests(verbose)
    if exit_code != 0:
        print("Unit tests failed!")
        return exit_code

    # Integration tests
    exit_code = run_integration_tests(verbose)
    if exit_code != 0:
        print("Integration tests failed!")
        return exit_code

    # Performance tests (skip if fast mode)
    if not fast:
        exit_code = run_performance_tests(verbose)
        if exit_code != 0:
            print("Performance tests failed!")
            return exit_code
    else:
        print("Skipping performance tests (fast mode)")

    print("All tests passed!")
    return 0


def validate_environment() -> bool:
    """
    Validate test environment setup.

    Returns:
        True if environment is valid
    """
    # Check if we're in the correct directory
    if not Path("src/database").exists():
        print("Error: Must run from project root directory")
        return False

    # Check required dependencies
    try:
        import pytest
        import sqlalchemy
        import pydantic
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        return False

    # Check optional dependencies
    optional_deps = {
        'redis': 'Redis caching tests',
        'psutil': 'Memory usage tests',
        'pytest_asyncio': 'Async tests',
    }

    missing_optional = []
    for dep, purpose in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(f"  - {dep}: {purpose}")

    if missing_optional:
        print("Warning: Optional dependencies missing:")
        for dep in missing_optional:
            print(dep)
        print("Some tests may be skipped.")

    return True


def setup_test_environment():
    """Setup test environment variables."""
    # Set test-specific environment variables
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests

    # Ensure test database isolation
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for MetaTrader Python Framework database layer"
    )

    parser.add_argument(
        "suite",
        nargs="?",
        choices=["unit", "integration", "performance", "coverage", "all"],
        default="all",
        help="Test suite to run (default: all)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "-p", "--pattern",
        help="Test pattern to match"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests (performance tests)"
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (coverage suite only)"
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip environment validation"
    )

    args = parser.parse_args()

    # Validate environment unless skipped
    if not args.no_validate and not validate_environment():
        return 1

    # Setup test environment
    setup_test_environment()

    # Run selected test suite
    if args.suite == "unit":
        return run_unit_tests(args.verbose, args.pattern)
    elif args.suite == "integration":
        return run_integration_tests(args.verbose, args.pattern)
    elif args.suite == "performance":
        return run_performance_tests(args.verbose, args.pattern)
    elif args.suite == "coverage":
        return run_coverage_tests(args.html)
    elif args.suite == "all":
        return run_all_tests(args.verbose, args.fast)
    else:
        print(f"Unknown test suite: {args.suite}")
        return 1


if __name__ == "__main__":
    sys.exit(main())