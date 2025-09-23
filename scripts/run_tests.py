#!/usr/bin/env python3
"""
Test runner script for MetaTrader Python Framework.

This script provides a convenient way to run tests with various options
including coverage reporting, performance monitoring, and filtered execution.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(command: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """
    Run a command and handle errors.

    Args:
        command: Command to run as list of arguments
        capture_output: Whether to capture output

    Returns:
        Completed process
    """
    print(f"Running: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=False
        )
        return result
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Please install it first.")
        sys.exit(1)


def install_dependencies():
    """Install test dependencies."""
    print("Installing test dependencies...")

    # Install from requirements
    run_command([
        sys.executable, "-m", "pip", "install",
        "-r", "requirements/testing.txt"
    ])

    # Install development dependencies for full testing
    run_command([
        sys.executable, "-m", "pip", "install",
        "-r", "requirements/development.txt"
    ])


def run_pytest(
    test_path: Optional[str] = None,
    markers: Optional[str] = None,
    coverage: bool = False,
    verbose: bool = False,
    parallel: bool = False,
    report_formats: Optional[List[str]] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    """
    Run pytest with specified options.

    Args:
        test_path: Specific test path to run
        markers: Pytest markers to filter tests
        coverage: Whether to run with coverage
        verbose: Verbose output
        parallel: Run tests in parallel
        report_formats: Coverage report formats
        extra_args: Additional pytest arguments

    Returns:
        Exit code
    """
    cmd = [sys.executable, "-m", "pytest"]

    # Add test path
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    # Add markers
    if markers:
        cmd.extend(["-m", markers])

    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("--tb=short")

    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])

    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
        ])

        # Add additional coverage report formats
        if report_formats:
            for fmt in report_formats:
                if fmt == "html":
                    cmd.append("--cov-report=html:htmlcov")
                elif fmt == "xml":
                    cmd.append("--cov-report=xml")
                elif fmt == "json":
                    cmd.append("--cov-report=json")

    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)

    # Run pytest
    result = run_command(cmd)
    return result.returncode


def run_linting():
    """Run code linting."""
    print("\n" + "="*50)
    print("Running code linting...")
    print("="*50)

    exit_code = 0

    # Run flake8
    print("\nRunning flake8...")
    result = run_command([sys.executable, "-m", "flake8", "src", "tests"])
    if result.returncode != 0:
        exit_code = 1

    # Run mypy
    print("\nRunning mypy...")
    result = run_command([sys.executable, "-m", "mypy", "src"])
    if result.returncode != 0:
        exit_code = 1

    # Run bandit for security
    print("\nRunning bandit...")
    result = run_command([
        sys.executable, "-m", "bandit",
        "-r", "src",
        "-f", "json",
        "-o", "bandit-report.json"
    ])
    if result.returncode != 0:
        exit_code = 1

    return exit_code


def run_formatting_check():
    """Check code formatting."""
    print("\n" + "="*50)
    print("Checking code formatting...")
    print("="*50)

    # Check with black
    print("\nChecking black formatting...")
    result = run_command([sys.executable, "-m", "black", "--check", "src", "tests"])
    black_exit = result.returncode

    # Check with isort
    print("\nChecking isort formatting...")
    result = run_command([sys.executable, "-m", "isort", "--check-only", "src", "tests"])
    isort_exit = result.returncode

    if black_exit != 0:
        print("\nCode is not formatted with black. Run: python -m black src tests")

    if isort_exit != 0:
        print("\nImports are not sorted. Run: python -m isort src tests")

    return max(black_exit, isort_exit)


def format_code():
    """Format code automatically."""
    print("\n" + "="*50)
    print("Formatting code...")
    print("="*50)

    # Format with black
    print("\nFormatting with black...")
    run_command([sys.executable, "-m", "black", "src", "tests"])

    # Sort imports with isort
    print("\nSorting imports with isort...")
    run_command([sys.executable, "-m", "isort", "src", "tests"])


def generate_test_report():
    """Generate comprehensive test report."""
    print("\n" + "="*50)
    print("Generating comprehensive test report...")
    print("="*50)

    # Run tests with coverage and multiple report formats
    exit_code = run_pytest(
        coverage=True,
        verbose=True,
        report_formats=["html", "xml", "json"],
        extra_args=["--junitxml=junit-report.xml"]
    )

    # Run linting
    lint_exit = run_linting()

    # Check formatting
    format_exit = run_formatting_check()

    print("\n" + "="*50)
    print("Test Report Summary")
    print("="*50)
    print(f"Tests: {'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"Linting: {'PASSED' if lint_exit == 0 else 'FAILED'}")
    print(f"Formatting: {'PASSED' if format_exit == 0 else 'FAILED'}")

    if exit_code == 0 and lint_exit == 0 and format_exit == 0:
        print("\n✅ All checks passed!")
        return 0
    else:
        print("\n❌ Some checks failed!")
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test runner for MetaTrader Python Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py                          # Run all tests
  python scripts/run_tests.py --unit                   # Run only unit tests
  python scripts/run_tests.py --integration            # Run only integration tests
  python scripts/run_tests.py --coverage               # Run with coverage
  python scripts/run_tests.py --parallel               # Run in parallel
  python scripts/run_tests.py --lint                   # Run linting only
  python scripts/run_tests.py --format                 # Format code
  python scripts/run_tests.py --full-report            # Generate full report
  python scripts/run_tests.py tests/unit/core/         # Run specific test directory
  python scripts/run_tests.py -k test_config           # Run tests matching pattern
        """
    )

    # Test selection options
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Specific test path to run"
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )

    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )

    parser.add_argument(
        "--mt5",
        action="store_true",
        help="Include MT5 tests (requires MT5 installation)"
    )

    parser.add_argument(
        "-k",
        "--keyword",
        help="Run tests matching keyword expression"
    )

    # Execution options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    # Report options
    parser.add_argument(
        "--html-cov",
        action="store_true",
        help="Generate HTML coverage report"
    )

    parser.add_argument(
        "--xml-cov",
        action="store_true",
        help="Generate XML coverage report"
    )

    # Tool options
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run linting only"
    )

    parser.add_argument(
        "--format",
        action="store_true",
        help="Format code automatically"
    )

    parser.add_argument(
        "--format-check",
        action="store_true",
        help="Check code formatting"
    )

    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies"
    )

    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate comprehensive test report"
    )

    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    # Install dependencies if requested
    if args.install_deps:
        install_dependencies()
        return 0

    # Format code if requested
    if args.format:
        format_code()
        return 0

    # Check formatting if requested
    if args.format_check:
        return run_formatting_check()

    # Run linting if requested
    if args.lint:
        return run_linting()

    # Generate full report if requested
    if args.full_report:
        return generate_test_report()

    # Build markers
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.mt5:
        markers.append("mt5")
    if not args.slow:
        markers.append("not slow")

    marker_string = " and ".join(markers) if markers else None

    # Build report formats
    report_formats = []
    if args.html_cov:
        report_formats.append("html")
    if args.xml_cov:
        report_formats.append("xml")

    # Build extra args
    extra_args = []
    if args.keyword:
        extra_args.extend(["-k", args.keyword])

    # Run tests
    return run_pytest(
        test_path=args.test_path,
        markers=marker_string,
        coverage=args.coverage,
        verbose=args.verbose,
        parallel=args.parallel,
        report_formats=report_formats or None,
        extra_args=extra_args or None,
    )


if __name__ == "__main__":
    sys.exit(main())