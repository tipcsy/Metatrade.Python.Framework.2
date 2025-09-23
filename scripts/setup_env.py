#!/usr/bin/env python3
"""
Environment setup script for MetaTrader Python Framework.

This script sets up the development environment, creates necessary directories,
initializes configuration files, and performs system checks.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class EnvironmentSetup:
    """Environment setup manager."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize environment setup.

        Args:
            project_root: Project root directory (defaults to script's parent)
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = project_root

        self.colors = {
            "RED": "\033[0;31m",
            "GREEN": "\033[0;32m",
            "YELLOW": "\033[0;33m",
            "BLUE": "\033[0;34m",
            "PURPLE": "\033[0;35m",
            "CYAN": "\033[0;36m",
            "WHITE": "\033[0;37m",
            "RESET": "\033[0m",
        }

    def print_colored(self, message: str, color: str = "WHITE") -> None:
        """Print colored message."""
        color_code = self.colors.get(color.upper(), self.colors["WHITE"])
        print(f"{color_code}{message}{self.colors['RESET']}")

    def print_header(self, title: str) -> None:
        """Print section header."""
        self.print_colored("\n" + "=" * 60, "CYAN")
        self.print_colored(f" {title}", "CYAN")
        self.print_colored("=" * 60, "CYAN")

    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        self.print_header("Checking Python Version")

        version = sys.version_info
        required_major, required_minor = 3, 9

        self.print_colored(f"Current Python version: {version.major}.{version.minor}.{version.micro}")

        if version.major < required_major or (version.major == required_major and version.minor < required_minor):
            self.print_colored(
                f"❌ Python {required_major}.{required_minor}+ is required. "
                f"Current version: {version.major}.{version.minor}.{version.micro}",
                "RED"
            )
            return False

        self.print_colored(f"✅ Python version is compatible", "GREEN")
        return True

    def check_system_dependencies(self) -> Dict[str, bool]:
        """Check system dependencies."""
        self.print_header("Checking System Dependencies")

        dependencies = {
            "git": ["git", "--version"],
            "make": ["make", "--version"],
        }

        results = {}

        for name, command in dependencies.items():
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.print_colored(f"✅ {name}: Available", "GREEN")
                results[name] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.print_colored(f"❌ {name}: Not found", "RED")
                results[name] = False

        return results

    def create_directories(self) -> None:
        """Create necessary project directories."""
        self.print_header("Creating Project Directories")

        directories = [
            "logs",
            "data",
            "data/backups",
            "data/cache",
            "data/market_data",
            "config",
            "scripts",
            "docs/build",
            "htmlcov",
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.print_colored(f"✅ Created: {directory}", "GREEN")

    def copy_env_example(self) -> None:
        """Copy .env.example to .env if it doesn't exist."""
        self.print_header("Setting Up Environment Variables")

        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"

        if env_example.exists() and not env_file.exists():
            shutil.copy2(env_example, env_file)
            self.print_colored("✅ Created .env file from .env.example", "GREEN")
            self.print_colored("Please review and update .env with your settings", "YELLOW")
        elif env_file.exists():
            self.print_colored("✅ .env file already exists", "GREEN")
        else:
            self.print_colored("❌ .env.example not found", "RED")

    def initialize_git_hooks(self) -> None:
        """Initialize git hooks if in a git repository."""
        self.print_header("Setting Up Git Hooks")

        git_dir = self.project_root / ".git"

        if not git_dir.exists():
            self.print_colored("❌ Not a git repository", "YELLOW")
            return

        try:
            # Check if pre-commit is available
            subprocess.run(["pre-commit", "--version"], capture_output=True, check=True)

            # Install pre-commit hooks
            result = subprocess.run(
                ["pre-commit", "install"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                self.print_colored("✅ Pre-commit hooks installed", "GREEN")
            else:
                self.print_colored("❌ Failed to install pre-commit hooks", "RED")

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.print_colored("❌ pre-commit not installed", "YELLOW")
            self.print_colored("Install with: pip install pre-commit", "YELLOW")

    def check_virtual_environment(self) -> bool:
        """Check if running in virtual environment."""
        self.print_header("Checking Virtual Environment")

        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.print_colored("✅ Running in virtual environment", "GREEN")
            env_path = os.environ.get('VIRTUAL_ENV', 'Unknown')
            self.print_colored(f"Environment path: {env_path}", "BLUE")
            return True
        else:
            self.print_colored("❌ Not running in virtual environment", "YELLOW")
            self.print_colored("Recommendation: Use virtual environment for development", "YELLOW")
            return False

    def install_development_dependencies(self) -> bool:
        """Install development dependencies."""
        self.print_header("Installing Development Dependencies")

        requirements_file = self.project_root / "requirements" / "development.txt"

        if not requirements_file.exists():
            self.print_colored("❌ development.txt not found", "RED")
            return False

        try:
            self.print_colored("Installing development dependencies...", "BLUE")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                self.print_colored("✅ Development dependencies installed", "GREEN")
                return True
            else:
                self.print_colored("❌ Failed to install dependencies", "RED")
                self.print_colored(result.stderr, "RED")
                return False

        except Exception as e:
            self.print_colored(f"❌ Error installing dependencies: {e}", "RED")
            return False

    def install_package_in_development_mode(self) -> bool:
        """Install package in development mode."""
        self.print_header("Installing Package in Development Mode")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                self.print_colored("✅ Package installed in development mode", "GREEN")
                return True
            else:
                self.print_colored("❌ Failed to install package", "RED")
                self.print_colored(result.stderr, "RED")
                return False

        except Exception as e:
            self.print_colored(f"❌ Error installing package: {e}", "RED")
            return False

    def run_initial_tests(self) -> bool:
        """Run initial tests to verify setup."""
        self.print_header("Running Initial Tests")

        try:
            # Check if pytest is available
            subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True, check=True)

            # Run a simple test
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                self.print_colored("✅ Initial tests passed", "GREEN")
                return True
            else:
                self.print_colored("❌ Some tests failed", "YELLOW")
                self.print_colored("This might be expected for a new setup", "YELLOW")
                return False

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.print_colored("❌ pytest not available", "YELLOW")
            return False

    def create_default_configs(self) -> None:
        """Create default configuration files."""
        self.print_header("Creating Default Configuration Files")

        try:
            # Import here to avoid circular imports
            from src.core.config import ConfigLoader

            config_dir = self.project_root / "config"
            loader = ConfigLoader(config_dir)
            loader.create_default_configs()

            self.print_colored("✅ Default configuration files created", "GREEN")

        except ImportError as e:
            self.print_colored(f"❌ Cannot import configuration module: {e}", "RED")
        except Exception as e:
            self.print_colored(f"❌ Error creating configs: {e}", "RED")

    def print_summary(self, results: Dict[str, bool]) -> None:
        """Print setup summary."""
        self.print_header("Setup Summary")

        total_checks = len(results)
        passed_checks = sum(results.values())

        self.print_colored(f"Setup checks: {passed_checks}/{total_checks} passed", "CYAN")

        for check, status in results.items():
            status_icon = "✅" if status else "❌"
            color = "GREEN" if status else "RED"
            self.print_colored(f"{status_icon} {check}", color)

        if passed_checks == total_checks:
            self.print_colored("\n🎉 Environment setup completed successfully!", "GREEN")
            self.print_colored("You can now start development with:", "GREEN")
            self.print_colored("  make test", "BLUE")
            self.print_colored("  make run-dev", "BLUE")
        else:
            self.print_colored("\n⚠️  Environment setup completed with some issues", "YELLOW")
            self.print_colored("Please review the failed checks above", "YELLOW")

    def print_next_steps(self) -> None:
        """Print next steps for the user."""
        self.print_header("Next Steps")

        steps = [
            "Review and update .env file with your settings",
            "Configure MetaTrader 5 connection settings if needed",
            "Run 'make test' to verify everything is working",
            "Run 'make run-dev' to start the development server",
            "Check the documentation in docs/ for more information",
            "Set up your IDE/editor with the project",
        ]

        for i, step in enumerate(steps, 1):
            self.print_colored(f"{i}. {step}", "BLUE")

    def run_setup(self, install_deps: bool = True, run_tests: bool = True) -> bool:
        """
        Run complete environment setup.

        Args:
            install_deps: Whether to install dependencies
            run_tests: Whether to run initial tests

        Returns:
            True if setup completed successfully
        """
        self.print_colored("\n🚀 MetaTrader Python Framework - Environment Setup", "CYAN")

        results = {}

        # Core checks
        results["Python Version"] = self.check_python_version()
        results["Virtual Environment"] = self.check_virtual_environment()

        # System dependencies
        sys_deps = self.check_system_dependencies()
        results.update(sys_deps)

        # Setup steps
        self.create_directories()
        self.copy_env_example()
        self.create_default_configs()

        # Git setup
        self.initialize_git_hooks()

        # Package installation
        if install_deps:
            results["Development Dependencies"] = self.install_development_dependencies()
            results["Package Installation"] = self.install_package_in_development_mode()

        # Testing
        if run_tests:
            results["Initial Tests"] = self.run_initial_tests()

        # Summary
        self.print_summary(results)
        self.print_next_steps()

        return all(results.values())


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up development environment for MetaTrader Python Framework"
    )

    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip dependency installation"
    )

    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip initial test run"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory"
    )

    args = parser.parse_args()

    # Initialize setup
    setup = EnvironmentSetup(args.project_root)

    # Run setup
    success = setup.run_setup(
        install_deps=not args.no_deps,
        run_tests=not args.no_tests
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()