# Makefile for MetaTrader Python Framework
# Provides common development tasks and automation

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := metatrader-python-framework
SRC_DIR := src
TESTS_DIR := tests
SCRIPTS_DIR := scripts
REQUIREMENTS_DIR := requirements

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: help install install-dev install-prod clean test test-unit test-integration test-slow test-coverage lint format format-check docs build publish setup-env check-deps security-check type-check all-checks pre-commit docker-build docker-run

## Help target
help: ## Show this help message
	@echo "$(CYAN)MetaTrader Python Framework - Development Commands$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(RESET)"
	@echo "  make install-dev     # Install development dependencies"
	@echo "  make test           # Run all tests"
	@echo "  make test-coverage  # Run tests with coverage"
	@echo "  make lint           # Run linting"
	@echo "  make format         # Format code"
	@echo "  make all-checks     # Run all quality checks"
	@echo ""

## Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(PIP) install -r $(REQUIREMENTS_DIR)/base.txt
	@echo "$(GREEN)Production dependencies installed successfully!$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -r $(REQUIREMENTS_DIR)/development.txt
	$(PIP) install -e .
	@echo "$(GREEN)Development environment set up successfully!$(RESET)"

install-prod: ## Install production dependencies optimized
	@echo "$(BLUE)Installing production dependencies (optimized)...$(RESET)"
	$(PIP) install --no-deps -r $(REQUIREMENTS_DIR)/production.txt
	@echo "$(GREEN)Production dependencies installed successfully!$(RESET)"

install-test: ## Install testing dependencies only
	@echo "$(BLUE)Installing testing dependencies...$(RESET)"
	$(PIP) install -r $(REQUIREMENTS_DIR)/testing.txt
	@echo "$(GREEN)Testing dependencies installed successfully!$(RESET)"

## Development environment targets
setup-env: ## Set up complete development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/setup_env.py
	@$(MAKE) install-dev
	@$(MAKE) setup-git-hooks
	@echo "$(GREEN)Development environment ready!$(RESET)"

setup-git-hooks: ## Set up git pre-commit hooks
	@echo "$(BLUE)Setting up git hooks...$(RESET)"
	$(PYTHON) -m pre_commit install
	$(PYTHON) -m pre_commit install --hook-type commit-msg
	@echo "$(GREEN)Git hooks installed successfully!$(RESET)"

## Cleaning targets
clean: ## Clean build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	@echo "$(GREEN)Cleaned successfully!$(RESET)"

clean-logs: ## Clean log files
	@echo "$(BLUE)Cleaning log files...$(RESET)"
	rm -rf logs/*.log
	rm -rf logs/*.log.*
	@echo "$(GREEN)Log files cleaned!$(RESET)"

clean-data: ## Clean data files (use with caution!)
	@echo "$(YELLOW)WARNING: This will delete all data files!$(RESET)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "$(BLUE)Cleaning data files...$(RESET)"; \
		rm -rf data/*.db*; \
		rm -rf data/backups/*; \
		rm -rf data/cache/*; \
		echo "$(GREEN)Data files cleaned!$(RESET)"; \
	else \
		echo ""; \
		echo "$(YELLOW)Cancelled.$(RESET)"; \
	fi

## Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py
	@echo "$(GREEN)Tests completed!$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --unit
	@echo "$(GREEN)Unit tests completed!$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --integration
	@echo "$(GREEN)Integration tests completed!$(RESET)"

test-slow: ## Run all tests including slow ones
	@echo "$(BLUE)Running all tests including slow ones...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --slow
	@echo "$(GREEN)All tests completed!$(RESET)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --coverage --html-cov
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --parallel
	@echo "$(GREEN)Parallel tests completed!$(RESET)"

test-mt5: ## Run MetaTrader 5 specific tests
	@echo "$(BLUE)Running MT5 tests...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --mt5
	@echo "$(GREEN)MT5 tests completed!$(RESET)"

## Code quality targets
lint: ## Run linting (flake8, mypy, bandit)
	@echo "$(BLUE)Running linting...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --lint
	@echo "$(GREEN)Linting completed!$(RESET)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --format
	@echo "$(GREEN)Code formatted successfully!$(RESET)"

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --format-check
	@echo "$(GREEN)Format check completed!$(RESET)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(RESET)"
	$(PYTHON) -m mypy $(SRC_DIR)
	@echo "$(GREEN)Type checking completed!$(RESET)"

security-check: ## Run security checks with bandit
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(PYTHON) -m bandit -r $(SRC_DIR) -f json -o bandit-report.json
	@echo "$(GREEN)Security check completed! Report: bandit-report.json$(RESET)"

## Dependency management
check-deps: ## Check dependencies for security issues
	@echo "$(BLUE)Checking dependencies...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/check_deps.py
	@echo "$(GREEN)Dependency check completed!$(RESET)"

update-deps: ## Update dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(PYTHON) -m pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 $(PIP) install -U
	@echo "$(GREEN)Dependencies updated!$(RESET)"

pip-audit: ## Run pip audit for security vulnerabilities
	@echo "$(BLUE)Running pip audit...$(RESET)"
	$(PYTHON) -m pip install pip-audit
	$(PYTHON) -m pip-audit
	@echo "$(GREEN)Pip audit completed!$(RESET)"

## Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	cd docs && $(PYTHON) -m sphinx -b html source build
	@echo "$(GREEN)Documentation generated in docs/build/$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	cd docs/build && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation...$(RESET)"
	rm -rf docs/build/
	@echo "$(GREEN)Documentation cleaned!$(RESET)"

## Build and distribution targets
build: ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)Distribution packages built in dist/$(RESET)"

build-wheel: ## Build wheel package only
	@echo "$(BLUE)Building wheel package...$(RESET)"
	$(PYTHON) -m build --wheel
	@echo "$(GREEN)Wheel package built!$(RESET)"

build-sdist: ## Build source distribution only
	@echo "$(BLUE)Building source distribution...$(RESET)"
	$(PYTHON) -m build --sdist
	@echo "$(GREEN)Source distribution built!$(RESET)"

## Publishing targets
publish-test: ## Publish to test PyPI
	@echo "$(BLUE)Publishing to test PyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)Published to test PyPI!$(RESET)"

publish: ## Publish to PyPI
	@echo "$(YELLOW)WARNING: This will publish to PyPI!$(RESET)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "$(BLUE)Publishing to PyPI...$(RESET)"; \
		$(PYTHON) -m twine upload dist/*; \
		echo "$(GREEN)Published to PyPI!$(RESET)"; \
	else \
		echo ""; \
		echo "$(YELLOW)Cancelled.$(RESET)"; \
	fi

## Comprehensive check targets
all-checks: ## Run all quality checks
	@echo "$(BLUE)Running all quality checks...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_tests.py --full-report
	@echo "$(GREEN)All checks completed!$(RESET)"

pre-commit: ## Run pre-commit checks
	@echo "$(BLUE)Running pre-commit checks...$(RESET)"
	$(PYTHON) -m pre_commit run --all-files
	@echo "$(GREEN)Pre-commit checks completed!$(RESET)"

ci-check: ## Run CI-like checks locally
	@echo "$(BLUE)Running CI checks...$(RESET)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) security-check
	@$(MAKE) test-coverage
	@echo "$(GREEN)CI checks completed!$(RESET)"

## Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)Docker image built!$(RESET)"

docker-run: ## Run application in Docker
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run --rm -it $(PROJECT_NAME):latest
	@echo "$(GREEN)Docker container finished!$(RESET)"

docker-test: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(RESET)"
	docker build -f Dockerfile.test -t $(PROJECT_NAME):test .
	docker run --rm $(PROJECT_NAME):test
	@echo "$(GREEN)Docker tests completed!$(RESET)"

## Development utilities
run-dev: ## Run application in development mode
	@echo "$(BLUE)Starting development server...$(RESET)"
	ENVIRONMENT=development $(PYTHON) -m $(SRC_DIR).main

run-prod: ## Run application in production mode
	@echo "$(BLUE)Starting production server...$(RESET)"
	ENVIRONMENT=production $(PYTHON) -m $(SRC_DIR).main

shell: ## Open interactive Python shell with project context
	@echo "$(BLUE)Opening Python shell...$(RESET)"
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC_DIR)'); import IPython; IPython.start_ipython()"

notebook: ## Start Jupyter notebook
	@echo "$(BLUE)Starting Jupyter notebook...$(RESET)"
	$(PYTHON) -m jupyter notebook

## Profiling and debugging
profile: ## Run profiling
	@echo "$(BLUE)Running profiler...$(RESET)"
	$(PYTHON) -m cProfile -o profile_output.prof -m $(SRC_DIR).main
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)Profiling completed!$(RESET)"

debug: ## Run with debugger
	@echo "$(BLUE)Starting debugger...$(RESET)"
	$(PYTHON) -m pdb -m $(SRC_DIR).main

## Database management
db-create: ## Create database
	@echo "$(BLUE)Creating database...$(RESET)"
	$(PYTHON) -c "from $(SRC_DIR).database import create_database; create_database()"
	@echo "$(GREEN)Database created!$(RESET)"

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	$(PYTHON) -c "from $(SRC_DIR).database import migrate_database; migrate_database()"
	@echo "$(GREEN)Database migrated!$(RESET)"

db-seed: ## Seed database with test data
	@echo "$(BLUE)Seeding database...$(RESET)"
	$(PYTHON) -c "from $(SRC_DIR).database import seed_database; seed_database()"
	@echo "$(GREEN)Database seeded!$(RESET)"

## Backup and restore
backup: ## Create backup
	@echo "$(BLUE)Creating backup...$(RESET)"
	$(PYTHON) -c "from $(SRC_DIR).core.utils import create_backup; create_backup()"
	@echo "$(GREEN)Backup created!$(RESET)"

restore: ## Restore from backup
	@echo "$(YELLOW)WARNING: This will restore from the latest backup!$(RESET)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "$(BLUE)Restoring from backup...$(RESET)"; \
		$(PYTHON) -c "from $(SRC_DIR).core.utils import restore_backup; restore_backup()"; \
		echo "$(GREEN)Restored from backup!$(RESET)"; \
	else \
		echo ""; \
		echo "$(YELLOW)Cancelled.$(RESET)"; \
	fi

## Status and information
status: ## Show project status
	@echo "$(CYAN)MetaTrader Python Framework Status$(RESET)"
	@echo "=================================="
	@echo "$(YELLOW)Python Version:$(RESET) $$($(PYTHON) --version)"
	@echo "$(YELLOW)Project Directory:$(RESET) $$(pwd)"
	@echo "$(YELLOW)Git Branch:$(RESET) $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "$(YELLOW)Git Status:$(RESET) $$(git status --porcelain 2>/dev/null | wc -l || echo 'N/A') modified files"
	@echo "$(YELLOW)Virtual Environment:$(RESET) $$(echo $$VIRTUAL_ENV || echo 'None')"
	@echo "$(YELLOW)Installed Packages:$(RESET) $$($(PIP) list | wc -l) packages"
	@echo ""

version: ## Show version information
	@echo "$(CYAN)Version Information$(RESET)"
	@echo "==================="
	@$(PYTHON) -c "from $(SRC_DIR).core.config import Settings; s = Settings(); print(f'Application: {s.app_name} v{s.app_version}')"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo ""

info: ## Show detailed project information
	@$(MAKE) status
	@$(MAKE) version
	@echo "$(YELLOW)Dependencies:$(RESET)"
	@$(PIP) list | head -10
	@echo "..."
	@echo ""