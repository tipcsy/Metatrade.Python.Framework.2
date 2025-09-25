#!/usr/bin/env python3
"""Debug script to find the exact source of LoggerFactory error."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Starting import debugging...")

try:
    print("1. Importing src.core.config...")
    from src.core.config import get_settings
    print("   ✓ Config imported successfully")

    print("2. Getting settings...")
    settings = get_settings()
    print("   ✓ Settings loaded successfully")

    print("3. Importing src.core.logging...")
    from src.core.logging import setup_logging, get_logger
    print("   ✓ Logging modules imported successfully")

    print("4. Setting up logging...")
    setup_logging(settings)
    print("   ✓ Logging setup successfully")

    print("5. Getting logger...")
    logger = get_logger(__name__)
    print("   ✓ Logger obtained successfully")

    print("6. Importing GUI modules...")
    from src.gui.app import create_application, run_application
    print("   ✓ GUI modules imported successfully")

    print("All imports successful! The error must occur during application creation.")

except Exception as e:
    print(f"Error at step: {e}")
    import traceback
    traceback.print_exc()