#!/usr/bin/env python3
"""Test script to verify LoggerFactory auto-initialization works."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing LoggerFactory auto-initialization...")

try:
    # Import logging directly
    print("1. Importing get_logger...")
    from src.core.logging import get_logger

    print("2. Getting logger (should auto-initialize)...")
    logger = get_logger(__name__)

    print("3. Testing logger functionality...")
    logger.info("LoggerFactory auto-initialization test successful!")

    print("✓ All tests passed! LoggerFactory auto-initialization is working.")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()