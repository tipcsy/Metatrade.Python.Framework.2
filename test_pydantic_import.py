#!/usr/bin/env python3
"""Test Pydantic imports to debug field_validator issue."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing Pydantic imports...")

try:
    import pydantic
    print(f"✓ Pydantic version: {pydantic.__version__}")

    print("Testing field_validator import...")
    from pydantic import field_validator
    print("✓ field_validator imported successfully")

    print("Testing model_validator import...")
    from pydantic import model_validator
    print("✓ model_validator imported successfully")

    print("Testing BaseModel import...")
    from pydantic import BaseModel, Field
    print("✓ BaseModel and Field imported successfully")

    print("Testing BaseSettings import...")
    from pydantic_settings import BaseSettings, SettingsConfigDict
    print("✓ BaseSettings and SettingsConfigDict imported successfully")

    print("All Pydantic imports successful!")

except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()