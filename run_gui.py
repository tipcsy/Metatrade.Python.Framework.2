#!/usr/bin/env python3
"""
Startup script for the MetaTrader Python Framework GUI.

This script provides a convenient entry point for running the GUI application
with proper initialization and error handling.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    """Main entry point for GUI application."""
    try:
        from src.gui.main import main as gui_main
        return gui_main()
    except ImportError as e:
        print(f"Error importing GUI modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements/base.txt")
        return 1
    except Exception as e:
        print(f"Error starting GUI application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())