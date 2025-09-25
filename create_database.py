#!/usr/bin/env python3
"""
Create database tables for MetaTrader Python Framework.

This script creates all database tables defined in the models.
Run this script once to initialize the database.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Create all database tables."""
    print("Creating database tables...")

    try:
        from src.database.database import DatabaseManager

        # Initialize database manager
        print("Initializing database manager...")
        db_manager = DatabaseManager()

        # Initialize the database connection
        print("Initializing database connection...")
        db_manager.initialize()

        # Create all tables
        print("Creating all tables...")
        db_manager.create_all_tables()

        print("✅ Database tables created successfully!")
        return True

    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)