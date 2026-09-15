"""
Script 01: Initialize SQLite Database Schema
============================================
Creates or verifies data/processed/database/aic2026.db
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager, DEFAULT_DB_PATH

def main():
    print("=" * 70)
    print("  PHASE 1 - SCRIPT 01: INITIALIZE SQLITE DATABASE (aic2026.db)")
    print("=" * 70)
    print(f"Target DB Path: {DEFAULT_DB_PATH}")

    db = DatabaseManager(DEFAULT_DB_PATH)
    
    # Verify tables
    tables = db.execute_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    print("\nDatabase initialized successfully. Tables created:")
    for t in tables:
        print(f"  - {t['name']}")

if __name__ == "__main__":
    main()
