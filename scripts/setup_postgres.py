#!/usr/bin/env python3
"""
Setup PostgreSQL database for Conductor Memory.

Run this script once to create the database and schema.

Usage:
    python scripts/setup_postgres.py --host localhost --user postgres

This will:
1. Create the 'conductor_memory' database (if it doesn't exist)
2. Create all required tables and indexes
3. Create the materialized view for fast stats
"""

import argparse
import sys
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("ERROR: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)


def create_database(host: str, user: str, password: str = "", port: int = 5432):
    """Create the conductor_memory database if it doesn't exist."""
    print(f"Connecting to PostgreSQL at {host}:{port} as {user}...")

    # Connect to default 'postgres' database to create our database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Check if database exists
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = 'conductor_memory'"
    )
    exists = cursor.fetchone()

    if exists:
        print("Database 'conductor_memory' already exists")
    else:
        print("Creating database 'conductor_memory'...")
        cursor.execute("CREATE DATABASE conductor_memory")
        print("Database created!")

    cursor.close()
    conn.close()


def create_schema(host: str, user: str, password: str = "", port: int = 5432):
    """Create the schema in the conductor_memory database."""
    print("Creating schema...")

    # Read schema SQL
    schema_path = Path(__file__).parent.parent / "src" / "conductor_memory" / "storage" / "schema.sql"
    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}")
        sys.exit(1)

    schema_sql = schema_path.read_text()

    # Connect to conductor_memory database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database="conductor_memory"
    )
    cursor = conn.cursor()

    try:
        cursor.execute(schema_sql)
        conn.commit()
        print("Schema created successfully!")
    except psycopg2.Error as e:
        print(f"Error creating schema: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def verify_setup(host: str, user: str, password: str = "", port: int = 5432):
    """Verify the database setup."""
    print("\nVerifying setup...")

    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database="conductor_memory"
    )
    cursor = conn.cursor()

    # Check tables exist
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cursor.fetchall()]

    print(f"Tables: {', '.join(tables)}")

    expected_tables = ['codebases', 'indexed_files', 'summaries']
    missing = set(expected_tables) - set(tables)
    if missing:
        print(f"WARNING: Missing tables: {missing}")
    else:
        print("✓ All required tables exist")

    # Check materialized view
    cursor.execute("""
        SELECT matviewname FROM pg_matviews
        WHERE schemaname = 'public'
    """)
    views = [row[0] for row in cursor.fetchall()]
    print(f"Materialized views: {', '.join(views) if views else 'none'}")

    if 'codebase_stats' in views:
        print("✓ codebase_stats materialized view exists")
    else:
        print("WARNING: codebase_stats materialized view not found")

    cursor.close()
    conn.close()

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print(f"\nAdd this to your config.json:")
    print(f'  "postgres_url": "postgresql://{user}@{host}:{port}/conductor_memory"')


def main():
    parser = argparse.ArgumentParser(description="Setup PostgreSQL for Conductor Memory")
    parser.add_argument("--host", required=True, help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--user", required=True, help="PostgreSQL user")
    parser.add_argument("--password", default="", help="PostgreSQL password")
    parser.add_argument("--skip-create-db", action="store_true",
                        help="Skip database creation (if already exists)")

    args = parser.parse_args()

    try:
        if not args.skip_create_db:
            create_database(args.host, args.user, args.password, args.port)

        create_schema(args.host, args.user, args.password, args.port)
        verify_setup(args.host, args.user, args.password, args.port)

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
