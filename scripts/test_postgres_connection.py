#!/usr/bin/env python3
"""Quick test of PostgreSQL connection for conductor-memory.

Usage:
    python scripts/test_postgres_connection.py
    python scripts/test_postgres_connection.py --config ~/.conductor-memory/config.json

Reads postgres_url from config file or CONDUCTOR_POSTGRES_URL environment variable.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conductor_memory.storage.postgres import PostgresMetadataStore


def get_postgres_url(config_path: str = None) -> str:
    """Get PostgreSQL URL from config or environment."""
    # Check environment variable first
    env_url = os.environ.get("CONDUCTOR_POSTGRES_URL")
    if env_url:
        return env_url

    # Try config file
    if config_path is None:
        config_path = Path.home() / ".conductor-memory" / "config.json"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            if "postgres_url" in config:
                return config["postgres_url"]

    return None


async def test_connection(conn_str: str):
    print(f"Testing connection to PostgreSQL...")
    # Mask password in output
    masked = conn_str.split("@")[1] if "@" in conn_str else conn_str
    print(f"Host: {masked}")

    store = PostgresMetadataStore(conn_str)

    try:
        await store.connect()
        print("[OK] Connected successfully!")

        # Test query
        counts = await store.get_counts()
        print(f"[OK] Query successful: {counts}")

        await store.close()
        print("[OK] Connection closed")

    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Test PostgreSQL connection")
    parser.add_argument("--config", help="Path to config.json file")
    parser.add_argument("--url", help="PostgreSQL connection URL (overrides config)")
    args = parser.parse_args()

    conn_str = args.url or get_postgres_url(args.config)

    if not conn_str:
        print("ERROR: No PostgreSQL URL found.")
        print("Provide via --url, config file, or CONDUCTOR_POSTGRES_URL env var.")
        sys.exit(1)

    success = asyncio.run(test_connection(conn_str))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
