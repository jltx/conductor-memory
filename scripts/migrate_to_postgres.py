#!/usr/bin/env python3
"""
Migrate metadata from ChromaDB to PostgreSQL.

This script:
1. Reads all indexed files from ChromaDB
2. Reads all summaries from ChromaDB
3. Inserts them into PostgreSQL
4. Refreshes the materialized view

Usage:
    python scripts/migrate_to_postgres.py --config ~/.conductor-memory/config.json

Prerequisites:
    1. Run setup_postgres.py first to create the database
    2. Ensure conductor-memory is not running (to avoid conflicts)
    3. Install postgres support: pip install conductor-memory[postgres]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conductor_memory.config.server import ServerConfig
from conductor_memory.storage.chroma import ChromaVectorStore

try:
    from conductor_memory.storage.postgres import PostgresMetadataStore
except ImportError as e:
    print("ERROR: PostgreSQL support requires additional dependencies.")
    print("Install with: pip install conductor-memory[postgres]")
    print(f"Details: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def migrate(config_path: str, dry_run: bool = False):
    """Migrate all metadata from ChromaDB to PostgreSQL."""

    # Load config
    config = ServerConfig.from_file(config_path)

    if not config.postgres_url:
        logger.error("No postgres_url configured in config file")
        sys.exit(1)

    logger.info(f"Migrating to PostgreSQL: {config.postgres_url}")
    logger.info(f"ChromaDB mode: {config.chroma_mode}")
    logger.info(f"Codebases: {[c.name for c in config.get_enabled_codebases()]}")

    if dry_run:
        logger.info("DRY RUN - no changes will be made")

    # Connect to PostgreSQL
    postgres = PostgresMetadataStore(config.postgres_url)
    await postgres.connect()
    logger.info("Connected to PostgreSQL")

    total_files = 0
    total_summaries = 0

    # Process each codebase
    for codebase in config.get_enabled_codebases():
        logger.info(f"\n{'='*50}")
        logger.info(f"Migrating codebase: {codebase.name}")
        logger.info(f"{'='*50}")

        # Ensure codebase exists in PostgreSQL
        if not dry_run:
            codebase_id = await postgres.upsert_codebase(
                name=codebase.name,
                path=codebase.path,
                description=codebase.description,
                enabled=codebase.enabled
            )
            logger.info(f"  Codebase ID: {codebase_id}")

        # Connect to ChromaDB
        if config.chroma_mode == "http":
            chroma_store = ChromaVectorStore(
                collection_name=f"codebase_{codebase.name}",
                host=config.chroma_host,
                port=config.chroma_port
            )
        else:
            chroma_store = ChromaVectorStore(
                collection_name=f"codebase_{codebase.name}",
                persist_directory=config.persist_directory
            )

        # Get all indexed files from ChromaDB
        indexed_files_dict = chroma_store.file_index.get_all_indexed_files()
        indexed_files = [
            {'file_path': path, 'relative_path': path, **metadata}
            for path, metadata in indexed_files_dict.items()
        ]
        logger.info(f"  Found {len(indexed_files)} indexed files in ChromaDB")

        # Migrate files
        file_count = 0
        for file_data in indexed_files:
            file_path = file_data.get('file_path', file_data.get('path', ''))
            relative_path = file_data.get('relative_path', file_path)

            if not dry_run:
                try:
                    file_id = await postgres.upsert_file(
                        codebase_name=codebase.name,
                        file_path=file_path,
                        relative_path=relative_path,
                        content_hash=file_data.get('content_hash'),
                        file_size=file_data.get('file_size'),
                        line_count=file_data.get('line_count'),
                        language=file_data.get('language'),
                    )
                    file_count += 1
                except Exception as e:
                    logger.warning(f"  Failed to migrate file {relative_path}: {e}")
            else:
                file_count += 1

        logger.info(f"  Migrated {file_count} files")
        total_files += file_count

        # Get all summaries from ChromaDB
        try:
            summarized_files = chroma_store.summary_index.get_all_summarized_files()
            logger.info(f"  Found {len(summarized_files)} summaries in ChromaDB")

            # Migrate summaries
            summary_count = 0
            for file_path in summarized_files:
                try:
                    summary_data = chroma_store.summary_index.get_summary_info(file_path)
                    if not summary_data:
                        continue

                    if not dry_run:
                        # Get file_id from PostgreSQL
                        file_id = await postgres.get_file_id(codebase.name, file_path)
                        if not file_id:
                            # File not in PostgreSQL, try to add it
                            file_id = await postgres.upsert_file(
                                codebase_name=codebase.name,
                                file_path=str(Path(codebase.path) / file_path),
                                relative_path=file_path,
                            )

                        if file_id:
                            # Extract summary fields
                            summary_text = summary_data.get('summary', '')
                            if isinstance(summary_text, dict):
                                # Summary might be nested
                                summary_text = summary_text.get('summary', str(summary_text))

                            await postgres.upsert_summary(
                                file_id=file_id,
                                summary_text=summary_text,
                                summary_type=summary_data.get('summary_type', 'llm'),
                                pattern=summary_data.get('pattern'),
                                domain=summary_data.get('domain'),
                                key_functions=summary_data.get('key_functions', []),
                                dependencies=summary_data.get('dependencies', []),
                                validation_status=summary_data.get('validation_status', 'pending'),
                            )
                            summary_count += 1
                    else:
                        summary_count += 1

                except Exception as e:
                    logger.warning(f"  Failed to migrate summary for {file_path}: {e}")

            logger.info(f"  Migrated {summary_count} summaries")
            total_summaries += summary_count

        except Exception as e:
            logger.warning(f"  Failed to get summaries: {e}")

    # Refresh materialized view
    if not dry_run:
        logger.info("\nRefreshing materialized view...")
        await postgres.refresh_stats()

    # Verify migration
    logger.info("\n" + "="*50)
    logger.info("MIGRATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Total files migrated: {total_files}")
    logger.info(f"Total summaries migrated: {total_summaries}")

    if not dry_run:
        # Show counts from PostgreSQL
        counts = await postgres.get_counts()
        logger.info(f"\nPostgreSQL stats:")
        logger.info(f"  Indexed files: {counts.get('indexed_count', 0)}")
        logger.info(f"  Summarized files: {counts.get('summarized_count', 0)}")
        logger.info(f"  LLM summaries: {counts.get('llm_count', 0)}")

    await postgres.close()
    logger.info("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Migrate metadata from ChromaDB to PostgreSQL")
    parser.add_argument(
        "--config",
        default=str(Path.home() / ".conductor-memory" / "config.json"),
        help="Path to config file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    asyncio.run(migrate(args.config, args.dry_run))


if __name__ == "__main__":
    main()
