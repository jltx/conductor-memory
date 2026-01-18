"""
PostgreSQL Metadata Store

Fast metadata storage for dashboard operations. ChromaDB handles vectors only.

Requires optional 'postgres' dependencies:
    pip install conductor-memory[postgres]
"""

import asyncio
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    ASYNCPG_AVAILABLE = False

logger = logging.getLogger(__name__)


class PostgresMetadataStore:
    """
    Handles all file/summary metadata operations via PostgreSQL.

    Provides:
    - Sub-millisecond count queries via materialized view
    - Fast pagination with proper SQL LIMIT/OFFSET
    - Complex filtering on pattern, domain, validation status

    Thread-safe: Uses a lock to ensure only one operation at a time.
    """

    def __init__(self, connection_string: str):
        """
        Initialize the store.

        Args:
            connection_string: PostgreSQL connection URL
                e.g., "postgresql://user:pass@host:5432/conductor_memory"

        Raises:
            ImportError: If asyncpg is not installed
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "PostgreSQL support requires the 'postgres' extra. "
                "Install with: pip install conductor-memory[postgres]"
            )
        self.connection_string = connection_string
        self.pool: Optional[Any] = None  # asyncpg.Pool when available
        self._initialized = False
        self._pool_loop: Optional[asyncio.AbstractEventLoop] = None
        self._connect_lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_lock(self) -> asyncio.Lock:
        """Get or create a lock for the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create a lock anyway
            if self._connect_lock is None:
                self._connect_lock = asyncio.Lock()
            return self._connect_lock

        # Recreate lock if event loop changed
        if self._connect_lock is None or self._lock_loop is not current_loop:
            self._connect_lock = asyncio.Lock()
            self._lock_loop = current_loop
        return self._connect_lock

    async def connect(self) -> None:
        """Connect to PostgreSQL and initialize schema."""
        current_loop = asyncio.get_running_loop()

        # If pool exists but was created in a different event loop, reset it
        if self.pool is not None:
            if self._pool_loop is not current_loop:
                logger.warning("PostgreSQL pool was created in different event loop, recreating...")
                try:
                    await self.pool.close()
                except Exception:
                    pass
                self.pool = None
                self._initialized = False
            elif self._initialized:
                return

        async with self._get_lock():
            # Double-check after acquiring lock
            if self.pool is not None and self._initialized and self._pool_loop is current_loop:
                return

            logger.info(f"Connecting to PostgreSQL...")

            try:
                # Use timeout to prevent hanging on connection issues
                self.pool = await asyncio.wait_for(
                    asyncpg.create_pool(
                        self.connection_string,
                        min_size=2,
                        max_size=10,
                        command_timeout=30,
                        server_settings={'search_path': 'conductor,public'}
                    ),
                    timeout=15.0  # 15 second timeout for pool creation
                )
                self._pool_loop = current_loop

                await self._init_schema()
                self._initialized = True
                logger.info("PostgreSQL connection established")
            except asyncio.TimeoutError:
                logger.error("PostgreSQL connection timed out after 15 seconds")
                self.pool = None
                self._initialized = False
                self._pool_loop = None
                raise ConnectionError("PostgreSQL connection timed out")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                self.pool = None
                self._initialized = False
                self._pool_loop = None
                raise

    async def ensure_connected(self) -> bool:
        """Ensure connection is valid and in current event loop, reconnect if needed."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop
            return False

        # Check if pool exists and is in the current event loop
        if self.pool is None or not self._initialized or self._pool_loop is not current_loop:
            try:
                await self.connect()
            except Exception as e:
                logger.warning(f"PostgreSQL reconnection failed: {e}")
                return False
        return True

    async def reset_pool(self) -> None:
        """Reset the connection pool (call after connection errors)."""
        logger.warning("Resetting PostgreSQL connection pool...")
        try:
            if self.pool:
                await self.pool.close()
        except Exception:
            pass
        self.pool = None
        self._initialized = False
        self._pool_loop = None
        # Reconnect
        await self.connect()

    async def _init_schema(self) -> None:
        """Initialize database schema from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        schema_sql = schema_path.read_text()

        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)

        logger.info("PostgreSQL schema initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    # =========================================================================
    # Codebase Operations
    # =========================================================================

    async def upsert_codebase(
        self,
        name: str,
        path: str,
        description: Optional[str] = None,
        enabled: bool = True
    ) -> int:
        """Insert or update a codebase, return its ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO codebases (name, path, description, enabled)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (name) DO UPDATE SET
                    path = EXCLUDED.path,
                    description = EXCLUDED.description,
                    enabled = EXCLUDED.enabled
                RETURNING id
            """, name, path, description, enabled)
            return row['id']

    async def get_codebase_id(self, name: str) -> Optional[int]:
        """Get codebase ID by name."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM codebases WHERE name = $1", name
            )
            return row['id'] if row else None

    async def list_codebases(self) -> List[Dict[str, Any]]:
        """List all codebases."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, path, description, enabled FROM codebases ORDER BY name"
            )
            return [dict(row) for row in rows]

    # =========================================================================
    # Fast Count Operations (via materialized view)
    # =========================================================================

    async def get_counts(self, codebase: Optional[str] = None, _retry: bool = True) -> Dict[str, int]:
        """
        Get indexed/summarized counts instantly from materialized view.

        Returns:
            Dict with indexed_count, summarized_count, llm_count, etc.
        """
        if not await self.ensure_connected():
            raise ConnectionError("PostgreSQL not connected")

        try:
            return await self._get_counts_impl(codebase)
        except (asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.ConnectionDoesNotExistError) as e:
            if _retry:
                logger.warning(f"PostgreSQL connection error in get_counts, resetting pool: {e}")
                await self.reset_pool()
                return await self.get_counts(codebase, _retry=False)
            raise

    async def _get_counts_impl(self, codebase: Optional[str] = None) -> Dict[str, int]:
        """Internal implementation of get_counts."""
        async with self.pool.acquire() as conn:
            if codebase:
                row = await conn.fetchrow("""
                    SELECT indexed_count, summarized_count, llm_count, simple_count,
                           approved_count, rejected_count, pending_count
                    FROM codebase_stats WHERE codebase_name = $1
                """, codebase)
            else:
                row = await conn.fetchrow("""
                    SELECT
                        COALESCE(SUM(indexed_count), 0) as indexed_count,
                        COALESCE(SUM(summarized_count), 0) as summarized_count,
                        COALESCE(SUM(llm_count), 0) as llm_count,
                        COALESCE(SUM(simple_count), 0) as simple_count,
                        COALESCE(SUM(approved_count), 0) as approved_count,
                        COALESCE(SUM(rejected_count), 0) as rejected_count,
                        COALESCE(SUM(pending_count), 0) as pending_count
                    FROM codebase_stats
                """)

            if row:
                return {
                    'indexed_count': int(row['indexed_count'] or 0),
                    'summarized_count': int(row['summarized_count'] or 0),
                    'llm_count': int(row['llm_count'] or 0),
                    'simple_count': int(row['simple_count'] or 0),
                    'approved_count': int(row['approved_count'] or 0),
                    'rejected_count': int(row['rejected_count'] or 0),
                    'pending_count': int(row['pending_count'] or 0),
                }
            return {
                'indexed_count': 0, 'summarized_count': 0, 'llm_count': 0,
                'simple_count': 0, 'approved_count': 0, 'rejected_count': 0,
                'pending_count': 0
            }

    async def get_counts_by_codebase(self, _retry: bool = True) -> Dict[str, Dict[str, int]]:
        """Get counts for all codebases at once."""
        if not await self.ensure_connected():
            raise ConnectionError("PostgreSQL not connected")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT codebase_name, indexed_count, summarized_count,
                           llm_count, simple_count, approved_count
                    FROM codebase_stats
                """)
                return {
                    row['codebase_name']: {
                        'indexed_count': int(row['indexed_count'] or 0),
                        'summarized_count': int(row['summarized_count'] or 0),
                        'llm_count': int(row['llm_count'] or 0),
                        'simple_count': int(row['simple_count'] or 0),
                        'approved_count': int(row['approved_count'] or 0),
                    }
                    for row in rows
                }
        except (asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.ConnectionDoesNotExistError) as e:
            if _retry:
                logger.warning(f"PostgreSQL connection error in get_counts_by_codebase, resetting pool: {e}")
                await self.reset_pool()
                return await self.get_counts_by_codebase(_retry=False)
            raise

    async def refresh_stats(self) -> None:
        """Refresh the materialized view (call after bulk operations)."""
        async with self.pool.acquire() as conn:
            await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY codebase_stats")
        logger.debug("Refreshed codebase_stats materialized view")

    # =========================================================================
    # File Operations
    # =========================================================================

    async def upsert_file(
        self,
        codebase_name: str,
        file_path: str,
        relative_path: str,
        content_hash: Optional[str] = None,
        file_size: Optional[int] = None,
        line_count: Optional[int] = None,
        language: Optional[str] = None,
        modified_at: Optional[datetime] = None
    ) -> int:
        """Insert or update an indexed file, return its ID."""
        codebase_id = await self.get_codebase_id(codebase_name)
        if not codebase_id:
            raise ValueError(f"Codebase not found: {codebase_name}")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO indexed_files
                    (codebase_id, file_path, relative_path, content_hash,
                     file_size, line_count, language, modified_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (codebase_id, relative_path) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    content_hash = EXCLUDED.content_hash,
                    file_size = EXCLUDED.file_size,
                    line_count = EXCLUDED.line_count,
                    language = EXCLUDED.language,
                    modified_at = EXCLUDED.modified_at,
                    indexed_at = NOW()
                RETURNING id
            """, codebase_id, file_path, relative_path, content_hash,
                file_size, line_count, language, modified_at)
            return row['id']

    async def get_file_id(self, codebase_name: str, relative_path: str) -> Optional[int]:
        """Get file ID by codebase and relative path."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT f.id FROM indexed_files f
                JOIN codebases c ON f.codebase_id = c.id
                WHERE c.name = $1 AND f.relative_path = $2
            """, codebase_name, relative_path)
            return row['id'] if row else None

    async def delete_file(self, codebase_name: str, relative_path: str) -> bool:
        """Delete a file (cascades to summary)."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM indexed_files f
                USING codebases c
                WHERE f.codebase_id = c.id AND c.name = $1 AND f.relative_path = $2
            """, codebase_name, relative_path)
            return result == "DELETE 1"

    async def get_files_paginated(
        self,
        codebase: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        has_summary: Optional[bool] = None,
        pattern: Optional[str] = None,
        domain: Optional[str] = None,
        _retry: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated file list with total count.

        Returns:
            Tuple of (files list, total count)
        """
        if not await self.ensure_connected():
            raise ConnectionError("PostgreSQL not connected")

        try:
            return await self._get_files_paginated_impl(
                codebase, limit, offset, has_summary, pattern, domain
            )
        except (asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.ConnectionDoesNotExistError) as e:
            if _retry:
                logger.warning(f"PostgreSQL connection error, resetting pool and retrying: {e}")
                await self.reset_pool()
                return await self.get_files_paginated(
                    codebase, limit, offset, has_summary, pattern, domain, _retry=False
                )
            raise

    async def _get_files_paginated_impl(
        self,
        codebase: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        has_summary: Optional[bool] = None,
        pattern: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Internal implementation of get_files_paginated."""

        # Build WHERE clause
        conditions = []
        params = []
        param_idx = 1

        if codebase:
            conditions.append(f"c.name = ${param_idx}")
            params.append(codebase)
            param_idx += 1

        if has_summary is not None:
            if has_summary:
                conditions.append("s.id IS NOT NULL")
            else:
                conditions.append("s.id IS NULL")

        if pattern:
            conditions.append(f"s.pattern = ${param_idx}")
            params.append(pattern)
            param_idx += 1

        if domain:
            conditions.append(f"s.domain = ${param_idx}")
            params.append(domain)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        async with self.pool.acquire() as conn:
            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM indexed_files f
                JOIN codebases c ON f.codebase_id = c.id
                LEFT JOIN summaries s ON s.file_id = f.id
                WHERE {where_clause}
            """
            total = await conn.fetchval(count_query, *params)

            # Get paginated results
            data_query = f"""
                SELECT f.id, f.file_path, f.relative_path, f.content_hash,
                       f.file_size, f.line_count, f.language, f.indexed_at,
                       c.name as codebase_name,
                       s.id as summary_id, s.summary_type, s.pattern, s.domain,
                       s.validation_status
                FROM indexed_files f
                JOIN codebases c ON f.codebase_id = c.id
                LEFT JOIN summaries s ON s.file_id = f.id
                WHERE {where_clause}
                ORDER BY f.indexed_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([limit, offset])
            rows = await conn.fetch(data_query, *params)

            files = [
                {
                    'id': row['id'],
                    'file_path': row['file_path'],
                    'relative_path': row['relative_path'],
                    'path': row['relative_path'],  # Alias for frontend compatibility
                    'content_hash': row['content_hash'],
                    'file_size': row['file_size'],
                    'line_count': row['line_count'],
                    'language': row['language'],
                    'indexed_at': row['indexed_at'].isoformat() if row['indexed_at'] else None,
                    'codebase': row['codebase_name'],
                    'has_summary': row['summary_id'] is not None,
                    'summary_type': row['summary_type'],
                    'pattern': row['pattern'],
                    'domain': row['domain'],
                    'validation_status': row['validation_status'],
                }
                for row in rows
            ]

            return files, total

    # =========================================================================
    # Summary Operations
    # =========================================================================

    async def upsert_summary(
        self,
        file_id: int,
        summary_text: str,
        summary_type: str = 'llm',
        pattern: Optional[str] = None,
        domain: Optional[str] = None,
        key_functions: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        validation_status: str = 'pending'
    ) -> int:
        """Insert or update a summary, return its ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO summaries
                    (file_id, summary_text, summary_type, pattern, domain,
                     key_functions, dependencies, validation_status, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (file_id) DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    summary_type = EXCLUDED.summary_type,
                    pattern = EXCLUDED.pattern,
                    domain = EXCLUDED.domain,
                    key_functions = EXCLUDED.key_functions,
                    dependencies = EXCLUDED.dependencies,
                    validation_status = EXCLUDED.validation_status,
                    updated_at = NOW()
                RETURNING id
            """, file_id, summary_text, summary_type, pattern, domain,
                key_functions, dependencies, validation_status)
            return row['id']

    async def get_summary(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get summary for a file."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT s.*, f.relative_path, f.file_path, c.name as codebase_name
                FROM summaries s
                JOIN indexed_files f ON s.file_id = f.id
                JOIN codebases c ON f.codebase_id = c.id
                WHERE s.file_id = $1
            """, file_id)
            return dict(row) if row else None

    async def get_summaries_paginated(
        self,
        codebase: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        pattern: Optional[str] = None,
        domain: Optional[str] = None,
        validation_status: Optional[str] = None,
        summary_type: Optional[str] = None,
        _retry: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated summaries with filtering.

        Returns:
            Tuple of (summaries list, total count)
        """
        if not await self.ensure_connected():
            raise ConnectionError("PostgreSQL not connected")

        try:
            return await self._get_summaries_paginated_impl(
                codebase, limit, offset, pattern, domain, validation_status, summary_type
            )
        except (asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.ConnectionDoesNotExistError) as e:
            if _retry:
                logger.warning(f"PostgreSQL connection error in get_summaries_paginated, resetting pool: {e}")
                await self.reset_pool()
                return await self.get_summaries_paginated(
                    codebase, limit, offset, pattern, domain, validation_status, summary_type, _retry=False
                )
            raise

    async def _get_summaries_paginated_impl(
        self,
        codebase: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        pattern: Optional[str] = None,
        domain: Optional[str] = None,
        validation_status: Optional[str] = None,
        summary_type: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Internal implementation of get_summaries_paginated."""
        # Build WHERE clause
        conditions = ["s.id IS NOT NULL"]  # Only files with summaries
        params = []
        param_idx = 1

        if codebase:
            conditions.append(f"c.name = ${param_idx}")
            params.append(codebase)
            param_idx += 1

        if pattern:
            conditions.append(f"s.pattern = ${param_idx}")
            params.append(pattern)
            param_idx += 1

        if domain:
            conditions.append(f"s.domain = ${param_idx}")
            params.append(domain)
            param_idx += 1

        if validation_status:
            conditions.append(f"s.validation_status = ${param_idx}")
            params.append(validation_status)
            param_idx += 1

        if summary_type:
            conditions.append(f"s.summary_type = ${param_idx}")
            params.append(summary_type)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        async with self.pool.acquire() as conn:
            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM summaries s
                JOIN indexed_files f ON s.file_id = f.id
                JOIN codebases c ON f.codebase_id = c.id
                WHERE {where_clause}
            """
            total = await conn.fetchval(count_query, *params)

            # Get paginated results
            data_query = f"""
                SELECT s.id, s.file_id, s.summary_text, s.summary_type,
                       s.pattern, s.domain, s.key_functions, s.dependencies,
                       s.validation_status, s.created_at, s.updated_at,
                       f.relative_path, f.file_path, f.language, f.line_count,
                       c.name as codebase_name
                FROM summaries s
                JOIN indexed_files f ON s.file_id = f.id
                JOIN codebases c ON f.codebase_id = c.id
                WHERE {where_clause}
                ORDER BY s.updated_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([limit, offset])
            rows = await conn.fetch(data_query, *params)

            summaries = [
                {
                    'id': row['id'],
                    'file_id': row['file_id'],
                    'file_path': row['file_path'],
                    'relative_path': row['relative_path'],
                    'path': row['relative_path'],  # Alias for frontend compatibility
                    'codebase': row['codebase_name'],
                    'language': row['language'],
                    'line_count': row['line_count'],
                    'summary_text': row['summary_text'],
                    'summary_type': row['summary_type'],
                    'pattern': row['pattern'],
                    'domain': row['domain'],
                    'key_functions': list(row['key_functions']) if row['key_functions'] else [],
                    'dependencies': list(row['dependencies']) if row['dependencies'] else [],
                    'validation_status': row['validation_status'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
                }
                for row in rows
            ]

            return summaries, total

    async def update_validation_status(
        self,
        file_id: int,
        status: str,
        validated_by: Optional[str] = None
    ) -> bool:
        """Update validation status for a summary."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE summaries SET
                    validation_status = $2,
                    validated_by = $3,
                    validated_at = NOW()
                WHERE file_id = $1
            """, file_id, status, validated_by)
            return result == "UPDATE 1"

    async def get_summary_stats(self, codebase: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated summary statistics by pattern and domain.
        """
        async with self.pool.acquire() as conn:
            # Pattern breakdown
            if codebase:
                pattern_rows = await conn.fetch("""
                    SELECT s.pattern, COUNT(*) as count
                    FROM summaries s
                    JOIN indexed_files f ON s.file_id = f.id
                    JOIN codebases c ON f.codebase_id = c.id
                    WHERE c.name = $1 AND s.pattern IS NOT NULL
                    GROUP BY s.pattern
                    ORDER BY count DESC
                """, codebase)
                domain_rows = await conn.fetch("""
                    SELECT s.domain, COUNT(*) as count
                    FROM summaries s
                    JOIN indexed_files f ON s.file_id = f.id
                    JOIN codebases c ON f.codebase_id = c.id
                    WHERE c.name = $1 AND s.domain IS NOT NULL
                    GROUP BY s.domain
                    ORDER BY count DESC
                """, codebase)
            else:
                pattern_rows = await conn.fetch("""
                    SELECT pattern, COUNT(*) as count
                    FROM summaries
                    WHERE pattern IS NOT NULL
                    GROUP BY pattern
                    ORDER BY count DESC
                """)
                domain_rows = await conn.fetch("""
                    SELECT domain, COUNT(*) as count
                    FROM summaries
                    WHERE domain IS NOT NULL
                    GROUP BY domain
                    ORDER BY count DESC
                """)

            return {
                'by_pattern': {row['pattern']: row['count'] for row in pattern_rows},
                'by_domain': {row['domain']: row['count'] for row in domain_rows},
            }

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def bulk_upsert_files(
        self,
        codebase_name: str,
        files: List[Dict[str, Any]]
    ) -> int:
        """Bulk insert/update files. Returns count of upserted files."""
        codebase_id = await self.get_codebase_id(codebase_name)
        if not codebase_id:
            raise ValueError(f"Codebase not found: {codebase_name}")

        async with self.pool.acquire() as conn:
            # Use COPY for bulk insert
            count = 0
            for file_data in files:
                await conn.execute("""
                    INSERT INTO indexed_files
                        (codebase_id, file_path, relative_path, content_hash,
                         file_size, line_count, language, modified_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (codebase_id, relative_path) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        file_size = EXCLUDED.file_size,
                        line_count = EXCLUDED.line_count,
                        language = EXCLUDED.language,
                        modified_at = EXCLUDED.modified_at,
                        indexed_at = NOW()
                """, codebase_id,
                    file_data.get('file_path'),
                    file_data.get('relative_path'),
                    file_data.get('content_hash'),
                    file_data.get('file_size'),
                    file_data.get('line_count'),
                    file_data.get('language'),
                    file_data.get('modified_at'))
                count += 1

            return count

    async def delete_missing_files(
        self,
        codebase_name: str,
        current_paths: List[str]
    ) -> int:
        """Delete files that are no longer present. Returns count deleted."""
        codebase_id = await self.get_codebase_id(codebase_name)
        if not codebase_id:
            return 0

        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM indexed_files
                WHERE codebase_id = $1 AND relative_path != ALL($2::text[])
            """, codebase_id, current_paths)

            # Parse "DELETE X" to get count
            try:
                return int(result.split()[1])
            except (IndexError, ValueError):
                return 0
