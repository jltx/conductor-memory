"""
MemoryService - Core memory service with multi-codebase support.

This is the single source of truth for all memory/indexing logic.
Both the HTTP server and stdio MCP server are thin wrappers around this service.
"""

import asyncio
import concurrent.futures
import logging
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..config.server import ServerConfig, CodebaseConfig
from ..config.summarization import SummarizationConfig
from ..storage.chroma import ChromaVectorStore
from ..search.chunking import ChunkingManager, ChunkingStrategy
from ..search.boosting import BoostCalculator
from ..search.heuristics import HeuristicExtractor
from ..search.import_graph import ImportGraph
from ..core.models import MemoryChunk, RoleEnum, MemoryType
from ..core.resources import detect_resources, get_optimal_device, SystemResources
from ..embedding.sentence_transformer import SentenceTransformerEmbedder
from ..search.hybrid import HybridSearcher, SearchMode
from ..search.call_graph import MethodCallGraph, CallGraphBuilder
from ..search.verification import (
    parse_verification_query,
    find_evidence,
    extract_key_terms,
    VerificationResult,
    VerificationInfo,
    VerificationStatus,
    SubjectInfo,
    Evidence,
)
from ..llm.summarizer import FileSummarizer, FileSummary
from ..storage.postgres import PostgresMetadataStore

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """
    Run an async coroutine synchronously, handling the case where
    an event loop may or may not already be running.
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - run in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run directly
        return asyncio.run(coro)


class MemoryService:
    """
    Core memory service with multi-codebase support.
    
    Provides both sync and async versions of all public methods:
    - search() / search_async()
    - store() / store_async()
    - initialize() / initialize_async()
    
    The sync versions use asyncio.run() internally and are suitable for
    the stdio MCP server. The async versions should be used by the HTTP server.
    """
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the MemoryService.
        
        Args:
            config: ServerConfig with codebase definitions and settings
        """
        self.config = config
        self.persist_directory = config.persist_directory
        
        # Vector stores for each codebase (one collection per codebase)
        self._vector_stores: Dict[str, ChromaVectorStore] = {}
        
        # Indexing status per codebase
        self._indexing_status: Dict[str, Dict[str, Any]] = {}
        
        # File watcher tasks (one per codebase)
        self._file_watcher_tasks: Dict[str, asyncio.Task] = {}
        self._watch_interval = config.watch_interval
        
        # Detect system resources
        self._resources = detect_resources()
        
        # Determine device to use
        if config.device == "auto":
            self._device = get_optimal_device()
        else:
            self._device = config.device
        
        # Determine embedding batch size
        if config.embedding_batch_size == "auto":
            self._embedding_batch_size = self._resources.recommended_embedding_batch_size
        else:
            try:
                self._embedding_batch_size = int(config.embedding_batch_size)
            except ValueError:
                logger.warning(f"Invalid embedding_batch_size '{config.embedding_batch_size}', using auto")
                self._embedding_batch_size = self._resources.recommended_embedding_batch_size
        
        logger.info(f"Using device: {self._device}, embedding batch size: {self._embedding_batch_size}")
        
        # Shared embedder and chunking manager
        self.embedder = SentenceTransformerEmbedder(
            model_name=config.embedding_model,
            device=self._device
        )
        self.chunking_manager = ChunkingManager(ChunkingStrategy.FUNCTION_CLASS)
        
        # Hybrid search support
        self.hybrid_searcher = HybridSearcher(rrf_k=60, semantic_weight=0.5)
        
        # Boost calculator for relevance score adjustments
        self.boost_calculator = BoostCalculator(config.boost_config)
        
        # Phase 1: Heuristic extraction and import graph analysis
        self.heuristic_extractor = HeuristicExtractor()
        self._import_graphs: Dict[str, ImportGraph] = {}  # One per codebase
        
        # Phase 4: Method call graph for tracking method-to-method relationships
        self._call_graphs: Dict[str, MethodCallGraph] = {}  # One per codebase
        self._call_graph_builders: Dict[str, CallGraphBuilder] = {}  # Builders for incremental construction
        
        # Phase 4: Background summarization
        self._summarization_config = config.summarization_config
        self._summarizer: Optional[FileSummarizer] = None
        self._summarizer_task: Optional[asyncio.Task] = None
        self._summarizer_thread: Optional[threading.Thread] = None
        self._summarizer_stop_event = threading.Event()
        self._summary_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        # Thread-safe queue for the background thread
        self._summary_queue_threaded: queue.PriorityQueue = queue.PriorityQueue()
        self._query_active: asyncio.Event = asyncio.Event()
        self._summarization_stats = {
            "files_summarized": 0,
            "files_queued": 0,
            "files_total_queued": 0,        # Total files originally queued (for progress calculation)
            "files_failed": 0,
            "files_skipped": 0,
            "files_skipped_pattern": 0,      # Pattern match - test files, build outputs, etc.
            "files_skipped_unchanged": 0,   # Already summarized, content hash matches
            "files_skipped_empty": 0,       # Missing or empty files
            "current_file": None,
            "is_running": False,
            "last_error": None,
            "total_time_seconds": 0.0,
            "avg_time_per_file": 0.0,
            "estimated_time_remaining": 0.0
        }
        
        # Indexing completion tracking for callback-based summarizer startup
        self._indexing_completion_callbacks = []
        self._completed_codebases = set()

        # In-memory count cache (avoids slow ChromaDB count() queries)
        # Structure: {codebase_name: {"indexed": int, "summarized": int}}
        self._count_cache: Dict[str, Dict[str, int]] = {}
        self._count_cache_initialized = False

        # PostgreSQL metadata store (optional, for fast dashboard queries)
        self._postgres: Optional[PostgresMetadataStore] = None
        self._postgres_initialized = False
        if config.postgres_url:
            self._postgres = PostgresMetadataStore(config.postgres_url)
            logger.info(f"PostgreSQL metadata store configured")

        # Initialize vector stores for each enabled codebase
        for codebase in config.get_enabled_codebases():
            self._init_codebase(codebase)
        
        logger.info(f"MemoryService initialized with {len(self._vector_stores)} codebase(s)")
        logger.info(f"Persistent storage at: {self.persist_directory}")
    
    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking function in a thread pool executor.

        Use this to wrap synchronous I/O operations (ChromaDB, embeddings)
        to prevent blocking the async event loop.
        """
        loop = asyncio.get_event_loop()
        if kwargs:
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        return await loop.run_in_executor(None, func, *args)

    async def _ensure_postgres(self) -> bool:
        """
        Lazily ensure PostgreSQL is connected.

        Allows retry after startup failure - called on each API request.
        Returns True if PostgreSQL is available, False otherwise.
        """
        logger.info("_ensure_postgres called")

        # Already initialized and working
        if self._postgres_initialized and self._postgres:
            logger.debug("_ensure_postgres: already initialized")
            return True

        # No PostgreSQL URL configured
        if not self.config.postgres_url:
            logger.warning("_ensure_postgres: no postgres_url configured")
            return False

        logger.info("_ensure_postgres: attempting connection...")

        # Recreate PostgreSQL store if it was set to None after failure
        if self._postgres is None:
            logger.info("_ensure_postgres: recreating PostgresMetadataStore")
            self._postgres = PostgresMetadataStore(self.config.postgres_url)

        try:
            logger.info("_ensure_postgres: calling connect()...")
            await self._postgres.connect()
            self._postgres_initialized = True
            logger.info("PostgreSQL metadata store connected (lazy init)")

            # Ensure codebases exist in PostgreSQL
            for codebase in self.config.get_enabled_codebases():
                await self._postgres.upsert_codebase(
                    name=codebase.name,
                    path=codebase.path,
                    description=codebase.description,
                    enabled=codebase.enabled
                )
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL lazy init failed: {e}")
            return False
    
    def _init_codebase(self, codebase: CodebaseConfig) -> None:
        """Initialize vector store and status for a codebase"""
        
        # Determine ChromaDB connection mode
        if self.config.chroma_mode == "http":
            # HTTP client mode - connect to standalone ChromaDB server
            self._vector_stores[codebase.name] = ChromaVectorStore(
                collection_name=f"codebase_{codebase.name}",
                host=self.config.chroma_host,
                port=self.config.chroma_port
            )
            logger.info(f"[{codebase.name}] Using ChromaDB HTTP client at {self.config.chroma_host}:{self.config.chroma_port}")
        else:
            # Embedded mode - use PersistentClient (current behavior)
            self._vector_stores[codebase.name] = ChromaVectorStore(
                collection_name=f"codebase_{codebase.name}",
                persist_directory=self.persist_directory
            )
        self._indexing_status[codebase.name] = {
            "name": codebase.name,
            "path": codebase.path,
            "status": "not_started",
            "progress": 0.0,
            "files_processed": 0,
            "total_files": 0,
            "indexed_files_count": 0,
            "current_file": None,
            "error_message": None
        }
        # Initialize import graph for this codebase
        self._import_graphs[codebase.name] = ImportGraph()
        
        # Initialize call graph and builder for this codebase
        self._call_graphs[codebase.name] = MethodCallGraph()
        self._call_graph_builders[codebase.name] = CallGraphBuilder(self._call_graphs[codebase.name])
    
    def _register_indexing_completion_callback(self, callback):
        """Register a callback to be called when all indexing is complete"""
        self._indexing_completion_callbacks.append(callback)
    
    def _on_codebase_indexing_complete(self, codebase_name: str):
        """Called when a codebase finishes indexing"""
        self._completed_codebases.add(codebase_name)
        logger.debug(f"[Indexing] Codebase {codebase_name} completed. Completed: {self._completed_codebases}")
        
        # Check if all enabled codebases are complete
        enabled_codebase_names = {cb.name for cb in self.config.get_enabled_codebases()}
        if self._completed_codebases >= enabled_codebase_names:
            logger.info(f"[Indexing] All codebases completed ({len(self._completed_codebases)}/{len(enabled_codebase_names)}), triggering {len(self._indexing_completion_callbacks)} completion callbacks")
            # Trigger all registered callbacks
            for callback in self._indexing_completion_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback())
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"Error in indexing completion callback: {e}")
            
            # Clear callbacks after triggering
            self._indexing_completion_callbacks.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Public API - Async versions
    # ─────────────────────────────────────────────────────────────────────────
    
    async def initialize_async(self, wait_for_indexing: bool = False) -> None:
        """
        Start indexing and file watchers for all codebases.

        Args:
            wait_for_indexing: If True, wait for all indexing to complete before returning.
                              If False, start indexing in background tasks.
        """
        # Initialize PostgreSQL connection if configured
        if self._postgres and not self._postgres_initialized:
            try:
                await self._postgres.connect()
                self._postgres_initialized = True
                logger.info("PostgreSQL metadata store connected")

                # Ensure codebases exist in PostgreSQL
                for codebase in self.config.get_enabled_codebases():
                    await self._postgres.upsert_codebase(
                        name=codebase.name,
                        path=codebase.path,
                        description=codebase.description,
                        enabled=codebase.enabled
                    )
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                logger.warning("Falling back to ChromaDB-only mode")
                self._postgres = None

        enabled_codebases = self.config.get_enabled_codebases()
        
        if not enabled_codebases:
            logger.info("No codebases configured, skipping indexing")
            return
        
        indexing_tasks = []
        
        # Start indexing for each codebase
        for codebase in enabled_codebases:
            codebase_path = Path(codebase.path)
            if not codebase_path.exists():
                logger.warning(f"[{codebase.name}] Codebase path does not exist: {codebase.path}")
                self._indexing_status[codebase.name]["status"] = "error"
                self._indexing_status[codebase.name]["error_message"] = "Path does not exist"
                continue
            
            vector_store = self._vector_stores[codebase.name]
            indexed_files = vector_store.get_indexed_files()
            
            if indexed_files:
                logger.info(f"[{codebase.name}] Found existing index with {len(indexed_files)} files")
            else:
                logger.info(f"[{codebase.name}] No existing index, will perform full indexing")
            
            if wait_for_indexing:
                # Add to list of tasks to await
                indexing_tasks.append((codebase, self._index_codebase(codebase)))
            else:
                # Start as background task
                asyncio.create_task(self._index_codebase(codebase))
                # Start file watcher immediately for background indexing
                if self.config.enable_file_watcher:
                    self._file_watcher_tasks[codebase.name] = asyncio.create_task(
                        self._watch_for_changes(codebase)
                    )
        
        # Wait for all indexing to complete if requested (sequentially for clean progress output)
        if wait_for_indexing and indexing_tasks:
            for codebase, task in indexing_tasks:
                await task
            
            # Note: File watchers and background summarizer are NOT started here 
            # because when called via _run_sync(), we're in a temporary event loop 
            # that will close. They should be started via start_file_watchers() 
            # and start_background_summarizer() in the main server's event loop.
        else:
            # For background indexing mode, start summarizer when indexing completes
            async def _on_indexing_complete():
                logger.info("[Summarization] All indexing completed, starting background summarizer")
                await self._start_background_summarizer()
            
            # Register callback to start summarizer when indexing is done
            self._register_indexing_completion_callback(_on_indexing_complete)
            
            # Check if indexing is already complete (e.g., all codebases already indexed)
            enabled_codebase_names = {cb.name for cb in self.config.get_enabled_codebases()}
            already_completed = {
                name for name, status in self._indexing_status.items() 
                if status.get("status") == "completed"
            }
            
            if already_completed >= enabled_codebase_names:
                logger.info("[Summarization] Indexing already complete, starting background summarizer immediately")
                asyncio.create_task(_on_indexing_complete())
            
            # Also add a fallback timeout in case callback system fails
            async def _fallback_summarizer_start():
                await asyncio.sleep(self._summarization_config.startup_delay_seconds)
                
                # Check if summarizer already started via callback
                if self._summarizer_task is None:
                    logger.warning("[Summarization] Fallback timeout reached, starting background summarizer anyway")
                    await self._start_background_summarizer()
            
            asyncio.create_task(_fallback_summarizer_start())
    
    async def search_async(
        self,
        query: str,
        codebase: Optional[str] = None,
        max_results: int = 10,
        project_id: Optional[str] = None,
        search_mode: str = "auto",
        # Phase 1: Boosting
        domain_boosts: Optional[Dict[str, float]] = None,
        # Phase 2: Tag Filtering
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        # Phase 2: Heuristic Filtering
        languages: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        function_names: Optional[List[str]] = None,
        annotations: Optional[List[str]] = None,
        has_annotations: Optional[bool] = None,
        has_docstrings: Optional[bool] = None,
        min_class_count: Optional[int] = None,
        min_function_count: Optional[int] = None,
        # Phase 1: Implementation Signal Filtering
        calls: Optional[List[str]] = None,
        accesses: Optional[List[str]] = None,
        subscripts: Optional[List[str]] = None,
        # Phase 5: Summary integration
        include_summaries: bool = False,
        boost_summarized: bool = True
    ) -> Dict[str, Any]:
        """
        Search for relevant memories using semantic similarity, keyword matching, or both.
        
        Args:
            query: Search query text
            codebase: Optional codebase name to search (None = search all)
            max_results: Maximum number of results to return
            project_id: Optional filter by project ID
            search_mode: "semantic", "keyword", "hybrid", "auto" (default), or "verify"
            domain_boosts: Per-query domain boost overrides (e.g., {'class': 1.5, 'test': 0.5})
            include_tags: Include only results matching these tags (supports prefix:* patterns)
            exclude_tags: Exclude results matching these tags (supports prefix:* patterns)
            languages: Filter by programming languages (e.g., ['python', 'java'])
            class_names: Filter by class names (e.g., ['UserService', 'TestClass'])
            function_names: Filter by function names (e.g., ['process_data', 'validate'])
            annotations: Filter by annotations (e.g., ['@Test', '@Component'])
            has_annotations: Filter files that have/don't have annotations
            has_docstrings: Filter files that have/don't have docstrings
            min_class_count: Minimum number of classes in file
            min_function_count: Minimum number of functions in file
            calls: Filter by method calls (matches calls:* tags, e.g., ['iloc', 'fit'])
            accesses: Filter by attribute access (matches reads:* tags, e.g., ['bar_index'])
            subscripts: Filter by subscript patterns (matches subscript:* tags, e.g., ['iloc'])
            include_summaries: Include LLM-generated summaries with results (default False)
            boost_summarized: Apply boost to results that have LLM summaries (default True)
            
        Returns:
            Dict with results, total_found, query_time_ms, summary, and search_mode_used
        """
        start_time = datetime.now()
        
        # Signal that a query is active - background summarizer will yield
        self._query_active.set()
        
        try:
            # Handle verification search mode (Phase 3)
            if search_mode == "verify":
                return await self._verify_search(
                    query=query,
                    codebase=codebase,
                    max_results=max_results,
                    include_summaries=include_summaries,
                )
            
            # Determine search mode
            mode = SearchMode(search_mode) if search_mode in [m.value for m in SearchMode] else SearchMode.AUTO
            if mode == SearchMode.AUTO:
                mode = self.hybrid_searcher.detect_search_mode(query)
            
            similar_chunks = []
            mode_used = mode.value
            
            if mode == SearchMode.KEYWORD:
                # Keyword-only search using BM25
                keyword_results = self.hybrid_searcher.keyword_search(query, codebase, max_results * 2)
                similar_chunks = [chunk for chunk, _ in keyword_results]
                # Set relevance scores from BM25
                if keyword_results:
                    max_score = max(score for _, score in keyword_results) or 1.0
                    for chunk, score in keyword_results:
                        chunk.relevance_score = score / max_score
                
            elif mode == SearchMode.SEMANTIC:
                # Semantic-only search using vectors
                query_embedding = await self._run_blocking(self.embedder.generate, query)
                
                if codebase:
                    vector_store = self._vector_stores.get(codebase)
                    if not vector_store:
                        return {
                            "error": f"Codebase not found: {codebase}",
                            "results": [],
                            "total_found": 0
                        }
                    similar_chunks = await self._run_blocking(vector_store.search, query_embedding, max_results * 2)
                else:
                    for codebase_name, vector_store in self._vector_stores.items():
                        chunks = await self._run_blocking(vector_store.search, query_embedding, max_results)
                        similar_chunks.extend(chunks)
                
            else:  # HYBRID
                # Combined semantic + keyword search with RRF fusion
                query_embedding = await self._run_blocking(self.embedder.generate, query)
                
                # Get semantic results
                semantic_chunks = []
                if codebase:
                    vector_store = self._vector_stores.get(codebase)
                    if not vector_store:
                        return {
                            "error": f"Codebase not found: {codebase}",
                            "results": [],
                            "total_found": 0
                        }
                    semantic_chunks = await self._run_blocking(vector_store.search, query_embedding, max_results * 2)
                else:
                    for codebase_name, vector_store in self._vector_stores.items():
                        chunks = await self._run_blocking(vector_store.search, query_embedding, max_results)
                        semantic_chunks.extend(chunks)
                
                # Get hybrid results with RRF fusion
                hybrid_results = self.hybrid_searcher.hybrid_search(
                    query=query,
                    semantic_results=semantic_chunks,
                    codebase=codebase,
                    top_k=max_results * 2
                )
                
                # Extract chunks and set combined scores
                for hr in hybrid_results:
                    hr.chunk.relevance_score = hr.combined_score
                    similar_chunks.append(hr.chunk)
            
            # Filter by project_id if specified
            if project_id:
                similar_chunks = [c for c in similar_chunks if c.project_id == project_id]
            
            # Apply boosting to adjust relevance scores
            if similar_chunks:
                similar_chunks = self.boost_calculator.apply_boosts_to_chunks(
                    similar_chunks, 
                    query_domain_boosts=domain_boosts
                )
                # Re-sort by updated relevance scores
                # Use chunk ID as secondary key for deterministic ordering when scores are tied
                similar_chunks.sort(key=lambda x: (x.relevance_score, x.id), reverse=True)
            
            # Apply tag filtering (Phase 2)
            if include_tags or exclude_tags:
                similar_chunks = self._filter_by_tags(similar_chunks, include_tags, exclude_tags)
            
            # Apply heuristic filtering (Phase 2)
            if any([languages, class_names, function_names, annotations, has_annotations is not None, 
                   has_docstrings is not None, min_class_count is not None, min_function_count is not None]):
                similar_chunks = self._filter_by_heuristics(
                    similar_chunks, languages, class_names, function_names, annotations,
                    has_annotations, has_docstrings, min_class_count, min_function_count
                )
            
            # Apply implementation signal filtering (Phase 1)
            if any([calls, accesses, subscripts]):
                similar_chunks = self._filter_by_implementation_signals(
                    similar_chunks, calls, accesses, subscripts
                )
            
            # Remove duplicates based on content similarity
            deduplicated_chunks = self._deduplicate_chunks(similar_chunks)
            
            # Phase 5: Apply summary boost if enabled
            # This boosts results for files that have LLM-generated summaries
            if boost_summarized and deduplicated_chunks:
                deduplicated_chunks = await self._apply_summary_boost_async(
                    deduplicated_chunks, codebase
                )
                # Re-sort after summary boost
                # Use chunk ID as secondary key for deterministic ordering when scores are tied
                deduplicated_chunks.sort(key=lambda x: (x.relevance_score, x.id), reverse=True)
            
            # Limit results
            filtered_chunks = deduplicated_chunks[:max_results]
            
            # Phase 5: Lookup summaries if requested
            file_summaries = {}
            if include_summaries:
                file_summaries = await self._get_file_summaries_async(filtered_chunks, codebase)
            
            # Format results
            results = []
            for chunk in filtered_chunks:
                result = {
                    "id": chunk.id,
                    "project_id": chunk.project_id,
                    "role": chunk.role.value if hasattr(chunk.role, 'value') else str(chunk.role),
                    "content": chunk.doc_text[:500] + "..." if len(chunk.doc_text) > 500 else chunk.doc_text,
                    "tags": chunk.tags,
                    "source": chunk.source,
                    "relevance_score": chunk.relevance_score
                }
                
                # Add summary if available and requested
                if include_summaries:
                    file_path = self._extract_file_path_from_chunk(chunk)
                    if file_path and file_path in file_summaries:
                        result["file_summary"] = file_summaries[file_path]
                        result["has_summary"] = True
                    else:
                        result["has_summary"] = False
                
                results.append(result)
            
            # Generate summary
            summary = self._generate_context_summary(filtered_chunks, query)
            
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "results": results,
                "total_found": len(similar_chunks),
                "query_time_ms": query_time,
                "summary": summary,
                "search_mode_used": mode_used
            }
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {
                "error": str(e),
                "results": [],
                "total_found": 0
            }
        finally:
            # Clear query active flag - allow background summarizer to resume
            self._query_active.clear()
    
    async def _verify_search(
        self,
        query: str,
        codebase: Optional[str] = None,
        max_results: int = 10,
        include_summaries: bool = False,
    ) -> Dict[str, Any]:
        """
        Handle verification search mode (Phase 3).
        
        Parses "verify X uses Y" style queries and checks implementation signals
        for evidence supporting or contradicting the claim.
        
        Args:
            query: The verification query (e.g., "verify _generate_features uses iloc")
            codebase: Optional codebase to search in
            max_results: Maximum results for subject search
            include_summaries: Include file summaries in results
            
        Returns:
            Dict with verification result (from VerificationResult.to_dict())
            Falls back to regular search if query cannot be parsed as verification
        """
        start_time = datetime.now()
        
        # Parse the verification query
        intent = parse_verification_query(query)
        
        if not intent:
            # Fall back to regular search if not a verification query
            logger.debug(f"[Verify] Query not parseable as verification, falling back: {query}")
            # Remove "verify" prefix if present and do regular search
            fallback_query = query
            if query.lower().startswith("verify "):
                fallback_query = query[7:].strip()
            
            return await self.search_async(
                query=fallback_query,
                codebase=codebase,
                max_results=max_results,
                search_mode="auto",
                include_summaries=include_summaries,
            )
        
        logger.debug(f"[Verify] Parsed query - subject: '{intent.subject}', claim: '{intent.claim}'")
        
        # Search for the subject using existing search
        subject_search = await self.search_async(
            query=intent.subject,
            codebase=codebase,
            max_results=max_results,
            search_mode="auto",  # Use auto mode for subject search
        )
        
        # Check if we found the subject
        results = subject_search.get("results", [])
        if not results:
            # Subject not found
            result = VerificationResult.subject_not_found(intent.subject, intent.claim)
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            result_dict = result.to_dict()
            result_dict["query_time_ms"] = query_time
            return result_dict
        
        # Find evidence across all matching chunks
        all_evidence: List[Evidence] = []
        best_file = None
        best_line = None
        subject_type = None
        
        for chunk_result in results:
            chunk_tags = chunk_result.get("tags", [])
            chunk_content = chunk_result.get("content", "")
            
            # Extract file path from tags
            file_path = None
            for tag in chunk_tags:
                if tag.startswith("file:"):
                    file_path = tag[5:]
                    if best_file is None:
                        best_file = file_path
                    break
            
            # Extract line info if available
            for tag in chunk_tags:
                if tag.startswith("lines:"):
                    try:
                        line_range = tag[6:]
                        best_line = int(line_range.split("-")[0])
                    except (ValueError, IndexError):
                        pass
                    break
            
            # Detect subject type from tags
            for tag in chunk_tags:
                if tag.startswith("domain:"):
                    domain = tag[7:]
                    if domain in ("function", "method"):
                        subject_type = "method"
                    elif domain == "class_summary":
                        subject_type = "class"
                    break
            
            # Find evidence in this chunk
            chunk_evidence = find_evidence(chunk_tags, chunk_content, intent.claim)
            all_evidence.extend(chunk_evidence)
        
        # Deduplicate evidence by (type, detail) and keep highest relevance
        seen_evidence: Dict[tuple, Evidence] = {}
        for ev in all_evidence:
            key = (ev.type, ev.detail)
            if key not in seen_evidence or ev.relevance > seen_evidence[key].relevance:
                seen_evidence[key] = ev
        
        # Sort by relevance, with detail as tiebreaker for deterministic ordering
        deduplicated_evidence = sorted(
            seen_evidence.values(),
            key=lambda e: (e.relevance, e.detail),
            reverse=True
        )
        
        # Determine verification status and confidence
        if deduplicated_evidence:
            # Calculate confidence based on evidence strength
            max_relevance = max(e.relevance for e in deduplicated_evidence)
            avg_relevance = sum(e.relevance for e in deduplicated_evidence) / len(deduplicated_evidence)
            confidence = min((max_relevance + avg_relevance) / 2 + 0.1, 1.0)
            
            status = VerificationStatus.SUPPORTED
            summary = f"VERIFIED: '{intent.subject}' {intent.claim} - found {len(deduplicated_evidence)} evidence item(s)."
        else:
            status = VerificationStatus.NOT_SUPPORTED
            confidence = 0.7  # We found the subject but no evidence
            summary = f"NOT VERIFIED: Found '{intent.subject}' but no evidence that it {intent.claim}."
        
        # Build the result
        result = VerificationResult(
            subject=SubjectInfo(
                name=intent.subject,
                file=best_file,
                found=True,
                line=best_line,
                type=subject_type,
            ),
            verification=VerificationInfo(
                status=status,
                confidence=confidence,
                evidence=deduplicated_evidence[:10],  # Limit to top 10 evidence items
            ),
            summary=summary,
        )
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        result_dict = result.to_dict()
        result_dict["query_time_ms"] = query_time
        
        return result_dict
    
    async def store_async(
        self,
        content: str,
        project_id: str = "default",
        codebase: Optional[str] = None,
        role: str = "user",
        tags: Optional[List[str]] = None,
        pin: bool = False,
        source: str = "api",
        memory_type: str = "conversation"
    ) -> Dict[str, Any]:
        """
        Store a new memory chunk.
        
        Args:
            content: Text content to store
            project_id: Project identifier
            codebase: Codebase to store in (default: first configured)
            role: Role - user, assistant, system, tool
            tags: Optional list of tags
            pin: Pin to prevent pruning
            source: Source of the memory
            memory_type: Type of memory - code, conversation, decision, lesson
            
        Returns:
            Dict with success status, id, and metadata
        """
        if not self._vector_stores:
            return {"error": "No codebases configured", "success": False}
        
        try:
            # Determine which vector store to use
            if codebase:
                vector_store = self._vector_stores.get(codebase)
                if not vector_store:
                    return {"error": f"Codebase not found: {codebase}", "success": False}
            else:
                # Use first available
                codebase = list(self._vector_stores.keys())[0]
                vector_store = self._vector_stores[codebase]
            
            # Map role string to enum
            role_map = {
                "user": RoleEnum.USER,
                "assistant": RoleEnum.ASSISTANT,
                "system": RoleEnum.SYSTEM,
                "tool": RoleEnum.TOOL
            }
            role_enum = role_map.get(role.lower(), RoleEnum.USER)
            
            # Map memory_type string to enum
            memory_type_map = {
                "code": MemoryType.CODE,
                "conversation": MemoryType.CONVERSATION,
                "decision": MemoryType.DECISION,
                "lesson": MemoryType.LESSON
            }
            memory_type_enum = memory_type_map.get(memory_type.lower(), MemoryType.CONVERSATION)
            
            all_tags = list(tags or [])
            all_tags.append(f"codebase:{codebase}")
            
            # Auto-pin decisions and lessons (they're valuable context)
            should_pin = pin or memory_type_enum in (MemoryType.DECISION, MemoryType.LESSON)
            
            memory_chunk = MemoryChunk(
                id=str(uuid.uuid4()),
                project_id=project_id,
                role=role_enum,
                prompt="",
                response="",
                doc_text=content,
                embedding_id="",
                tags=all_tags,
                pin=should_pin,
                relevance_score=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source=source,
                memory_type=memory_type_enum
            )
            
            # Generate embedding and store (non-blocking)
            embedding = await self._run_blocking(self.embedder.generate, content)
            await self._run_blocking(vector_store.add, memory_chunk, embedding)
            
            # Also add to BM25 index for hybrid/keyword search
            self.hybrid_searcher.add_to_index(codebase, memory_chunk)
            
            logger.info(f"Stored memory chunk {memory_chunk.id} in codebase {codebase}")
            
            return {
                "success": True,
                "id": memory_chunk.id,
                "project_id": project_id,
                "codebase": codebase,
                "tags": all_tags
            }
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {"error": str(e), "success": False}
    
    async def prune_async(
        self,
        project_id: Optional[str] = None,
        max_age_days: int = 30,
        codebase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prune obsolete memories based on age and relevance.
        
        This will NOT prune:
        - Pinned memories (decisions, lessons)
        - Code chunks from codebase indexing
        - Memories younger than max_age_days
        
        Args:
            project_id: Optional project ID to filter pruning
            max_age_days: Maximum age in days for memories (default 30)
            codebase: Optional codebase to prune from
            
        Returns:
            Dict with pruning results
        """
        from datetime import timedelta
        
        if not self._vector_stores:
            return {"error": "No codebases configured", "pruned": 0, "kept": 0, "total_processed": 0}
        
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            total_pruned = 0
            total_kept = 0
            total_processed = 0
            
            # Determine which vector stores to process
            if codebase:
                stores_to_process = {codebase: self._vector_stores.get(codebase)}
                if not stores_to_process[codebase]:
                    return {"error": f"Codebase not found: {codebase}", "pruned": 0, "kept": 0, "total_processed": 0}
            else:
                stores_to_process = self._vector_stores
            
            for codebase_name, vector_store in stores_to_process.items():
                # Get all memories from this codebase
                # We need to query without filters to get everything
                # Use a dummy query since Chroma requires one
                all_results = vector_store.collection.get(
                    include=["metadatas", "documents"]
                )
                
                if not all_results["ids"]:
                    continue
                
                ids_to_prune = []
                
                for i, chunk_id in enumerate(all_results["ids"]):
                    metadata = all_results["metadatas"][i] if all_results["metadatas"] else {}
                    total_processed += 1
                    
                    # Skip code chunks from codebase indexing
                    source = metadata.get("source", "")
                    if source == "codebase_indexing":
                        total_kept += 1
                        continue
                    
                    # Skip pinned memories (decisions, lessons)
                    memory_type = metadata.get("memory_type", "")
                    if memory_type in ("decision", "lesson"):
                        total_kept += 1
                        continue
                    
                    # Filter by project_id if specified
                    chunk_project = metadata.get("project_id", "")
                    if project_id and chunk_project != project_id:
                        total_kept += 1
                        continue
                    
                    # Check age
                    created_at_str = metadata.get("created_at")
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str)
                            if created_at > cutoff_date:
                                total_kept += 1
                                continue
                        except (ValueError, TypeError):
                            # Can't parse date, assume it's old
                            pass
                    
                    # Mark for pruning
                    ids_to_prune.append(chunk_id)
                
                # Delete pruned chunks
                if ids_to_prune:
                    vector_store.collection.delete(ids=ids_to_prune)
                    total_pruned += len(ids_to_prune)
                    logger.info(f"[{codebase_name}] Pruned {len(ids_to_prune)} obsolete memories")
            
            logger.info(f"Pruning complete: {total_pruned} pruned, {total_kept} kept, {total_processed} processed")
            
            return {
                "pruned": total_pruned,
                "kept": total_kept,
                "total_processed": total_processed,
                "max_age_days": max_age_days,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during pruning: {e}")
            return {"error": str(e), "pruned": 0, "kept": 0, "total_processed": 0}
    
    async def delete_async(
        self,
        memory_id: str,
        codebase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a specific memory by ID.
        
        This can delete any memory including pinned ones (decisions, lessons).
        Use this when a decision is superseded or a lesson is outdated.
        
        Args:
            memory_id: The ID of the memory to delete
            codebase: Optional codebase to delete from (searches all if not specified)
            
        Returns:
            Dict with deletion result
        """
        if not self._vector_stores:
            return {"error": "No codebases configured", "success": False}
        
        try:
            # Determine which vector stores to search
            if codebase:
                stores_to_search = {codebase: self._vector_stores.get(codebase)}
                if not stores_to_search[codebase]:
                    return {"error": f"Codebase not found: {codebase}", "success": False}
            else:
                stores_to_search = self._vector_stores
            
            # Try to find and delete the memory
            for codebase_name, vector_store in stores_to_search.items():
                try:
                    # Check if the memory exists
                    result = vector_store.collection.get(ids=[memory_id])
                    if result["ids"]:
                        # Found it - delete and return
                        vector_store.delete(memory_id)
                        
                        # Also remove from hybrid search index if present
                        try:
                            self.hybrid_searcher.remove_from_index(codebase_name, memory_id)
                        except Exception:
                            pass  # Not all memories are in the keyword index
                        
                        logger.info(f"Deleted memory {memory_id} from codebase {codebase_name}")
                        return {
                            "success": True,
                            "deleted_id": memory_id,
                            "codebase": codebase_name
                        }
                except Exception as e:
                    logger.debug(f"Memory {memory_id} not found in {codebase_name}: {e}")
                    continue
            
            return {"error": f"Memory not found: {memory_id}", "success": False}
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"error": str(e), "success": False}
    
    async def reindex_codebase_async(self, codebase_name: str) -> Dict[str, Any]:
        """
        Trigger a full re-index of a specific codebase.
        
        Args:
            codebase_name: Name of the codebase to re-index
            
        Returns:
            Dict with status message
        """
        codebase = self.config.get_codebase_by_name(codebase_name)
        if not codebase:
            return {"error": f"Codebase not found: {codebase_name}"}
        
        vector_store = self._vector_stores.get(codebase_name)
        if vector_store:
            vector_store.clear_collection()
        
        # Reset status
        self._indexing_status[codebase_name]["status"] = "pending"
        self._indexing_status[codebase_name]["progress"] = 0.0
        self._indexing_status[codebase_name]["files_processed"] = 0
        self._indexing_status[codebase_name]["total_files"] = 0
        
        # Reset call graph for rebuild
        if codebase_name in self._call_graphs:
            self._call_graphs[codebase_name].clear()
            self._call_graph_builders[codebase_name] = CallGraphBuilder(self._call_graphs[codebase_name])
        
        # Start re-indexing
        await self._index_codebase(codebase)
        
        return {"message": f"Re-indexing completed for: {codebase_name}"}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Public API - Sync versions (wrap async)
    # ─────────────────────────────────────────────────────────────────────────
    
    def initialize(self) -> None:
        """Sync wrapper for initialize_async - blocks until indexing completes"""
        _run_sync(self.initialize_async(wait_for_indexing=True))
    
    async def start_file_watchers_async(self) -> None:
        """
        Start file watchers for all codebases.
        
        This should be called from the main event loop AFTER initialization,
        not from within initialize() which may run in a temporary event loop.
        """
        if not self.config.enable_file_watcher:
            return
        
        enabled_codebases = self.config.get_enabled_codebases()
        for codebase in enabled_codebases:
            if codebase.name in self._vector_stores and codebase.name not in self._file_watcher_tasks:
                self._file_watcher_tasks[codebase.name] = asyncio.create_task(
                    self._watch_for_changes(codebase)
                )
        
        if self._file_watcher_tasks:
            logger.info(f"File watchers started for {len(self._file_watcher_tasks)} codebase(s)")
    
    async def stop_file_watchers_async(self) -> None:
        """Stop all file watcher tasks"""
        for name, task in self._file_watcher_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._file_watcher_tasks.clear()
        logger.info("All file watchers stopped")
    
    async def start_background_summarizer_async(self) -> None:
        """
        Start background summarization.
        
        This should be called from the main event loop AFTER initialization,
        not from within initialize() which may run in a temporary event loop.
        """
        await self._start_background_summarizer()
    
    def search(
        self,
        query: str,
        codebase: Optional[str] = None,
        max_results: int = 10,
        project_id: Optional[str] = None,
        search_mode: str = "auto",
        domain_boosts: Optional[Dict[str, float]] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        function_names: Optional[List[str]] = None,
        annotations: Optional[List[str]] = None,
        has_annotations: Optional[bool] = None,
        has_docstrings: Optional[bool] = None,
        min_class_count: Optional[int] = None,
        min_function_count: Optional[int] = None,
        calls: Optional[List[str]] = None,
        accesses: Optional[List[str]] = None,
        subscripts: Optional[List[str]] = None,
        include_summaries: bool = False,
        boost_summarized: bool = True
    ) -> Dict[str, Any]:
        """Sync wrapper for search_async"""
        return _run_sync(self.search_async(
            query, codebase, max_results, project_id, search_mode, 
            domain_boosts, include_tags, exclude_tags,
            languages, class_names, function_names, annotations,
            has_annotations, has_docstrings, min_class_count, min_function_count,
            calls, accesses, subscripts,
            include_summaries, boost_summarized
        ))
    
    def store(
        self,
        content: str,
        project_id: str = "default",
        codebase: Optional[str] = None,
        role: str = "user",
        tags: Optional[List[str]] = None,
        pin: bool = False,
        source: str = "api",
        memory_type: str = "conversation"
    ) -> Dict[str, Any]:
        """Sync wrapper for store_async"""
        return _run_sync(self.store_async(content, project_id, codebase, role, tags, pin, source, memory_type))
    
    def reindex_codebase(self, codebase_name: str) -> Dict[str, Any]:
        """Sync wrapper for reindex_codebase_async"""
        return _run_sync(self.reindex_codebase_async(codebase_name))
    
    def prune(
        self,
        project_id: Optional[str] = None,
        max_age_days: int = 30,
        codebase: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sync wrapper for prune_async"""
        return _run_sync(self.prune_async(project_id, max_age_days, codebase))
    
    def delete(
        self,
        memory_id: str,
        codebase: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sync wrapper for delete_async"""
        return _run_sync(self.delete_async(memory_id, codebase))
    
    def reset_all(self) -> Dict[str, Any]:
        """
        Clear all indexed data from all codebases.
        
        This removes:
        - All vector store collections (code chunks, memories)
        - All file index metadata
        - All BM25 indices
        
        After reset, the next initialize() call will perform a full reindex.
        
        Returns:
            Dict with reset status
        """
        cleared_codebases = []
        
        for codebase_name, vector_store in self._vector_stores.items():
            try:
                # Clear the main collection
                vector_store.clear_collection()
                
                # Clear file index metadata
                if hasattr(vector_store, 'file_index') and vector_store.file_index:
                    try:
                        # Get all file paths and delete them
                        all_files = vector_store.file_index.get_all_indexed_files()
                        for file_path in all_files.keys():
                            vector_store.file_index.delete_file_info(file_path)
                    except Exception as e:
                        logger.warning(f"[{codebase_name}] Error clearing file index: {e}")
                
                # Clear BM25 index
                self.hybrid_searcher.clear_index(codebase_name)
                
                # Reset indexing status
                self._indexing_status[codebase_name]["status"] = "not_started"
                self._indexing_status[codebase_name]["progress"] = 0.0
                self._indexing_status[codebase_name]["files_processed"] = 0
                self._indexing_status[codebase_name]["total_files"] = 0
                self._indexing_status[codebase_name]["indexed_files_count"] = 0
                self._indexing_status[codebase_name]["current_file"] = None
                self._indexing_status[codebase_name]["error_message"] = None
                
                cleared_codebases.append(codebase_name)
                logger.info(f"[{codebase_name}] Cleared all indexed data")
                
            except Exception as e:
                logger.error(f"[{codebase_name}] Error during reset: {e}")
        
        logger.info(f"Reset complete. Cleared {len(cleared_codebases)} codebase(s)")
        
        return {
            "success": True,
            "cleared_codebases": cleared_codebases,
            "message": f"Cleared {len(cleared_codebases)} codebase(s)"
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Status methods (sync - no async I/O needed)
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        # Aggregate status
        all_completed = all(
            s["status"] == "completed" for s in self._indexing_status.values()
        ) if self._indexing_status else True
        any_indexing = any(
            s["status"] == "indexing" for s in self._indexing_status.values()
        )
        any_error = any(
            s["status"] == "error" for s in self._indexing_status.values()
        )
        
        if any_indexing:
            overall_status = "indexing"
        elif any_error:
            overall_status = "error"
        elif all_completed:
            overall_status = "completed"
        else:
            overall_status = "not_started"
        
        total_files = sum(s["total_files"] for s in self._indexing_status.values())
        processed_files = sum(s["files_processed"] for s in self._indexing_status.values())
        
        return {
            "initialized": bool(self._vector_stores),
            "status": overall_status,
            "progress": processed_files / total_files if total_files > 0 else 1.0,
            "total_codebases": len(self._vector_stores),
            "codebases": {
                name: {
                    "status": status["status"],
                    "progress": status["progress"],
                    "files_processed": status["files_processed"],
                    "total_files": status["total_files"],
                    "indexed_files_count": status["indexed_files_count"]
                }
                for name, status in self._indexing_status.items()
            },
            "persist_directory": self.persist_directory
        }
    
    def get_codebase_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific codebase"""
        status = self._indexing_status.get(name)
        if not status:
            return None
        
        vector_store = self._vector_stores.get(name)
        indexed_files = vector_store.get_indexed_files() if vector_store else {}
        
        return {
            "name": status["name"],
            "path": status["path"],
            "status": status["status"],
            "progress": status["progress"],
            "files_processed": status["files_processed"],
            "total_files": status["total_files"],
            "indexed_files_count": len(indexed_files),
            "current_file": status["current_file"],
            "error_message": status["error_message"],
            "indexed_files": list(indexed_files.keys())[:100]  # Limit to first 100
        }
    
    def get_all_codebase_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all codebases"""
        return {name: self.get_codebase_status(name) for name in self._indexing_status}
    
    def list_codebases(self) -> List[Dict[str, Any]]:
        """
        List all configured codebases with their info.
        
        NOTE: This method is expensive as it queries ChromaDB for accurate file counts.
        For fast status checks, use list_codebases_basic() instead.
        """
        codebases = []
        for codebase in self.config.get_enabled_codebases():
            vector_store = self._vector_stores.get(codebase.name)
            indexed_count = len(vector_store.get_indexed_files()) if vector_store else 0
            
            codebases.append({
                "name": codebase.name,
                "path": codebase.path,
                "description": codebase.description,
                "enabled": codebase.enabled,
                "indexed_files_count": indexed_count,
                "extensions": codebase.extensions
            })
        
        return codebases
    
    def list_codebases_basic(self) -> List[Dict[str, Any]]:
        """
        List all configured codebases with basic info (fast, no ChromaDB queries).
        
        Uses cached file counts from indexing status rather than querying ChromaDB.
        This is suitable for health checks and dashboard polling.
        
        Returns:
            List of codebase info dictionaries
        """
        codebases = []
        for codebase in self.config.get_enabled_codebases():
            # Use pre-computed count from indexing status (no ChromaDB query)
            status = self._indexing_status.get(codebase.name, {})
            indexed_count = status.get("indexed_files_count", 0)
            
            codebases.append({
                "name": codebase.name,
                "path": codebase.path,
                "description": codebase.description,
                "enabled": codebase.enabled,
                "indexed_files_count": indexed_count,
                "extensions": codebase.extensions
            })
        
        return codebases
    
    def get_vector_store(self, codebase_name: str) -> Optional[ChromaVectorStore]:
        """Get vector store for a specific codebase (for advanced use)"""
        return self._vector_stores.get(codebase_name)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal methods - Indexing
    # ─────────────────────────────────────────────────────────────────────────
    
    def _print_progress(self, codebase_name: str, current: int, total: int, phase: str = "Indexing") -> None:
        """Print a single-line progress indicator that updates in place"""
        if total == 0:
            pct = 100
        else:
            pct = int((current / total) * 100)
        
        bar_width = 30
        filled = int(bar_width * current / total) if total > 0 else bar_width
        bar = "#" * filled + "-" * (bar_width - filled)
        
        # Use carriage return to update line in place
        line = f"\r[{codebase_name}] {phase}: [{bar}] {current}/{total} files ({pct}%)"
        sys.stderr.write(line)
        sys.stderr.flush()
    
    def _print_progress_complete(self, codebase_name: str, total: int, new_files: int, modified: int, deleted: int) -> None:
        """Print completion message on a new line"""
        sys.stderr.write("\n")  # Move to new line after progress bar
        sys.stderr.flush()
        logger.info(f"[{codebase_name}] Indexing complete: {total} files ({new_files} new, {modified} modified, {deleted} deleted)")

    async def _index_codebase(self, codebase: CodebaseConfig) -> None:
        """Index a single codebase - incremental if index exists"""
        codebase_path = Path(codebase.path)
        status = self._indexing_status[codebase.name]
        vector_store = self._vector_stores[codebase.name]
        
        try:
            status["status"] = "indexing"
            status["progress"] = 0.0
            
            logger.info(f"[{codebase.name}] Starting indexing: {codebase_path}")
            
            # Clean up any orphan chunks from previous incomplete indexing
            orphan_count = vector_store.cleanup_orphan_chunks(codebase.name)
            if orphan_count > 0:
                logger.info(f"[{codebase.name}] Recovered from incomplete indexing")
            
            # Find all code files
            code_extensions = codebase.get_extension_set()
            
            # Show scanning phase
            self._print_progress(codebase.name, 0, 1, "Scanning")
            
            code_files = []
            for ext in code_extensions:
                code_files.extend(list(codebase_path.rglob(f'*{ext}')))
            
            # Filter out ignored patterns
            code_files = [f for f in code_files if not codebase.should_ignore(str(f))]
            
            self._print_progress(codebase.name, 1, 1, "Scanning")
            sys.stderr.write(f" - found {len(code_files)} files\n")
            sys.stderr.flush()
            
            # Get existing indexed files
            indexed_files = vector_store.get_indexed_files()
            current_file_paths = set()
            
            # Determine which files need indexing - show analyzing phase
            files_to_index = []
            files_to_reindex = []
            
            total_to_analyze = len(code_files)
            for idx, file_path in enumerate(code_files):
                if idx % 50 == 0:  # Update every 50 files to reduce flickering
                    self._print_progress(codebase.name, idx, total_to_analyze, "Analyzing")
                
                relative_path = str(file_path.relative_to(codebase_path))
                current_file_paths.add(relative_path)
                
                try:
                    mtime = file_path.stat().st_mtime
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    content_hash = ChromaVectorStore.compute_content_hash(content)
                    
                    if vector_store.needs_reindex(relative_path, mtime, content_hash):
                        if relative_path in indexed_files:
                            files_to_reindex.append((file_path, mtime, content_hash, content))
                        else:
                            files_to_index.append((file_path, mtime, content_hash, content))
                except Exception as e:
                    logger.debug(f"[{codebase.name}] Error checking file {file_path}: {e}")
            
            self._print_progress(codebase.name, total_to_analyze, total_to_analyze, "Analyzing")
            sys.stderr.write("\n")
            sys.stderr.flush()
            
            # Find and remove deleted files
            deleted_files = set(indexed_files.keys()) - current_file_paths
            for deleted_path in deleted_files:
                vector_store.remove_file_chunks(deleted_path)
            
            # Remove old chunks for files to re-index
            for file_path, _, _, _ in files_to_reindex:
                relative_path = str(file_path.relative_to(codebase_path))
                vector_store.remove_file_chunks(relative_path)
            
            # Combine files to process
            all_files = files_to_index + files_to_reindex
            status["total_files"] = len(all_files)
            
            # Check if up to date
            if not all_files and not deleted_files:
                existing_count = len(indexed_files)
                logger.info(f"[{codebase.name}] Index is up to date ({existing_count} files)")
                status["status"] = "completed"
                status["progress"] = 1.0
                status["indexed_files_count"] = existing_count
                self._on_codebase_indexing_complete(codebase.name)
                return
            
            # Process files in batches for better GPU utilization
            total_files = len(all_files)
            await self._index_files_batched(
                all_files, codebase, vector_store, status, total_files
            )
            
            status["status"] = "completed"
            status["progress"] = 1.0
            status["indexed_files_count"] = len(vector_store.get_indexed_files())
            self._on_codebase_indexing_complete(codebase.name)
            
            # Build BM25 index for hybrid search
            import time
            bm25_start = time.time()
            logger.info(f"[{codebase.name}] Building BM25 index...")
            self.hybrid_searcher.rebuild_index(codebase.name)
            bm25_elapsed = time.time() - bm25_start
            logger.info(f"[{codebase.name}] BM25 index built in {bm25_elapsed:.2f}s")
            
            # Calculate import graph centrality after all files are processed
            try:
                import_graph = self._import_graphs[codebase.name]
                centrality_scores = import_graph.calculate_centrality()
                if centrality_scores:
                    logger.info(f"[{codebase.name}] Calculated centrality for {len(centrality_scores)} files")
                    # TODO: Store centrality scores for future LLM summarization prioritization
            except Exception as e:
                logger.warning(f"[{codebase.name}] Failed to calculate import graph centrality: {e}")
            
            # Log call graph statistics
            try:
                call_graph = self._call_graphs[codebase.name]
                call_graph_stats = call_graph.get_stats()
                if call_graph_stats["total_methods"] > 0:
                    logger.info(
                        f"[{codebase.name}] Built call graph: {call_graph_stats['total_methods']} methods, "
                        f"{call_graph_stats['total_edges']} call edges"
                    )
            except Exception as e:
                logger.debug(f"[{codebase.name}] Failed to get call graph stats: {e}")
            
            # Final summary
            self._print_progress_complete(
                codebase.name,
                status["indexed_files_count"],
                len(files_to_index),
                len(files_to_reindex),
                len(deleted_files)
            )
            
        except Exception as e:
            sys.stderr.write("\n")  # Ensure we're on a new line
            logger.error(f"[{codebase.name}] Error during indexing: {e}")
            status["status"] = "error"
            status["error_message"] = str(e)
    
    async def _index_files_batched(
        self,
        all_files: List[tuple],
        codebase: CodebaseConfig,
        vector_store: ChromaVectorStore,
        status: dict,
        total_files: int
    ) -> None:
        """
        Index files using batched embedding generation for better GPU utilization.
        
        Collects chunks across multiple files and processes them in large batches
        to maximize GPU throughput.
        """
        codebase_path = Path(codebase.path)
        
        # Determine optimal batch size based on available GPU memory
        # RTX 4090 with 24GB can handle large batches
        embedding_batch_size = self._get_optimal_batch_size()
        
        # Collect all chunks first, then batch embed
        all_chunk_data = []  # List of (file_path, mtime, content_hash, chunk_text, metadata)
        file_chunk_ranges = {}  # file_path -> (start_idx, end_idx) in all_chunk_data
        files_without_chunks = []  # Files too small to index but should still be tracked
        
        # Phase 1: Chunk all files (CPU-bound, relatively fast)
        for i, (file_path, mtime, content_hash, content) in enumerate(all_files):
            if i % 20 == 0:
                self._print_progress(codebase.name, i, total_files, "Chunking")
            
            relative_path = str(file_path.relative_to(codebase_path))
            
            if not content.strip():
                # Empty file - track it so file watcher doesn't keep re-checking
                files_without_chunks.append((relative_path, mtime, content_hash))
                continue
            
            chunks = self.chunking_manager.chunk_text(content, relative_path)
            
            # Phase 1: Extract heuristic metadata for this file
            heuristic_metadata = None
            try:
                heuristic_metadata = self.heuristic_extractor.extract_file_metadata(relative_path, content)
                if heuristic_metadata:
                    # Add to import graph for centrality calculation
                    import_graph = self._import_graphs[codebase.name]
                    import_graph.add_file(relative_path, heuristic_metadata.imports)
                    
                    # Phase 4: Build call graph from method implementation details
                    if heuristic_metadata.method_details:
                        call_graph_builder = self._call_graph_builders[codebase.name]
                        call_graph_builder.build_from_heuristics(relative_path, heuristic_metadata)
            except Exception as e:
                logger.debug(f"[{codebase.name}] Heuristic extraction failed for {relative_path}: {e}")
            
            start_idx = len(all_chunk_data)
            for chunk_text, metadata in chunks:
                if len(chunk_text.strip()) >= 50:
                    # Enhance metadata with heuristic information
                    if heuristic_metadata:
                        self._enhance_chunk_metadata(metadata, heuristic_metadata)
                    all_chunk_data.append((file_path, mtime, content_hash, chunk_text, metadata, heuristic_metadata))
            end_idx = len(all_chunk_data)
            
            if end_idx > start_idx:
                file_chunk_ranges[relative_path] = (start_idx, end_idx, file_path, mtime, content_hash)
            else:
                # File has content but all chunks too small - still track it
                files_without_chunks.append((relative_path, mtime, content_hash))
        
        self._print_progress(codebase.name, total_files, total_files, "Chunking")
        sys.stderr.write(f" - {len(all_chunk_data)} chunks\n")
        sys.stderr.flush()
        
        if not all_chunk_data:
            # No indexable chunks, but still track files without chunks
            # so the file watcher doesn't keep re-checking them
            if files_without_chunks:
                file_index_batch = [
                    {
                        'file_path': relative_path,
                        'mtime': mtime,
                        'content_hash': content_hash,
                        'chunk_ids': []
                    }
                    for relative_path, mtime, content_hash in files_without_chunks
                ]
                vector_store.update_file_index_batch(file_index_batch)
                logger.info(f"[{codebase.name}] Tracked {len(files_without_chunks)} files without indexable content")
            return
        
        # Phase 2: Generate embeddings in large batches (GPU-bound)
        all_texts = [chunk_text for _, _, _, chunk_text, _, _ in all_chunk_data]
        total_chunks = len(all_texts)
        all_embeddings = []
        
        for batch_start in range(0, total_chunks, embedding_batch_size):
            batch_end = min(batch_start + embedding_batch_size, total_chunks)
            batch_texts = all_texts[batch_start:batch_end]
            
            self._print_progress(codebase.name, batch_start, total_chunks, "Embedding")
            
            # Run embedding in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None, self.embedder.generate_batch, batch_texts
            )
            all_embeddings.extend(batch_embeddings)
            await asyncio.sleep(0)  # Yield control
        
        self._print_progress(codebase.name, total_chunks, total_chunks, "Embedding")
        sys.stderr.write("\n")
        sys.stderr.flush()
        
        # Phase 3: Store chunks in batches and update file index
        # Batch size for ChromaDB inserts (larger = faster, but more memory)
        chroma_batch_size = 500
        
        # First, prepare all MemoryChunks and track file->chunk mappings
        all_memory_chunks = []
        file_to_chunk_ids = {}  # relative_path -> list of chunk_ids
        
        self._print_progress(codebase.name, 0, len(all_chunk_data), "Preparing")
        
        for idx, (file_path, mtime, content_hash, chunk_text, metadata, heuristic_metadata) in enumerate(all_chunk_data):
            relative_path = str(file_path.relative_to(codebase_path))
            
            chunk_id = str(uuid.uuid4())
            
            # Track chunk IDs per file for file index update
            if relative_path not in file_to_chunk_ids:
                file_to_chunk_ids[relative_path] = {
                    'chunk_ids': [],
                    'file_path': file_path,
                    'mtime': mtime,
                    'content_hash': content_hash
                }
            file_to_chunk_ids[relative_path]['chunk_ids'].append(chunk_id)
            
            # Build tags
            tags = [
                f"file:{relative_path}",
                f"ext:{file_path.suffix}",
                f"codebase:{codebase.name}"
            ]
            if metadata.module:
                tags.append(f"module:{metadata.module}")
            if metadata.domain:
                tags.append(f"domain:{metadata.domain}")
            if metadata.parent_class:
                tags.append(f"parent:{metadata.parent_class}")
            tags.append(f"lines:{metadata.start_line}-{metadata.end_line}")
            
            # Add heuristic metadata tags
            if heuristic_metadata:
                heuristic_dict = heuristic_metadata.to_dict()
                tags.append(f"lang:{heuristic_dict['language']}")
                
                # Add class names
                for class_name in heuristic_dict.get('class_names', []):
                    tags.append(f"class:{class_name}")
                
                # Add function names
                for func_name in heuristic_dict.get('function_names', []):
                    tags.append(f"function:{func_name}")
                
                # Add annotations
                for annotation in heuristic_dict.get('annotations', []):
                    tags.append(f"annotation:{annotation}")
                
                # Add import modules
                for module in heuristic_dict.get('import_modules', []):
                    if module:  # Skip empty modules
                        tags.append(f"imports:{module}")
                
                # Add metadata counts for filtering
                tags.append(f"class_count:{heuristic_dict.get('class_count', 0)}")
                tags.append(f"function_count:{heuristic_dict.get('function_count', 0)}")
                tags.append(f"method_count:{heuristic_dict.get('method_count', 0)}")
                tags.append(f"import_count:{heuristic_dict.get('import_count', 0)}")
                
                if heuristic_dict.get('has_annotations', False):
                    tags.append("has_annotations:true")
                if heuristic_dict.get('has_docstrings', False):
                    tags.append("has_docstrings:true")
            
            # Phase 1 Enhancement: Add implementation signal tags from chunk metadata
            # These enable filtering by method calls, attribute access, subscripts, and parameter usage
            signal_tags = metadata.get_signal_tags()
            if signal_tags:
                tags.extend(signal_tags)
            
            memory_chunk = MemoryChunk(
                id=chunk_id,
                project_id=codebase.name,
                role=RoleEnum.SYSTEM,
                prompt="",
                response="",
                doc_text=chunk_text,
                embedding_id="",
                tags=tags,
                pin=False,
                relevance_score=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="codebase_indexing"
            )
            
            all_memory_chunks.append(memory_chunk)
            
            # Add to BM25 index (this is fast, in-memory)
            self.hybrid_searcher.add_to_index(codebase.name, memory_chunk)
        
        # Batch insert into ChromaDB
        total_chunks = len(all_memory_chunks)
        for batch_start in range(0, total_chunks, chroma_batch_size):
            batch_end = min(batch_start + chroma_batch_size, total_chunks)
            
            chunk_batch = all_memory_chunks[batch_start:batch_end]
            embedding_batch = all_embeddings[batch_start:batch_end]
            
            self._print_progress(codebase.name, batch_start, total_chunks, "Storing")
            
            vector_store.add_batch(chunk_batch, embedding_batch)
            await asyncio.sleep(0)  # Yield control
        
        self._print_progress(codebase.name, total_chunks, total_chunks, "Storing")
        sys.stderr.write("\n")
        sys.stderr.flush()
        
        # Batch update file index for all files
        import time
        file_index_start = time.time()
        
        # Prepare batch data - include files with chunks
        file_index_batch = [
            {
                'file_path': relative_path,
                'mtime': file_info['mtime'],
                'content_hash': file_info['content_hash'],
                'chunk_ids': file_info['chunk_ids']
            }
            for relative_path, file_info in file_to_chunk_ids.items()
        ]
        
        # Also include files without indexable chunks (empty or too small)
        # This prevents the file watcher from repeatedly trying to re-index them
        for relative_path, mtime, content_hash in files_without_chunks:
            file_index_batch.append({
                'file_path': relative_path,
                'mtime': mtime,
                'content_hash': content_hash,
                'chunk_ids': []  # No chunks, but file is tracked
            })
        
        # Single batch upsert
        vector_store.update_file_index_batch(file_index_batch)
        
        file_index_elapsed = time.time() - file_index_start
        logger.info(f"[{codebase.name}] File index updated for {len(file_index_batch)} files ({len(files_without_chunks)} empty/small) in {file_index_elapsed:.2f}s")
    
    def _enhance_chunk_metadata(self, chunk_metadata, heuristic_metadata):
        """
        Enhance chunk metadata with heuristic information.
        
        Args:
            chunk_metadata: ChunkMetadata object from chunking
            heuristic_metadata: HeuristicMetadata object from heuristic extraction
        """
        try:
            # Add language information
            if not hasattr(chunk_metadata, 'language'):
                chunk_metadata.language = heuristic_metadata.language
            
            # Enhance domain detection with heuristic information
            if chunk_metadata.domain is None:
                # Try to infer domain from heuristic metadata
                if heuristic_metadata.annotations:
                    # Check for common framework annotations
                    for annotation in heuristic_metadata.annotations:
                        annotation_lower = annotation.lower()
                        if any(test_ann in annotation_lower for test_ann in ['test', 'junit', 'fact']):
                            chunk_metadata.domain = 'test'
                            break
                        elif any(web_ann in annotation_lower for web_ann in ['controller', 'restcontroller', 'getmapping', 'postmapping']):
                            chunk_metadata.domain = 'web'
                            break
                        elif any(data_ann in annotation_lower for data_ann in ['repository', 'entity', 'table']):
                            chunk_metadata.domain = 'data'
                            break
                        elif any(service_ann in annotation_lower for service_ann in ['service', 'component', 'bean']):
                            chunk_metadata.domain = 'service'
                            break
                
                # Check function/class names for domain hints
                if chunk_metadata.domain is None:
                    all_names = heuristic_metadata.class_names + heuristic_metadata.function_names
                    for name in all_names:
                        name_lower = name.lower()
                        if any(test_name in name_lower for test_name in ['test', 'spec', 'should']):
                            chunk_metadata.domain = 'test'
                            break
                        elif any(util_name in name_lower for util_name in ['util', 'helper', 'common']):
                            chunk_metadata.domain = 'utility'
                            break
                        elif any(config_name in name_lower for config_name in ['config', 'setting', 'property']):
                            chunk_metadata.domain = 'config'
                            break
            
            # Add additional metadata fields for search enhancement
            if not hasattr(chunk_metadata, 'heuristic_data'):
                chunk_metadata.heuristic_data = {
                    'language': heuristic_metadata.language,
                    'class_count': len(heuristic_metadata.classes),
                    'function_count': len(heuristic_metadata.functions),
                    'method_count': len(heuristic_metadata.methods),
                    'import_count': len(heuristic_metadata.imports),
                    'has_annotations': heuristic_metadata.has_annotations,
                    'has_docstrings': heuristic_metadata.has_docstrings,
                    'annotations': heuristic_metadata.annotations[:5],  # Limit to first 5
                    'class_names': heuristic_metadata.class_names[:3],  # Limit to first 3
                    'function_names': heuristic_metadata.function_names[:5]  # Limit to first 5
                }
                
        except Exception as e:
            logger.debug(f"Failed to enhance chunk metadata: {e}")
    
    def _get_optimal_batch_size(self) -> int:
        """
        Get the embedding batch size (already calculated during init).
        
        Returns:
            Batch size for embedding generation
        """
        return self._embedding_batch_size

    async def _index_single_file(
        self,
        file_path: Path,
        content: str,
        mtime: float,
        content_hash: str,
        codebase: CodebaseConfig,
        vector_store: ChromaVectorStore
    ) -> None:
        """Index a single file (used by file watcher for incremental updates)"""
        codebase_path = Path(codebase.path)
        status = self._indexing_status[codebase.name]
        status["current_file"] = str(file_path)
        relative_path = str(file_path.relative_to(codebase_path))
        
        if not content.strip():
            # Empty file - still track it so we don't keep re-checking
            vector_store.update_file_index(relative_path, mtime, content_hash, [])
            return
        
        chunks = self.chunking_manager.chunk_text(content, relative_path)
        
        # Filter chunks and prepare texts for batch embedding
        valid_chunks = []
        chunk_texts = []
        for chunk_text, metadata in chunks:
            if len(chunk_text.strip()) < 50:
                continue
            valid_chunks.append((chunk_text, metadata))
            chunk_texts.append(chunk_text)
        
        if not valid_chunks:
            # File has content but no indexable chunks - still track it
            vector_store.update_file_index(relative_path, mtime, content_hash, [])
            return
        
        # Generate embeddings in batch (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.embedder.generate_batch, chunk_texts
        )
        
        chunk_ids = []
        for (chunk_text, metadata), embedding in zip(valid_chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            # Build tags from chunk metadata
            tags = [
                f"file:{relative_path}",
                f"ext:{file_path.suffix}",
                f"codebase:{codebase.name}"
            ]
            
            # Add module info from AST chunking
            if metadata.module:
                tags.append(f"module:{metadata.module}")
            if metadata.domain:
                tags.append(f"domain:{metadata.domain}")
            if metadata.parent_class:
                tags.append(f"parent:{metadata.parent_class}")
            
            # Add line range for precise location
            tags.append(f"lines:{metadata.start_line}-{metadata.end_line}")
            
            # Phase 1 Enhancement: Add implementation signal tags from chunk metadata
            # These enable filtering by method calls, attribute access, subscripts, and parameter usage
            signal_tags = metadata.get_signal_tags()
            if signal_tags:
                tags.extend(signal_tags)
            
            memory_chunk = MemoryChunk(
                id=chunk_id,
                project_id=codebase.name,
                role=RoleEnum.SYSTEM,
                prompt="",
                response="",
                doc_text=chunk_text,
                embedding_id="",
                tags=tags,
                pin=False,
                relevance_score=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="codebase_indexing"
            )
            
            vector_store.add(memory_chunk, embedding)
            
            # Also add to BM25 index for hybrid search
            self.hybrid_searcher.add_to_index(codebase.name, memory_chunk)
        
        if chunk_ids:
            vector_store.update_file_index(relative_path, mtime, content_hash, chunk_ids)
            logger.debug(f"[{codebase.name}] Indexed {len(chunk_ids)} chunks for: {relative_path}")
    
    def _filter_by_tags(
        self, 
        chunks: List[MemoryChunk], 
        include_tags: Optional[List[str]], 
        exclude_tags: Optional[List[str]]
    ) -> List[MemoryChunk]:
        """
        Filter chunks by tag inclusion/exclusion with prefix support.
        
        Tag patterns:
        - "exact_tag" - exact match
        - "prefix:*" - matches any tag starting with "prefix:"
        
        Args:
            chunks: List of MemoryChunk objects
            include_tags: Tags to include (must match at least one)
            exclude_tags: Tags to exclude (must not match any)
            
        Returns:
            Filtered list of chunks
        """
        if not include_tags and not exclude_tags:
            return chunks
        
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_tags = set(chunk.tags)
            
            # Check include tags (must match at least one)
            if include_tags:
                include_match = False
                for include_tag in include_tags:
                    if self._tag_matches(include_tag, chunk_tags):
                        include_match = True
                        break
                if not include_match:
                    continue
            
            # Check exclude tags (must not match any)
            if exclude_tags:
                exclude_match = False
                for exclude_tag in exclude_tags:
                    if self._tag_matches(exclude_tag, chunk_tags):
                        exclude_match = True
                        break
                if exclude_match:
                    continue
            
            filtered_chunks.append(chunk)
        
        if len(filtered_chunks) < len(chunks):
            logger.debug(f"Tag filtering: {len(chunks)} -> {len(filtered_chunks)} chunks")
        
        return filtered_chunks
    
    def _tag_matches(self, pattern: str, chunk_tags: set) -> bool:
        """
        Check if a tag pattern matches any chunk tags.
        
        Args:
            pattern: Tag pattern (exact or prefix:*)
            chunk_tags: Set of tags from the chunk
            
        Returns:
            True if pattern matches any tag
        """
        if pattern.endswith("*"):
            # Prefix match
            prefix = pattern[:-1]
            return any(tag.startswith(prefix) for tag in chunk_tags)
        else:
            # Exact match
            return pattern in chunk_tags
    
    def _filter_by_heuristics(
        self,
        chunks: List[MemoryChunk],
        languages: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        function_names: Optional[List[str]] = None,
        annotations: Optional[List[str]] = None,
        has_annotations: Optional[bool] = None,
        has_docstrings: Optional[bool] = None,
        min_class_count: Optional[int] = None,
        min_function_count: Optional[int] = None
    ) -> List[MemoryChunk]:
        """
        Filter chunks by heuristic metadata extracted during indexing.
        
        Args:
            chunks: List of MemoryChunk objects
            languages: Filter by programming languages
            class_names: Filter by class names
            function_names: Filter by function names  
            annotations: Filter by annotations
            has_annotations: Filter files that have/don't have annotations
            has_docstrings: Filter files that have/don't have docstrings
            min_class_count: Minimum number of classes in file
            min_function_count: Minimum number of functions in file
            
        Returns:
            Filtered list of chunks
        """
        if not any([languages, class_names, function_names, annotations, has_annotations is not None,
                   has_docstrings is not None, min_class_count is not None, min_function_count is not None]):
            return chunks
        
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_tags = set(chunk.tags)
            
            # Filter by language
            if languages:
                lang_match = False
                for lang in languages:
                    if f"lang:{lang}" in chunk_tags:
                        lang_match = True
                        break
                if not lang_match:
                    continue
            
            # Filter by class names
            if class_names:
                class_match = False
                for class_name in class_names:
                    if f"class:{class_name}" in chunk_tags:
                        class_match = True
                        break
                if not class_match:
                    continue
            
            # Filter by function names
            if function_names:
                func_match = False
                for func_name in function_names:
                    if f"function:{func_name}" in chunk_tags:
                        func_match = True
                        break
                if not func_match:
                    continue
            
            # Filter by annotations
            if annotations:
                annotation_match = False
                for annotation in annotations:
                    if f"annotation:{annotation}" in chunk_tags:
                        annotation_match = True
                        break
                if not annotation_match:
                    continue
            
            # Filter by has_annotations
            if has_annotations is not None:
                has_ann_tag = "has_annotations:true" in chunk_tags
                if has_annotations != has_ann_tag:
                    continue
            
            # Filter by has_docstrings
            if has_docstrings is not None:
                has_doc_tag = "has_docstrings:true" in chunk_tags
                if has_docstrings != has_doc_tag:
                    continue
            
            # Filter by minimum class count
            if min_class_count is not None:
                class_count = 0
                for tag in chunk_tags:
                    if tag.startswith("class_count:"):
                        try:
                            class_count = int(tag.split(":")[1])
                            break
                        except (ValueError, IndexError):
                            pass
                if class_count < min_class_count:
                    continue
            
            # Filter by minimum function count
            if min_function_count is not None:
                func_count = 0
                for tag in chunk_tags:
                    if tag.startswith("function_count:"):
                        try:
                            func_count = int(tag.split(":")[1])
                            break
                        except (ValueError, IndexError):
                            pass
                if func_count < min_function_count:
                    continue
            
            filtered_chunks.append(chunk)
        
        if len(filtered_chunks) < len(chunks):
            logger.debug(f"Heuristic filtering: {len(chunks)} -> {len(filtered_chunks)} chunks")
        
        return filtered_chunks
    
    def _filter_by_implementation_signals(
        self,
        chunks: List[MemoryChunk],
        calls: Optional[List[str]] = None,
        accesses: Optional[List[str]] = None,
        subscripts: Optional[List[str]] = None
    ) -> List[MemoryChunk]:
        """
        Filter chunks by implementation signals extracted during indexing.
        
        These signals are generated from AST analysis during Phase 1 indexing
        and stored as tags on chunks. This enables queries like:
        - "Find methods that call 'iloc'" -> calls=["iloc"]
        - "Find methods that access 'bar_index'" -> accesses=["bar_index"]
        - "Find methods with subscript patterns on 'iloc'" -> subscripts=["iloc"]
        
        Args:
            chunks: List of MemoryChunk objects
            calls: Filter by method calls (matches calls:* tags)
            accesses: Filter by attribute access (matches reads:* tags)
            subscripts: Filter by subscript patterns (matches subscript:* tags)
            
        Returns:
            Filtered list of chunks
        """
        if not any([calls, accesses, subscripts]):
            return chunks
        
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_tags = set(chunk.tags)
            
            # Filter by method calls (calls:method_name)
            if calls:
                call_match = False
                for call_name in calls:
                    # Check for exact match or partial match
                    for tag in chunk_tags:
                        if tag.startswith("calls:"):
                            tag_value = tag[6:]  # Remove "calls:" prefix
                            # Match if the call name appears in the tag value
                            if call_name in tag_value or tag_value.endswith(f".{call_name}"):
                                call_match = True
                                break
                    if call_match:
                        break
                if not call_match:
                    continue
            
            # Filter by attribute access (reads:attribute_name)
            if accesses:
                access_match = False
                for access_name in accesses:
                    for tag in chunk_tags:
                        if tag.startswith("reads:"):
                            tag_value = tag[6:]  # Remove "reads:" prefix
                            # Match if the access name appears in the tag value
                            if access_name in tag_value:
                                access_match = True
                                break
                    if access_match:
                        break
                if not access_match:
                    continue
            
            # Filter by subscript patterns (subscript:pattern)
            if subscripts:
                subscript_match = False
                for subscript_pattern in subscripts:
                    for tag in chunk_tags:
                        if tag.startswith("subscript:"):
                            tag_value = tag[10:]  # Remove "subscript:" prefix
                            # Match if the subscript pattern appears in the tag value
                            if subscript_pattern in tag_value:
                                subscript_match = True
                                break
                    if subscript_match:
                        break
                if not subscript_match:
                    continue
            
            filtered_chunks.append(chunk)
        
        if len(filtered_chunks) < len(chunks):
            logger.debug(f"Implementation signal filtering: {len(chunks)} -> {len(filtered_chunks)} chunks")
        
        return filtered_chunks

    def _deduplicate_chunks(self, chunks: List) -> List:
        """
        Remove duplicate chunks based on content similarity.
        
        Args:
            chunks: List of MemoryChunk objects
            
        Returns:
            List of deduplicated chunks, preserving order and keeping the first occurrence
        """
        if not chunks:
            return chunks
        
        seen_content = set()
        deduplicated = []
        
        for chunk in chunks:
            # Create a content hash for deduplication
            # Use first 200 chars to balance accuracy vs performance
            content_key = chunk.doc_text[:200].strip().lower()
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append(chunk)
            else:
                logger.debug(f"Removed duplicate chunk: {chunk.id[:8]}... (content: {content_key[:50]}...)")
        
        if len(deduplicated) < len(chunks):
            logger.info(f"Deduplication: {len(chunks)} -> {len(deduplicated)} chunks ({len(chunks) - len(deduplicated)} duplicates removed)")
        
        return deduplicated
    
    def _scan_for_file_changes_sync(
        self,
        codebase: CodebaseConfig,
        codebase_path: Path,
        vector_store: ChromaVectorStore,
        code_extensions: set
    ) -> tuple:
        """
        Synchronous file scanning - runs in executor to avoid blocking event loop.

        Returns:
            Tuple of (indexed_files, current_file_paths, files_changed)
        """
        indexed_files = vector_store.get_indexed_files()
        current_file_paths = set()
        files_changed = []

        for ext in code_extensions:
            for file_path in codebase_path.rglob(f'*{ext}'):
                if codebase.should_ignore(str(file_path)):
                    continue

                relative_path = str(file_path.relative_to(codebase_path))
                current_file_paths.add(relative_path)

                try:
                    mtime = file_path.stat().st_mtime
                    file_info = indexed_files.get(relative_path)
                    if file_info and file_info.get("mtime") == mtime:
                        continue

                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    content_hash = ChromaVectorStore.compute_content_hash(content)

                    if vector_store.needs_reindex(relative_path, mtime, content_hash):
                        files_changed.append((file_path, mtime, content_hash, content, relative_path in indexed_files))
                except Exception as e:
                    logger.debug(f"[{codebase.name}] Error checking file {file_path}: {e}")

        return indexed_files, current_file_paths, files_changed

    async def _watch_for_changes(self, codebase: CodebaseConfig) -> None:
        """Background task to watch for file changes"""
        logger.info(f"[{codebase.name}] Starting file watcher for: {codebase.path}")

        codebase_path = Path(codebase.path)
        vector_store = self._vector_stores[codebase.name]
        status = self._indexing_status[codebase.name]
        code_extensions = codebase.get_extension_set()

        while True:
            try:
                await asyncio.sleep(self._watch_interval)

                # Skip while indexing
                if status["status"] == "indexing":
                    continue

                # Run entire file scan in executor to avoid blocking event loop
                indexed_files, current_file_paths, files_changed = await self._run_blocking(
                    self._scan_for_file_changes_sync,
                    codebase,
                    codebase_path,
                    vector_store,
                    code_extensions
                )
                
                # Handle deleted files
                deleted_files = set(indexed_files.keys()) - current_file_paths
                for deleted_path in deleted_files:
                    vector_store.remove_file_chunks(deleted_path)
                    logger.info(f"[{codebase.name}] File watcher: Removed deleted file: {deleted_path}")
                
                # Re-index changed files
                new_files = []
                modified_files = []
                for file_path, mtime, content_hash, content, was_indexed in files_changed:
                    relative_path = str(file_path.relative_to(codebase_path))
                    
                    if was_indexed:
                        vector_store.remove_file_chunks(relative_path)
                        modified_files.append(relative_path)
                    else:
                        new_files.append(relative_path)
                    
                    await self._index_single_file(file_path, content, mtime, content_hash, codebase, vector_store)
                
                # Log changes
                if new_files:
                    for f in new_files:
                        logger.info(f"[{codebase.name}] File watcher: Added new file: {f}")
                if modified_files:
                    for f in modified_files:
                        logger.info(f"[{codebase.name}] File watcher: Re-indexed modified file: {f}")
                
                if files_changed or deleted_files:
                    # Rebuild BM25 index after changes
                    self.hybrid_searcher.rebuild_index(codebase.name)
                    logger.info(f"[{codebase.name}] File watcher summary: {len(new_files)} added, {len(modified_files)} modified, {len(deleted_files)} deleted")
                    status["indexed_files_count"] = len(vector_store.get_indexed_files())
                    
                    # Phase 6: Queue changed files for re-summarization
                    if self._summarization_stats["is_running"]:
                        for file_path, mtime, content_hash, content, was_indexed in files_changed:
                            relative_path = str(file_path.relative_to(codebase_path))
                            await self.queue_file_for_summarization(
                                codebase.name, 
                                relative_path, 
                                str(codebase_path),
                                priority=0.8  # Higher priority for changed files
                            )
                        
                        # Remove summaries for deleted files
                        for deleted_path in deleted_files:
                            await self.remove_file_summary(codebase.name, deleted_path)
                    
            except asyncio.CancelledError:
                logger.info(f"[{codebase.name}] File watcher stopped")
                break
            except Exception as e:
                logger.error(f"[{codebase.name}] Error in file watcher: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Phase 5: Summary Integration for Search Results
    # ─────────────────────────────────────────────────────────────────────────
    
    def _extract_file_path_from_chunk(self, chunk: MemoryChunk) -> Optional[str]:
        """Extract file path from a chunk's tags."""
        for tag in chunk.tags:
            if tag.startswith("file:"):
                return tag[5:]  # Remove "file:" prefix
        return None
    
    async def _get_file_summaries_async(
        self, 
        chunks: List[MemoryChunk], 
        codebase: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Look up LLM-generated summaries for files referenced in search results.
        
        Args:
            chunks: List of result chunks to find summaries for
            codebase: Optional codebase to search in
            
        Returns:
            Dict mapping file paths to their summary data
        """
        # Extract unique file paths from chunks
        file_paths = set()
        for chunk in chunks:
            file_path = self._extract_file_path_from_chunk(chunk)
            if file_path:
                file_paths.add(file_path)
        
        if not file_paths:
            return {}
        
        summaries = {}
        
        # Query vector stores for summary chunks (tagged with "summary")
        if codebase:
            stores_to_search = {codebase: self._vector_stores.get(codebase)}
        else:
            stores_to_search = self._vector_stores
        
        for codebase_name, vector_store in stores_to_search.items():
            if not vector_store:
                continue
            
            try:
                # Get all summary chunks for this codebase
                # Query by tag filter using ChromaDB's where clause
                summary_results = vector_store.collection.get(
                    where={"source": "llm_summarization"},
                    include=["metadatas", "documents"]
                )
                
                if not summary_results["ids"]:
                    continue
                
                # Match summaries to file paths
                for i, doc_id in enumerate(summary_results["ids"]):
                    doc = summary_results["documents"][i] if summary_results["documents"] else ""
                    metadata = summary_results["metadatas"][i] if summary_results["metadatas"] else {}
                    
                    # Extract file path from tags in metadata
                    tags = metadata.get("tags", [])
                    if isinstance(tags, str):
                        tags = tags.split(",")
                    
                    for tag in tags:
                        tag = tag.strip()
                        if tag.startswith("file:"):
                            summary_file = tag[5:]
                            if summary_file in file_paths:
                                # Parse summary fields from the document text
                                summaries[summary_file] = self._parse_summary_text(doc, metadata)
                                break
                            
            except Exception as e:
                logger.debug(f"[{codebase_name}] Error fetching summaries: {e}")
        
        return summaries
    
    def _parse_summary_text(self, doc_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse summary document text into structured format."""
        summary_data = {
            "raw_summary": doc_text
        }
        
        # Parse structured fields from the summary text
        lines = doc_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Purpose:"):
                summary_data["purpose"] = line[8:].strip()
            elif line.startswith("Pattern:"):
                summary_data["pattern"] = line[8:].strip()
            elif line.startswith("Domain:"):
                summary_data["domain"] = line[7:].strip()
            elif line.startswith("Language:"):
                summary_data["language"] = line[9:].strip()
            elif line.startswith("Key exports:"):
                exports_str = line[12:].strip()
                if exports_str and exports_str != "none":
                    summary_data["key_exports"] = [e.strip() for e in exports_str.split(",")]
            elif line.startswith("Dependencies:"):
                deps_str = line[13:].strip()
                if deps_str and deps_str != "none":
                    summary_data["dependencies"] = [d.strip() for d in deps_str.split(",")]
        
        # Add pattern and domain from metadata tags if available
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = tags.split(",")
        
        for tag in tags:
            tag = tag.strip()
            if tag.startswith("pattern:") and "pattern" not in summary_data:
                summary_data["pattern"] = tag[8:]
            elif tag.startswith("domain:") and "domain" not in summary_data:
                summary_data["domain"] = tag[7:]
        
        return summary_data
    
    async def _apply_summary_boost_async(
        self, 
        chunks: List[MemoryChunk], 
        codebase: Optional[str] = None,
        boost_factor: float = 1.15
    ) -> List[MemoryChunk]:
        """
        Apply a relevance boost to chunks from files that have LLM summaries.
        
        Files with summaries are considered more "understood" by the system,
        so they get a small boost in search relevance.
        
        Args:
            chunks: List of chunks to potentially boost
            codebase: Optional codebase filter
            boost_factor: Multiplier for boosting (default 1.15 = 15% boost)
            
        Returns:
            Chunks with updated relevance scores
        """
        # Get set of files that have summaries
        summarized_files = await self._get_summarized_files_async(codebase)
        
        if not summarized_files:
            return chunks
        
        boosted_count = 0
        for chunk in chunks:
            file_path = self._extract_file_path_from_chunk(chunk)
            if file_path and file_path in summarized_files:
                chunk.relevance_score *= boost_factor
                boosted_count += 1
        
        if boosted_count > 0:
            logger.debug(f"Applied summary boost to {boosted_count}/{len(chunks)} chunks")
        
        return chunks
    
    async def _get_summarized_files_async(self, codebase: Optional[str] = None) -> set:
        """
        Get set of file paths that have LLM summaries.
        
        This is cached per search to avoid repeated DB queries.
        
        Args:
            codebase: Optional codebase filter
            
        Returns:
            Set of file paths with summaries
        """
        summarized_files = set()
        
        if codebase:
            stores_to_search = {codebase: self._vector_stores.get(codebase)}
        else:
            stores_to_search = self._vector_stores
        
        for codebase_name, vector_store in stores_to_search.items():
            if not vector_store:
                continue
            
            try:
                # Query for summaries only
                summary_results = vector_store.collection.get(
                    where={"source": "llm_summarization"},
                    include=["metadatas"]
                )
                
                if not summary_results["ids"]:
                    continue
                
                for metadata in (summary_results["metadatas"] or []):
                    tags = metadata.get("tags", [])
                    if isinstance(tags, str):
                        tags = tags.split(",")
                    
                    for tag in tags:
                        tag = tag.strip()
                        if tag.startswith("file:"):
                            summarized_files.add(tag[5:])
                            break
                            
            except Exception as e:
                logger.debug(f"[{codebase_name}] Error fetching summarized files: {e}")
        
        return summarized_files
    
    def _generate_context_summary(self, chunks: List[MemoryChunk], query: str) -> str:
        """Generate a summary of search results"""
        if not chunks:
            return "No relevant context found."
        
        file_types = set()
        sources = set()
        codebases_found = set()
        
        for chunk in chunks:
            for tag in chunk.tags:
                if tag.startswith("ext:"):
                    file_types.add(tag[4:])
                elif tag.startswith("file:"):
                    sources.add(tag[5:].split('/')[0])
                elif tag.startswith("codebase:"):
                    codebases_found.add(tag[9:])
        
        parts = [f"Found {len(chunks)} relevant results"]
        if codebases_found:
            parts.append(f"Codebases: {', '.join(sorted(codebases_found))}")
        if file_types:
            parts.append(f"File types: {', '.join(sorted(file_types))}")
        if sources:
            parts.append(f"Areas: {', '.join(sorted(sources)[:5])}")
        
        return " | ".join(parts)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Background Summarization (Phase 4)
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _start_background_summarizer(self) -> None:
        """
        Initialize and start the background summarizer after indexing completes.
        
        This method:
        1. Checks if summarization is enabled
        2. Checks schema version and logs warning if outdated
        3. Initializes the LLM client and summarizer
        4. Queues files by centrality score (most important first)
        5. Starts the background summarization task
        """
        if not self._summarization_config.enabled:
            logger.info("[Summarization] Background summarization is disabled")
            return
        
        if not self._summarization_config.llm_enabled:
            logger.info("[Summarization] LLM summarization is disabled")
            return
        
        # Check schema version for all codebases and auto-invalidate if mismatched
        from ..storage.chroma import SUMMARY_SCHEMA_VERSION
        codebases_invalidated = []
        
        for codebase_name, vector_store in self._vector_stores.items():
            if not vector_store:
                continue
            
            is_current, stored_version = vector_store.summary_index.check_schema_version()
            
            if stored_version is None:
                # First time - set the schema version
                vector_store.summary_index.set_schema_version(SUMMARY_SCHEMA_VERSION)
                logger.info(f"[{codebase_name}] Initialized summary schema version: {SUMMARY_SCHEMA_VERSION}")
            elif not is_current:
                # Schema version mismatch - auto-invalidate summaries
                logger.info(
                    f"[{codebase_name}] Summary schema version upgrade: {stored_version} -> {SUMMARY_SCHEMA_VERSION}. "
                    f"Auto-invalidating existing summaries for re-summarization."
                )
                
                # Invalidate this codebase's summaries
                try:
                    result = await self.invalidate_summaries_async(codebase=codebase_name)
                    if result.get("success"):
                        summaries_cleared = result.get("total_summaries_cleared", 0)
                        chunks_removed = result.get("total_chunks_removed", 0)
                        logger.info(
                            f"[{codebase_name}] Auto-invalidation complete: "
                            f"{summaries_cleared} summaries cleared, {chunks_removed} chunks removed"
                        )
                        codebases_invalidated.append(codebase_name)
                    else:
                        logger.warning(f"[{codebase_name}] Auto-invalidation failed: {result.get('error')}")
                except Exception as e:
                    logger.error(f"[{codebase_name}] Error during auto-invalidation: {e}")
        
        try:
            from ..llm.ollama_client import OllamaClient
            
            # Step 1: Check if Ollama is installed
            is_installed, install_msg = OllamaClient.check_ollama_installed()
            if not is_installed:
                logger.warning(f"[Summarization] {install_msg}")
                logger.warning("[Summarization] Background summarization will not start.")
                return
            else:
                logger.info(f"[Summarization] {install_msg}")
            
            # Step 2: Check if Ollama is running
            is_running, running_msg = OllamaClient.check_ollama_running(self._summarization_config.ollama_url)
            if not is_running:
                logger.warning(f"[Summarization] {running_msg}")
                logger.warning("[Summarization] Background summarization will not start.")
                return
            
            # Create LLM client
            llm_client = OllamaClient(
                base_url=self._summarization_config.ollama_url,
                model=self._summarization_config.model,
                timeout=self._summarization_config.timeout_seconds
            )
            
            # Step 3: Health check with auto-pull if model is missing
            if not await llm_client.health_check(auto_pull=True):
                logger.warning(
                    f"[Summarization] Ollama health check failed. "
                    f"Model '{self._summarization_config.model}' may not be available. "
                    f"Background summarization will not start."
                )
                await llm_client.close()
                return
            
            # Warm up model (pre-load into memory to avoid timeouts)
            if not await llm_client.warm_up(timeout=120.0):
                logger.warning(
                    f"[Summarization] Failed to warm up model '{self._summarization_config.model}'. "
                    f"Summarization may be slow or fail on first requests."
                )
                # Continue anyway - model might load on first real request
            
            # Initialize summarizer
            self._summarizer = FileSummarizer(llm_client, self._summarization_config)
            
            # Reset session stats for fresh start
            self._reset_session_stats_if_idle()
            
            # Queue files by centrality (highest first), or all indexed files if no import graph
            total_queued = 0
            for codebase in self.config.get_enabled_codebases():
                codebase_path = Path(codebase.path)
                vector_store = self._vector_stores.get(codebase.name)
                
                if not vector_store:
                    logger.warning(f"[Summarization] No vector store found for codebase: {codebase.name}")
                    continue
                
                # Check indexing status for this codebase
                codebase_status = self._indexing_status.get(codebase.name, {})
                logger.info(f"[Summarization] Codebase {codebase.name} status: {codebase_status.get('status', 'unknown')}")
                
                # Try to get files by centrality
                priority_queue = self.get_file_centrality_scores(codebase.name, max_files=1000)
                
                if priority_queue:
                    # Use centrality-ordered files
                    logger.info(f"[Summarization] Using centrality ordering for {codebase.name}, found {len(priority_queue)} files")
                    skipped_count = 0
                    for file_path, score in priority_queue:
                        if self._summarizer._should_skip_file(file_path):
                            skipped_count += 1
                            self._summarization_stats["files_skipped_pattern"] += 1
                            continue
                        await self._summary_queue.put((-score, codebase.name, file_path, str(codebase_path)))
                        total_queued += 1
                    if skipped_count > 0:
                        logger.info(f"[Summarization] Skipped {skipped_count} files matching skip patterns for {codebase.name}")
                else:
                    # Fallback: queue all indexed files with equal priority
                    indexed_files = vector_store.get_indexed_files()
                    logger.info(f"[Summarization] No import graph for {codebase.name}, found {len(indexed_files)} indexed files")
                    
                    if not indexed_files:
                        logger.warning(f"[Summarization] No indexed files found for {codebase.name} - indexing may not be complete")
                        continue
                    
                    skipped_count = 0
                    for file_path in indexed_files.keys():
                        if self._summarizer._should_skip_file(file_path):
                            skipped_count += 1
                            self._summarization_stats["files_skipped_pattern"] += 1
                            continue
                        await self._summary_queue.put((0.5, codebase.name, file_path, str(codebase_path)))
                        total_queued += 1
                    
                    if skipped_count > 0:
                        logger.info(f"[Summarization] Skipped {skipped_count} files matching skip patterns for {codebase.name}")
                    
                logger.info(f"[Summarization] Queued {total_queued} files from {codebase.name}")
            
            self._summarization_stats["files_queued"] = total_queued
            self._summarization_stats["files_total_queued"] = total_queued
            
            if total_queued == 0:
                logger.warning("[Summarization] No files queued for summarization")
                
                # Provide diagnostic information
                for codebase in self.config.get_enabled_codebases():
                    vector_store = self._vector_stores.get(codebase.name)
                    if vector_store:
                        indexed_files = vector_store.get_indexed_files()
                        logger.info(f"[Summarization] {codebase.name}: {len(indexed_files)} indexed files")
                        
                        # Check how many already have summaries
                        summarized_files = vector_store.summary_index.get_all_summarized_files()
                        logger.info(f"[Summarization] {codebase.name}: {len(summarized_files)} files already summarized")
                
                await llm_client.close()
                return
            
            # Transfer items from async queue to thread-safe queue
            while not self._summary_queue.empty():
                try:
                    item = self._summary_queue.get_nowait()
                    self._summary_queue_threaded.put(item)
                except asyncio.QueueEmpty:
                    break
            
            # Start background thread (not async task) to avoid blocking main event loop
            self._summarizer_stop_event.clear()
            self._summarizer_thread = threading.Thread(
                target=self._background_summarizer_thread,
                name="SummarizerThread",
                daemon=True
            )
            self._summarizer_thread.start()
            self._summarization_stats["is_running"] = True
            
            logger.info(
                f"[Summarization] Started background summarizer THREAD with {total_queued} files queued "
                f"(using {self._summarization_config.model})"
            )
            
        except ImportError as e:
            logger.warning(f"[Summarization] LLM client not available: {e}")
        except Exception as e:
            logger.error(f"[Summarization] Failed to start background summarizer: {e}")
    
    async def _background_summarizer(self) -> None:
        """
        Background task that summarizes files in priority order.
        
        This runs continuously, processing files from the queue sequentially.
        It yields to active queries to avoid impacting search latency.
        """
        logger.info("[Summarization] Background summarizer started (sequential processing)")
        self._summarization_stats["is_running"] = True
        
        while True:
            # Yield to active queries - don't summarize while searches are happening
            if self._query_active.is_set():
                await asyncio.sleep(0.1)
                continue
            
            try:
                # Get next file from queue
                try:
                    file_data = await asyncio.wait_for(
                        self._summary_queue.get(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # No files in queue
                    if self._summary_queue.empty():
                        logger.info("[Summarization] Queue empty, background summarizer idle")
                        await asyncio.sleep(5.0)
                    continue
                
                priority, codebase_name, file_path, codebase_path = file_data
                
                # Update current file
                self._summarization_stats["current_file"] = file_path
                
                try:
                    success, processing_time = await self._process_single_file_for_summary(
                        codebase_name, codebase_path, file_path
                    )
                    
                    if success:
                        self._summarization_stats["files_summarized"] += 1
                        
                        # Update timing statistics
                        self._summarization_stats["total_time_seconds"] += processing_time
                        completed_count = self._summarization_stats["files_summarized"]
                        self._summarization_stats["avg_time_per_file"] = (
                            self._summarization_stats["total_time_seconds"] / completed_count
                        )
                        
                        # Calculate estimated time remaining (sequential)
                        remaining_files = self._summary_queue.qsize()
                        avg_time_with_rate_limit = (
                            self._summarization_stats["avg_time_per_file"] + 
                            self._summarization_config.rate_limit_seconds
                        )
                        est_time = avg_time_with_rate_limit * remaining_files
                        self._summarization_stats["estimated_time_remaining"] = est_time
                        
                        # Log progress periodically
                        if completed_count % 10 == 0:
                            failed = self._summarization_stats["files_failed"]
                            skipped = self._summarization_stats.get("files_skipped", 0)
                            avg_time = self._summarization_stats["avg_time_per_file"]
                            
                            logger.info(
                                f"[Summarization] Progress: {completed_count} completed, {failed} failed, {skipped} skipped, "
                                f"{remaining_files} remaining | Avg: {avg_time:.1f}s/file, Est. remaining: {est_time/60:.1f}min"
                            )
                    
                except Exception as e:
                    logger.error(f"[Summarization] Error processing {file_path}: {e}")
                    self._summarization_stats["last_error"] = str(e)
                finally:
                    self._summary_queue.task_done()
                    # Rate limit after each file
                    await asyncio.sleep(self._summarization_config.rate_limit_seconds)
                
            except asyncio.CancelledError:
                logger.info("[Summarization] Background summarizer stopped")
                self._summarization_stats["is_running"] = False
                self._summarization_stats["current_file"] = None
                break
            except Exception as e:
                logger.error(f"[Summarization] Error in background summarizer: {e}")
                self._summarization_stats["last_error"] = str(e)
                await asyncio.sleep(1.0)  # Brief pause before retrying
    
    def _background_summarizer_thread(self) -> None:
        """
        Background thread that summarizes files in priority order.
        
        This runs in a separate thread with its own event loop to avoid
        blocking the main async event loop. This is critical for API responsiveness.
        """
        logger.info("[Summarization] Background summarizer thread started")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while not self._summarizer_stop_event.is_set():
                try:
                    # Get next file from thread-safe queue (with timeout to check stop event)
                    try:
                        file_data = self._summary_queue_threaded.get(timeout=5.0)
                    except queue.Empty:
                        # No files in queue, check if we should stop
                        if self._summary_queue_threaded.empty():
                            logger.debug("[Summarization] Queue empty, thread idle")
                        continue
                    
                    priority, codebase_name, file_path, codebase_path = file_data
                    
                    # Update current file
                    self._summarization_stats["current_file"] = file_path
                    
                    try:
                        # Run the async processing in this thread's event loop
                        success, processing_time = loop.run_until_complete(
                            self._process_single_file_for_summary(
                                codebase_name, codebase_path, file_path
                            )
                        )
                        
                        if success:
                            self._summarization_stats["files_summarized"] += 1
                            
                            # Update timing statistics
                            self._summarization_stats["total_time_seconds"] += processing_time
                            completed_count = self._summarization_stats["files_summarized"]
                            self._summarization_stats["avg_time_per_file"] = (
                                self._summarization_stats["total_time_seconds"] / completed_count
                            )
                            
                            # Calculate estimated time remaining
                            remaining_files = self._summary_queue_threaded.qsize()
                            avg_time_with_rate_limit = (
                                self._summarization_stats["avg_time_per_file"] + 
                                self._summarization_config.rate_limit_seconds
                            )
                            est_time = avg_time_with_rate_limit * remaining_files
                            self._summarization_stats["estimated_time_remaining"] = est_time
                            
                            # Log progress periodically
                            if completed_count % 10 == 0:
                                failed = self._summarization_stats["files_failed"]
                                skipped = self._summarization_stats.get("files_skipped", 0)
                                avg_time = self._summarization_stats["avg_time_per_file"]
                                
                                logger.info(
                                    f"[Summarization] Progress: {completed_count} completed, {failed} failed, {skipped} skipped, "
                                    f"{remaining_files} remaining | Avg: {avg_time:.1f}s/file, Est. remaining: {est_time/60:.1f}min"
                                )
                        
                    except Exception as e:
                        logger.error(f"[Summarization] Error processing {file_path}: {e}")
                        self._summarization_stats["last_error"] = str(e)
                    finally:
                        self._summary_queue_threaded.task_done()
                        # Rate limit after each file
                        time.sleep(self._summarization_config.rate_limit_seconds)
                    
                except Exception as e:
                    logger.error(f"[Summarization] Error in background summarizer thread: {e}")
                    self._summarization_stats["last_error"] = str(e)
                    time.sleep(1.0)  # Brief pause before retrying
        finally:
            loop.close()
            self._summarization_stats["is_running"] = False
            self._summarization_stats["current_file"] = None
            logger.info("[Summarization] Background summarizer thread stopped")
    
    async def _process_single_file_for_summary(
        self,
        codebase_name: str,
        codebase_path: str,
        file_path: str
    ) -> tuple[bool, float]:
        """
        Process a single file for summarization.
        
        Returns:
            Tuple of (success: bool, processing_time: float)
        """
        import time
        
        # Read file content
        full_path = Path(codebase_path) / file_path
        if not full_path.exists():
            logger.debug(f"[Summarization] Skipping missing file: {file_path}")
            self._summarization_stats["files_skipped_empty"] += 1
            return (False, 0.0)
        
        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"[Summarization] Failed to read {file_path}: {e}")
            self._summarization_stats["files_skipped_empty"] += 1
            return (False, 0.0)
        
        if not content.strip():
            self._summarization_stats["files_skipped_empty"] += 1
            return (False, 0.0)
        
        # Compute content hash for incremental tracking
        content_hash = ChromaVectorStore.compute_content_hash(content)
        
        # Check if file needs (re-)summarization
        vector_store = self._vector_stores.get(codebase_name)
        if vector_store and not vector_store.summary_index.needs_resummarization(file_path, content_hash):
            logger.debug(f"[Summarization] Skipping (unchanged): {file_path}")
            self._summarization_stats["files_skipped"] = self._summarization_stats.get("files_skipped", 0) + 1
            self._summarization_stats["files_skipped_unchanged"] += 1
            return (False, 0.0)
        
        # Get heuristic metadata if available
        heuristic_metadata = None
        try:
            heuristic_metadata = self.heuristic_extractor.extract_file_metadata(file_path, content)
        except Exception:
            pass  # Heuristics are optional
        
        # Generate summary with timing
        start_time = time.time()
        logger.debug(f"[Summarization] Processing: {file_path}")
        summary = await self._summarizer.summarize_file(file_path, content, heuristic_metadata)
        end_time = time.time()
        processing_time = end_time - start_time
        
        if summary.error:
            self._summarization_stats["files_failed"] += 1
            self._summarization_stats["last_error"] = f"{file_path}: {summary.error}"
            logger.warning(f"[Summarization] Failed to summarize {file_path}: {summary.error}")
            return (False, processing_time)
        else:
            # Store summary with content hash for tracking
            await self._store_summary(codebase_name, file_path, summary, content_hash)
            return (True, processing_time)
    
    async def _store_summary(
        self, 
        codebase: str, 
        file_path: str, 
        summary: FileSummary,
        content_hash: Optional[str] = None
    ) -> None:
        """
        Store a file summary as a searchable memory chunk.
        
        The summary is stored in two places:
        1. Summary index (SummaryIndexMetadata) - Full structured JSON for retrieval
        2. Main collection - Searchable text chunk with embeddings
        
        Args:
            codebase: Codebase name
            file_path: Path to the file
            summary: FileSummary with LLM-generated summary
            content_hash: Optional hash of file content for tracking changes
        """
        vector_store = self._vector_stores.get(codebase)
        if not vector_store:
            logger.warning(f"[Summarization] Vector store not found for codebase: {codebase}")
            return
        
        # Check if there's an existing summary to remove from main collection
        existing_info = await self._run_blocking(
            vector_store.summary_index.get_summary_info, file_path
        )
        if existing_info:
            old_chunk_id = existing_info.get("summary_chunk_id")
            if old_chunk_id:
                try:
                    await self._run_blocking(vector_store.delete, old_chunk_id)
                    logger.debug(f"[Summarization] Removed old summary for: {file_path}")
                except Exception as e:
                    logger.debug(f"[Summarization] Failed to remove old summary: {e}")
        
        # --- Store structured JSON in summary index ---
        # This enables structured retrieval via get_file_summary_async()
        summary_dict = summary.to_dict()
        summary_dict["summarized_at"] = datetime.now().isoformat()
        
        await self._run_blocking(
            vector_store.summary_index.store_summary,
            file_path=file_path,
            summary_data=summary_dict,
            content_hash=content_hash or ""
        )
        
        # --- Store searchable text in main collection ---
        # Format summary as searchable text for vector search
        key_exports = ", ".join(summary.key_exports) if summary.key_exports else "none"
        dependencies = ", ".join(summary.dependencies) if summary.dependencies else "none"
        
        summary_text = f"""File Summary: {file_path}
Language: {summary.language}
Pattern: {summary.pattern}
Domain: {summary.domain}

Purpose: {summary.purpose}

Key exports: {key_exports}
Dependencies: {dependencies}"""
        
        # Phase 2: Add implementation-aware fields if present
        if summary.how_it_works:
            summary_text += f"\n\nHow it works: {summary.how_it_works}"
        
        if summary.key_mechanisms:
            mechanisms = ", ".join(summary.key_mechanisms)
            summary_text += f"\n\nKey mechanisms: {mechanisms}"
        
        if summary.method_summaries:
            summary_text += "\n\nMethod summaries:"
            for method_name, method_desc in summary.method_summaries.items():
                summary_text += f"\n  - {method_name}: {method_desc}"
        
        # Build tags for filtering
        tags = [
            "summary",
            f"file:{file_path}",
            f"pattern:{summary.pattern.lower()}",
            f"domain:{summary.domain}",
            f"lang:{summary.language}"
        ]
        
        # Add key exports as tags for searchability
        for export in summary.key_exports[:5]:  # Limit to first 5
            tags.append(f"export:{export}")
        
        # Phase 2: Add mechanism tags for searchability (e.g., mechanism:caching, mechanism:retry)
        if summary.key_mechanisms:
            for mechanism in summary.key_mechanisms[:10]:  # Limit to first 10
                # Normalize mechanism name for tag format (lowercase, replace spaces with hyphens)
                mechanism_tag = mechanism.lower().replace(" ", "-").replace("_", "-")
                tags.append(f"mechanism:{mechanism_tag}")
        
        # Store searchable chunk with pinning so summaries aren't pruned
        result = await self.store_async(
            content=summary_text,
            codebase=codebase,
            role="system",
            tags=tags,
            pin=True,
            source="llm_summarization",
            memory_type="code"  # Treat as code-related memory
        )
        
        # Update summary index with the chunk ID for backward compatibility
        # This links the searchable chunk to the structured summary
        if result.get("success") and result.get("id") and content_hash:
            await self._run_blocking(
                vector_store.summary_index.update_summary_info,
                file_path=file_path,
                content_hash=content_hash,
                summary_chunk_id=result["id"],
                model=summary.model_used,
                pattern=summary.pattern,
                domain=summary.domain
            )
        
        logger.debug(f"[Summarization] Stored summary for: {file_path}")

        # Update count cache (only increment if this was a new summary, not an update)
        if not existing_info:
            self._update_count_cache(codebase, summarized_delta=1)

    async def stop_background_summarizer_async(self) -> None:
        """Stop the background summarization task/thread."""
        # Stop the thread if running
        if self._summarizer_thread and self._summarizer_thread.is_alive():
            logger.info("[Summarization] Stopping background summarizer thread...")
            self._summarizer_stop_event.set()
            self._summarizer_thread.join(timeout=10.0)
            if self._summarizer_thread.is_alive():
                logger.warning("[Summarization] Thread did not stop gracefully")
        
        # Also handle legacy async task if present
        if self._summarizer_task and not self._summarizer_task.done():
            self._summarizer_task.cancel()
            try:
                await self._summarizer_task
            except asyncio.CancelledError:
                pass
        
        if self._summarizer and hasattr(self._summarizer, 'llm_client'):
            await self._summarizer.llm_client.close()
        
        self._summarization_stats["is_running"] = False
        self._summarization_stats["current_file"] = None
        logger.info("[Summarization] Background summarizer stopped")
    
    def _reset_session_stats_if_idle(self) -> None:
        """
        Reset session stats when transitioning from idle to active.
        
        This ensures the "N Items Summarized" counter reflects work done
        since the last time the queue was empty, not cumulative session totals.
        """
        queue_size = self._summary_queue_threaded.qsize() if self._summary_queue_threaded else 0
        if queue_size == 0:
            # Queue is empty, reset session counters for next batch
            self._summarization_stats["files_summarized"] = 0
            self._summarization_stats["files_failed"] = 0
            self._summarization_stats["files_skipped"] = 0
            self._summarization_stats["files_skipped_pattern"] = 0
            self._summarization_stats["files_skipped_unchanged"] = 0
            self._summarization_stats["files_skipped_empty"] = 0
            self._summarization_stats["files_total_queued"] = 0
            self._summarization_stats["total_time_seconds"] = 0.0
            self._summarization_stats["avg_time_per_file"] = 0.0
            self._summarization_stats["estimated_time_remaining"] = 0.0
            self._summarization_stats["last_error"] = None
            logger.debug("[Summarization] Reset session stats for new batch")
    
    def get_summarization_status(self) -> Dict[str, Any]:
        """
        Get the current status of background summarization.
        
        Returns:
            Dictionary with summarization status and progress
        """
        # Get summary stats from all codebases
        total_summarized = 0
        total_simple = 0
        total_llm = 0
        summary_stats_by_codebase = {}
        
        for codebase_name, vector_store in self._vector_stores.items():
            try:
                stats = vector_store.summary_index.get_summary_stats()
                summary_stats_by_codebase[codebase_name] = stats
                total_summarized += stats.get("total_summarized", 0)
                total_simple += stats.get("simple_count", 0)
                total_llm += stats.get("llm_count", 0)
            except Exception as e:
                logger.debug(f"[{codebase_name}] Error getting summary stats: {e}")
        
        # Use thread-safe queue size (the threaded queue is what's actually being processed)
        files_queued = self._summary_queue_threaded.qsize() if self._summary_queue_threaded else 0
        
        # Calculate timing estimates (including rate limit delay)
        avg_time_per_file = self._summarization_stats.get("avg_time_per_file", 0.0)
        avg_time_with_rate_limit = avg_time_per_file + self._summarization_config.rate_limit_seconds
        estimated_time_remaining = avg_time_with_rate_limit * files_queued
        
        # Calculate progress
        files_completed = self._summarization_stats["files_summarized"]
        files_failed = self._summarization_stats["files_failed"]
        files_skipped_total = self._summarization_stats.get("files_skipped", 0)
        files_total_queued = self._summarization_stats.get("files_total_queued", 0)
        
        # Progress calculation: (completed + failed + skipped) / total_originally_queued
        files_processed = files_completed + files_failed + files_skipped_total
        progress_percentage = (files_processed / files_total_queued * 100.0) if files_total_queued > 0 else 0.0
        
        return {
            "enabled": self._summarization_config.enabled,
            "llm_enabled": self._summarization_config.llm_enabled,
            "model": self._summarization_config.model,
            "is_running": self._summarization_stats["is_running"],
            "files_queued": files_queued,
            "files_failed": files_failed,
            "files_total_queued": files_total_queued,
            "files_processed": files_processed,
            "progress_percentage": progress_percentage,
            # Summary breakdown: total vs simple vs LLM
            "total_summarized": total_summarized,
            "simple_count": total_simple,
            "llm_count": total_llm,
            "current_file": self._summarization_stats["current_file"],
            "last_error": self._summarization_stats["last_error"],
            "by_codebase": summary_stats_by_codebase,
            # Timing estimates
            "avg_time_per_file_seconds": avg_time_per_file,
            "estimated_time_remaining_seconds": estimated_time_remaining,
            "estimated_time_remaining_minutes": estimated_time_remaining / 60.0,
            "total_processing_time_seconds": self._summarization_stats.get("total_time_seconds", 0.0)
        }
    
    def get_summarization_status_basic(self) -> Dict[str, Any]:
        """
        Get basic summarization status (fast, uses in-memory cache).

        This version uses the in-memory count cache instead of querying
        ChromaDB. Suitable for health checks and dashboard polling.

        Returns:
            Dictionary with basic summarization status including timing info
        """
        # Initialize cache if needed (one-time startup cost)
        if not self._count_cache_initialized:
            self._initialize_count_cache()

        # Get summary counts from in-memory cache (instant)
        total_summarized = 0
        summary_stats_by_codebase = {}

        for codebase_name in self._vector_stores.keys():
            counts = self._count_cache.get(codebase_name, {})
            summarized = counts.get("summarized", 0)
            total_summarized += summarized
            summary_stats_by_codebase[codebase_name] = {
                "total_summarized": summarized,
                "simple_count": None,  # Not tracked in cache
                "llm_count": None,
            }

        # simple/llm breakdown not available from cache
        total_simple = 0
        total_llm = 0
        
        # Use thread-safe queue size (the threaded queue is what's actually being processed)
        files_queued = self._summary_queue_threaded.qsize() if self._summary_queue_threaded else 0
        
        # Timing info from in-memory stats
        files_completed = self._summarization_stats["files_summarized"]
        files_failed = self._summarization_stats["files_failed"]
        files_total_queued = self._summarization_stats.get("files_total_queued", 0)
        
        files_processed = files_completed + files_failed + self._summarization_stats.get("files_skipped", 0)
        progress_percentage = (files_processed / files_total_queued * 100.0) if files_total_queued > 0 else 0.0
        
        # Calculate timing estimates
        avg_time_per_file = self._summarization_stats.get("avg_time_per_file", 0.0)
        avg_time_with_rate_limit = avg_time_per_file + self._summarization_config.rate_limit_seconds
        estimated_time_remaining = avg_time_with_rate_limit * files_queued
        
        return {
            "enabled": self._summarization_config.enabled,
            "llm_enabled": self._summarization_config.llm_enabled,
            "model": self._summarization_config.model if self._summarization_config else None,
            "is_running": self._summarization_stats["is_running"],
            "files_queued": files_queued,
            "files_failed": files_failed,
            "files_processed": files_processed,
            "files_total_queued": files_total_queued,
            "progress_percentage": progress_percentage,
            # Summary breakdown
            "total_summarized": total_summarized,
            "simple_count": total_simple,
            "llm_count": total_llm,
            "current_file": self._summarization_stats["current_file"],
            "last_error": self._summarization_stats["last_error"],
            "by_codebase": summary_stats_by_codebase,
            # Timing estimates
            "avg_time_per_file_seconds": avg_time_per_file,
            "estimated_time_remaining_seconds": estimated_time_remaining,
            "estimated_time_remaining_minutes": estimated_time_remaining / 60.0,
            "total_processing_time_seconds": self._summarization_stats.get("total_time_seconds", 0.0)
        }

    def _initialize_count_cache(self) -> None:
        """
        Initialize count cache from ChromaDB (called once at startup).

        This queries ChromaDB once to populate the in-memory cache.
        Subsequent operations update the cache incrementally.
        """
        if self._count_cache_initialized:
            return

        logger.info("Initializing count cache from ChromaDB...")
        start = time.time()

        for codebase_name, vector_store in self._vector_stores.items():
            try:
                indexed = vector_store.file_index.collection.count()
                summarized = vector_store.summary_index.get_summary_count()
                self._count_cache[codebase_name] = {
                    "indexed": indexed,
                    "summarized": summarized,
                }
            except Exception as e:
                logger.warning(f"Failed to get counts for {codebase_name}: {e}")
                self._count_cache[codebase_name] = {"indexed": 0, "summarized": 0}

        self._count_cache_initialized = True
        elapsed = (time.time() - start) * 1000
        logger.info(f"Count cache initialized in {elapsed:.0f}ms")

    def _update_count_cache(self, codebase: str, indexed_delta: int = 0, summarized_delta: int = 0) -> None:
        """Update count cache incrementally (call when files are added/removed)."""
        if codebase not in self._count_cache:
            self._count_cache[codebase] = {"indexed": 0, "summarized": 0}

        self._count_cache[codebase]["indexed"] += indexed_delta
        self._count_cache[codebase]["summarized"] += summarized_delta

    def get_status_counts(self) -> Dict[str, Any]:
        """
        Get ultra-lightweight status counts for dashboard polling.

        Returns counts from in-memory cache (instant, never queries ChromaDB).
        Cache is populated once at startup and updated incrementally.

        Returns:
            Dictionary with minimal status info for polling
        """
        start = time.time()

        # Initialize cache if needed (one-time startup cost)
        if not self._count_cache_initialized:
            logger.info("get_status_counts: cache not initialized, initializing...")
            self._initialize_count_cache()
        else:
            logger.debug("get_status_counts: using cached counts")

        # Sum counts from cache (instant - no ChromaDB queries)
        total_indexed = sum(c.get("indexed", 0) for c in self._count_cache.values())
        total_summarized = sum(c.get("summarized", 0) for c in self._count_cache.values())

        elapsed = (time.time() - start) * 1000
        logger.info(f"get_status_counts completed in {elapsed:.0f}ms (cached={self._count_cache_initialized})")

        # Queue stats from in-memory (instant, always fresh)
        files_queued = self._summary_queue_threaded.qsize() if self._summary_queue_threaded else 0

        return {
            "total_indexed": total_indexed,
            "total_summarized": total_summarized,
            "files_queued": files_queued,
            "is_running": self._summarization_stats["is_running"],
            "current_file": self._summarization_stats["current_file"],
            "files_completed": self._summarization_stats["files_summarized"],
            "files_failed": self._summarization_stats["files_failed"],
        }

    async def queue_file_for_summarization(
        self, 
        codebase_name: str, 
        file_path: str, 
        codebase_path: str,
        priority: float = 0.5
    ) -> bool:
        """
        Queue a single file for (re-)summarization.
        
        Called by the file watcher when a file is modified.
        
        Args:
            codebase_name: Name of the codebase
            file_path: Relative path to the file
            codebase_path: Absolute path to the codebase root
            priority: Priority score (higher = process sooner, default 0.5)
            
        Returns:
            True if file was queued, False if summarization is not running
        """
        if not self._summarization_stats["is_running"]:
            return False
        
        if not self._summarizer:
            return False
        
        # Check if file should be skipped
        if self._summarizer._should_skip_file(file_path):
            return False
        
        # Reset session stats if queue was empty (starting new batch)
        self._reset_session_stats_if_idle()
        
        # Queue with negative priority (PriorityQueue uses min-first)
        # Use the thread-safe queue since the summarizer runs in a separate thread
        self._summary_queue_threaded.put((-priority, codebase_name, file_path, codebase_path))
        # Update total queued for individual file additions
        self._summarization_stats["files_total_queued"] += 1
        
        logger.info(f"[Summarization] Queued for re-summarization: {file_path}")
        return True
    
    async def remove_file_summary(self, codebase_name: str, file_path: str) -> bool:
        """
        Remove the summary for a deleted file.
        
        Args:
            codebase_name: Name of the codebase
            file_path: Relative path to the file
            
        Returns:
            True if summary was removed
        """
        vector_store = self._vector_stores.get(codebase_name)
        if not vector_store:
            return False
        
        chunk_id = vector_store.summary_index.delete_summary_info(file_path)
        if chunk_id:
            try:
                vector_store.delete(chunk_id)
                logger.info(f"[Summarization] Removed summary for deleted file: {file_path}")
                return True
            except Exception as e:
                logger.debug(f"[Summarization] Failed to remove summary chunk: {e}")
        
        return False
    
    async def invalidate_summaries_async(
        self,
        codebase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invalidate all existing summaries to force re-summarization.
        
        This clears:
        1. All entries from the summary index (SummaryIndexMetadata)
        2. All summary chunks from the main collection (source="llm_summarization")
        
        After invalidation, summaries will be regenerated on the next summarization run.
        
        Args:
            codebase: Optional codebase name to invalidate (None = all codebases)
            
        Returns:
            Dict with invalidation results including counts
        """
        if not self._vector_stores:
            return {"error": "No codebases configured", "success": False}
        
        results = {
            "success": True,
            "codebases": {},
            "total_summaries_cleared": 0,
            "total_chunks_removed": 0
        }
        
        # Determine which codebases to process
        if codebase:
            stores_to_process = {codebase: self._vector_stores.get(codebase)}
            if not stores_to_process[codebase]:
                return {"error": f"Codebase not found: {codebase}", "success": False}
        else:
            stores_to_process = self._vector_stores
        
        for codebase_name, vector_store in stores_to_process.items():
            if not vector_store:
                continue
            
            codebase_result = {
                "summaries_cleared": 0,
                "chunks_removed": 0
            }
            
            try:
                # Step 1: Clear the summary index
                summaries_cleared = vector_store.summary_index.clear_all_summaries()
                codebase_result["summaries_cleared"] = summaries_cleared
                results["total_summaries_cleared"] += summaries_cleared
                
                # Step 2: Remove summary chunks from main collection
                # These are stored with source="llm_summarization"
                try:
                    summary_results = vector_store.collection.get(
                        where={"source": "llm_summarization"},
                        include=[]  # Only need IDs
                    )
                    
                    if summary_results["ids"]:
                        chunk_ids = summary_results["ids"]
                        # Delete in batches
                        BATCH_SIZE = 500
                        for i in range(0, len(chunk_ids), BATCH_SIZE):
                            batch = chunk_ids[i:i + BATCH_SIZE]
                            vector_store.collection.delete(ids=batch)
                        
                        codebase_result["chunks_removed"] = len(chunk_ids)
                        results["total_chunks_removed"] += len(chunk_ids)
                        logger.info(f"[{codebase_name}] Removed {len(chunk_ids)} summary chunks from main collection")
                        
                except Exception as e:
                    logger.warning(f"[{codebase_name}] Error removing summary chunks: {e}")
                
                # Step 3: Update schema version to current
                from ..storage.chroma import SUMMARY_SCHEMA_VERSION
                vector_store.summary_index.set_schema_version(SUMMARY_SCHEMA_VERSION)
                
                results["codebases"][codebase_name] = codebase_result
                logger.info(
                    f"[{codebase_name}] Invalidated {summaries_cleared} summaries, "
                    f"removed {codebase_result['chunks_removed']} chunks"
                )
                
            except Exception as e:
                logger.error(f"[{codebase_name}] Error invalidating summaries: {e}")
                codebase_result["error"] = str(e)
                results["codebases"][codebase_name] = codebase_result
        
        return results
    
    def invalidate_summaries(self, codebase: Optional[str] = None) -> Dict[str, Any]:
        """Sync wrapper for invalidate_summaries_async"""
        return _run_sync(self.invalidate_summaries_async(codebase))
    
    async def queue_codebase_for_summarization(
        self, 
        codebase_name: str,
        only_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Queue all files from a specific codebase for summarization.
        
        This is useful when:
        - A new codebase is added to config
        - You want to re-summarize an existing codebase
        - Summarization was interrupted
        
        Args:
            codebase_name: Name of the codebase to queue
            only_missing: If True, only queue files without summaries (default)
                         If False, queue all files for re-summarization
            
        Returns:
            Dict with status info: {
                "success": bool,
                "message": str,
                "files_queued": int,
                "files_skipped": int
            }
        """
        if not self._summarization_stats["is_running"]:
            return {
                "success": False,
                "message": "Summarization is not running. Check that summarization.enabled=true and LLM is available.",
                "files_queued": 0,
                "files_skipped": 0
            }
        
        if not self._summarizer:
            return {
                "success": False,
                "message": "Summarizer not initialized",
                "files_queued": 0,
                "files_skipped": 0
            }
        
        # Find the codebase config
        codebase = None
        for cb in self.config.get_enabled_codebases():
            if cb.name == codebase_name:
                codebase = cb
                break
        
        if not codebase:
            return {
                "success": False,
                "message": f"Codebase not found: {codebase_name}",
                "files_queued": 0,
                "files_skipped": 0
            }
        
        vector_store = self._vector_stores.get(codebase_name)
        if not vector_store:
            return {
                "success": False,
                "message": f"Vector store not initialized for: {codebase_name}",
                "files_queued": 0,
                "files_skipped": 0
            }
        
        codebase_path = Path(codebase.path)
        files_queued = 0
        files_skipped = 0
        
        # Reset session stats if queue was empty (starting new batch)
        self._reset_session_stats_if_idle()
        
        # Get existing summaries if only_missing is True
        summarized_files = set()
        if only_missing:
            summarized_files = vector_store.summary_index.get_all_summarized_files()
            logger.info(f"[Summarization] {codebase_name} already has {len(summarized_files)} summaries")
        
        # Try to get files by centrality first
        priority_queue = self.get_file_centrality_scores(codebase_name, max_files=10000)
        
        if priority_queue:
            logger.info(f"[Summarization] Using centrality ordering for {codebase_name}, found {len(priority_queue)} files")
            for file_path, score in priority_queue:
                # Skip if file matches skip pattern
                if self._summarizer._should_skip_file(file_path):
                    files_skipped += 1
                    self._summarization_stats["files_skipped_pattern"] += 1
                    continue
                
                # Skip if already summarized and only_missing=True
                if only_missing and file_path in summarized_files:
                    files_skipped += 1
                    continue
                
                # Use thread-safe queue since summarizer runs in separate thread
                self._summary_queue_threaded.put((-score, codebase_name, file_path, str(codebase_path)))
                files_queued += 1
        else:
            # Fallback: queue all indexed files with equal priority
            indexed_files = vector_store.get_indexed_files()
            logger.info(f"[Summarization] No import graph for {codebase_name}, using {len(indexed_files)} indexed files")
            
            for file_path in indexed_files.keys():
                # Skip if file matches skip pattern
                if self._summarizer._should_skip_file(file_path):
                    files_skipped += 1
                    self._summarization_stats["files_skipped_pattern"] += 1
                    continue
                
                # Skip if already summarized and only_missing=True  
                if only_missing and file_path in summarized_files:
                    files_skipped += 1
                    continue
                
                # Use thread-safe queue since summarizer runs in separate thread
                self._summary_queue_threaded.put((0.5, codebase_name, file_path, str(codebase_path)))
                files_queued += 1
        
        # Update total queued if this is adding new files
        if files_queued > 0:
            self._summarization_stats["files_total_queued"] += files_queued
        
        logger.info(
            f"[Summarization] Queued {files_queued} files from {codebase_name} "
            f"({files_skipped} skipped)"
        )
        
        return {
            "success": True,
            "message": f"Queued {files_queued} files from {codebase_name} for summarization",
            "files_queued": files_queued,
            "files_skipped": files_skipped,
            "total_queue_size": self._summary_queue_threaded.qsize()
        }
    
    # Sync wrapper
    def queue_codebase_for_summarization_sync(
        self, 
        codebase_name: str,
        only_missing: bool = True
    ) -> Dict[str, Any]:
        """Sync wrapper for queue_codebase_for_summarization"""
        return _run_sync(self.queue_codebase_for_summarization(codebase_name, only_missing))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Browse/List API - For Web Dashboard
    # ─────────────────────────────────────────────────────────────────────────
    
    async def list_indexed_files_async(
        self,
        codebase: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        search_filter: Optional[str] = None,
        has_summary: Optional[bool] = None,
        pattern: Optional[str] = None,
        domain: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List indexed files with pagination and filtering support.

        Args:
            codebase: Codebase name (None = first available)
            limit: Maximum number of results
            offset: Offset for pagination
            search_filter: Optional filename filter (substring match)
            has_summary: Filter by summary presence (True/False/None)
            pattern: Filter by pattern from summary (e.g., "service", "utility")
            domain: Filter by domain from summary (e.g., "api", "database")
            language: Filter by programming language

        Returns:
            Dict with files list, total count, and pagination info
        """
        if not self._vector_stores:
            return {"error": "No codebases configured", "files": [], "total": 0}

        # FAST PATH: Use PostgreSQL if available (sub-50ms queries)
        if not search_filter and await self._ensure_postgres():
            try:
                files, total = await self._postgres.get_files_paginated(
                    codebase=codebase,
                    limit=limit,
                    offset=offset,
                    has_summary=has_summary,
                    pattern=pattern,
                    domain=domain
                )
                return {
                    "files": files,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "codebase": codebase or next(iter(self._vector_stores.keys())),
                    "source": "postgresql"
                }
            except Exception as e:
                logger.warning(f"PostgreSQL query failed, falling back to ChromaDB: {e}")
        
        # Get target codebase
        if codebase:
            vector_store = self._vector_stores.get(codebase)
            if not vector_store:
                return {"error": f"Codebase not found: {codebase}", "files": [], "total": 0}
        else:
            # Use first available codebase
            codebase = next(iter(self._vector_stores.keys()))
            vector_store = self._vector_stores[codebase]
        
        try:
            # Check if any filters are applied
            has_filters = any([search_filter, has_summary is not None, pattern, domain, language])

            if not has_filters:
                # FAST PATH: No filters - use database-level pagination
                paginated_files, total = vector_store.file_index.get_indexed_files_paginated(
                    limit=limit, offset=offset
                )

                # Get summary data for just these files (cached, fast lookup)
                summarized_files = vector_store.summary_index.get_all_summarized_files()

                files = []
                for file_data in paginated_files:
                    file_path = file_data["path"]
                    file_has_summary = file_path in summarized_files
                    summary_data = summarized_files.get(file_path, {})

                    chunk_ids_str = file_data.get("chunk_ids", "")
                    chunk_count = len(chunk_ids_str.split(",")) if chunk_ids_str else 0

                    files.append({
                        "path": file_path,
                        "mtime": file_data.get("mtime"),
                        "content_hash": file_data.get("content_hash", "")[:12] + "...",
                        "chunk_count": chunk_count,
                        "indexed_at": file_data.get("indexed_at"),
                        "has_summary": file_has_summary,
                        "pattern": summary_data.get("pattern"),
                        "domain": summary_data.get("domain"),
                        "simple_file": summary_data.get("simple_file", False),
                        "simple_file_reason": summary_data.get("simple_file_reason")
                    })
            else:
                # FILTERED PATH: Need to load all and filter in Python
                indexed_files = vector_store.file_index.get_all_indexed_files()
                summarized_files = vector_store.summary_index.get_all_summarized_files()

                files = []
                for file_path, metadata in indexed_files.items():
                    # Apply search filter if provided
                    if search_filter and search_filter.lower() not in file_path.lower():
                        continue

                    file_has_summary = file_path in summarized_files
                    summary_data = summarized_files.get(file_path, {}) if file_has_summary else {}

                    # Apply has_summary filter
                    if has_summary is not None:
                        if has_summary and not file_has_summary:
                            continue
                        if not has_summary and file_has_summary:
                            continue

                    # Apply pattern filter (from summary)
                    if pattern:
                        file_pattern = summary_data.get("pattern", "")
                        if pattern.lower() != file_pattern.lower():
                            continue

                    # Apply domain filter (from summary)
                    if domain:
                        file_domain = summary_data.get("domain", "")
                        if domain.lower() != file_domain.lower():
                            continue

                    # Apply language filter (from file extension)
                    if language:
                        ext_to_lang = {
                            ".py": "python",
                            ".kt": "kotlin",
                            ".java": "java",
                            ".ts": "typescript",
                            ".tsx": "typescript",
                            ".js": "javascript",
                            ".jsx": "javascript",
                        }
                        file_ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                        file_language = ext_to_lang.get(file_ext, "")
                        if language.lower() != file_language:
                            continue

                    chunk_ids_str = metadata.get("chunk_ids", "")
                    chunk_count = len(chunk_ids_str.split(",")) if chunk_ids_str else 0

                    files.append({
                        "path": file_path,
                        "mtime": metadata.get("mtime"),
                        "content_hash": metadata.get("content_hash", "")[:12] + "...",
                        "chunk_count": chunk_count,
                        "indexed_at": metadata.get("indexed_at"),
                        "has_summary": file_has_summary,
                        "pattern": summary_data.get("pattern"),
                        "domain": summary_data.get("domain"),
                        "simple_file": summary_data.get("simple_file", False),
                        "simple_file_reason": summary_data.get("simple_file_reason")
                    })

                # Sort by path
                files.sort(key=lambda f: f["path"])

                # Apply pagination
                total = len(files)
                files = files[offset:offset + limit]
            
            return {
                "codebase": codebase,
                "files": files,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
            
        except Exception as e:
            logger.error(f"Error listing indexed files: {e}")
            return {"error": str(e), "files": [], "total": 0}
    
    async def get_file_details_async(
        self,
        codebase: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific indexed file.
        
        Args:
            codebase: Codebase name
            file_path: Relative path to the file
            
        Returns:
            Dict with file metadata, chunks, and summary if available
        """
        vector_store = self._vector_stores.get(codebase)
        if not vector_store:
            return {"error": f"Codebase not found: {codebase}"}

        try:
            # Get file index metadata - try both path separator formats
            file_info = vector_store.file_index.get_file_info(file_path)
            if not file_info:
                # Try with normalized forward slashes
                normalized_path = file_path.replace("\\", "/")
                file_info = vector_store.file_index.get_file_info(normalized_path)
            if not file_info:
                # Try with backslashes
                normalized_path = file_path.replace("/", "\\")
                file_info = vector_store.file_index.get_file_info(normalized_path)
            if not file_info:
                return {"error": f"File not found in index: {file_path}"}
            
            # Get chunk IDs
            chunk_ids_str = file_info.get("chunk_ids", "")
            chunk_ids = chunk_ids_str.split(",") if chunk_ids_str else []
            
            # Fetch chunks from main collection
            chunks = []
            if chunk_ids:
                try:
                    result = vector_store.collection.get(
                        ids=chunk_ids,
                        include=["documents", "metadatas"]
                    )
                    
                    for i, chunk_id in enumerate(result["ids"]):
                        doc = result["documents"][i] if result["documents"] else ""
                        meta = result["metadatas"][i] if result["metadatas"] else {}
                        
                        chunks.append({
                            "id": chunk_id,
                            "content": doc,
                            "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                            "source": meta.get("source", ""),
                            "memory_type": meta.get("memory_type", "code")
                        })
                except Exception as e:
                    logger.warning(f"Error fetching chunks for {file_path}: {e}")
            
            # Get summary if available (use get_full_summary for Phase 2 fields)
            summary = None
            full_summary = vector_store.summary_index.get_full_summary(file_path)
            if full_summary:
                summary_chunk_id = full_summary.get("summary_chunk_id")
                if summary_chunk_id:
                    try:
                        summary_result = vector_store.collection.get(
                            ids=[summary_chunk_id],
                            include=["documents"]
                        )
                        if summary_result["ids"]:
                            summary = {
                                "content": summary_result["documents"][0] if summary_result["documents"] else "",
                                "model": full_summary.get("model_used", full_summary.get("model", "")),
                                "pattern": full_summary.get("pattern", ""),
                                "domain": full_summary.get("domain", ""),
                                "summarized_at": full_summary.get("summarized_at", ""),
                                "simple_file": full_summary.get("simple_file", False),
                                "simple_file_reason": full_summary.get("simple_file_reason"),
                                # Phase 2 fields
                                "purpose": full_summary.get("purpose", ""),
                                "how_it_works": full_summary.get("how_it_works"),
                                "key_mechanisms": full_summary.get("key_mechanisms"),
                                "method_summaries": full_summary.get("method_summaries"),
                                "exports": full_summary.get("exports"),
                            }
                    except Exception as e:
                        logger.warning(f"Error fetching summary for {file_path}: {e}")
            
            return {
                "codebase": codebase,
                "file_path": file_path,
                "mtime": file_info.get("mtime"),
                "content_hash": file_info.get("content_hash", ""),
                "indexed_at": file_info.get("indexed_at"),
                "chunk_count": len(chunks),
                "chunks": chunks,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error getting file details: {e}")
            return {"error": str(e)}
    
    async def list_memories_async(
        self,
        memory_type: Optional[str] = None,
        codebase: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List stored memories (conversations, decisions, lessons).
        
        Args:
            memory_type: Filter by type: "conversation", "decision", "lesson" (None = all non-code)
            codebase: Codebase name (None = search all)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Dict with memories list, total count, and pagination info
        """
        if not self._vector_stores:
            return {"error": "No codebases configured", "memories": [], "total": 0}
        
        try:
            all_memories = []
            
            # Determine which codebases to search
            if codebase:
                stores_to_search = {codebase: self._vector_stores.get(codebase)}
                if not stores_to_search[codebase]:
                    return {"error": f"Codebase not found: {codebase}", "memories": [], "total": 0}
            else:
                stores_to_search = self._vector_stores
            
            # Memory types to include (exclude 'code' type)
            valid_types = ["conversation", "decision", "lesson"]
            if memory_type and memory_type in valid_types:
                filter_types = [memory_type]
            else:
                filter_types = valid_types
            
            # Query each codebase
            for codebase_name, vector_store in stores_to_search.items():
                if not vector_store:
                    continue
                
                for mtype in filter_types:
                    try:
                        # Query by memory_type
                        result = vector_store.collection.get(
                            where={"memory_type": mtype},
                            include=["documents", "metadatas"]
                        )
                        
                        if result["ids"]:
                            for i, memory_id in enumerate(result["ids"]):
                                doc = result["documents"][i] if result["documents"] else ""
                                meta = result["metadatas"][i] if result["metadatas"] else {}
                                
                                # Skip LLM summaries (they're stored with code type but have special source)
                                if meta.get("source") == "llm_summarization":
                                    continue
                                
                                all_memories.append({
                                    "id": memory_id,
                                    "type": mtype,
                                    "codebase": codebase_name,
                                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                                    "content": doc,
                                    "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                                    "source": meta.get("source", ""),
                                    "created_at": meta.get("created_at", ""),
                                    "role": meta.get("role", "")
                                })
                    except Exception as e:
                        logger.debug(f"Error querying memories from {codebase_name}: {e}")
            
            # Sort by created_at (newest first)
            all_memories.sort(key=lambda m: m.get("created_at", ""), reverse=True)
            
            # Apply pagination
            total = len(all_memories)
            memories = all_memories[offset:offset + limit]
            
            return {
                "memories": memories,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
                "filter_type": memory_type
            }
            
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return {"error": str(e), "memories": [], "total": 0}
    
    async def list_summaries_async(
        self,
        codebase: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List files that have LLM-generated summaries.

        Args:
            codebase: Codebase name (None = first available)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            Dict with summaries list, total count, and pagination info
        """
        if not self._vector_stores:
            return {"error": "No codebases configured", "summaries": [], "total": 0}

        # FAST PATH: Use PostgreSQL if available (sub-50ms queries)
        if await self._ensure_postgres():
            try:
                summaries, total = await self._postgres.get_summaries_paginated(
                    codebase=codebase,
                    limit=limit,
                    offset=offset
                )
                return {
                    "codebase": codebase or next(iter(self._vector_stores.keys())),
                    "summaries": [
                        {
                            "file_path": s.get("relative_path", s.get("file_path", "")),
                            "pattern": s.get("pattern", ""),
                            "domain": s.get("domain", ""),
                            "model": "llm",
                            "summarized_at": s.get("created_at", ""),
                            "content_preview": (s.get("summary_text", "")[:300] + "...")
                                if len(s.get("summary_text", "")) > 300
                                else s.get("summary_text", ""),
                            "content": s.get("summary_text", "")
                        }
                        for s in summaries
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total,
                    "source": "postgresql"
                }
            except Exception as e:
                logger.warning(f"PostgreSQL query failed, falling back to ChromaDB: {e}")

        # Get target codebase
        if codebase:
            vector_store = self._vector_stores.get(codebase)
            if not vector_store:
                return {"error": f"Codebase not found: {codebase}", "summaries": [], "total": 0}
        else:
            codebase = next(iter(self._vector_stores.keys()))
            vector_store = self._vector_stores[codebase]
        
        try:
            # Get all summarized files metadata (cached, fast)
            summarized_files = vector_store.summary_index.get_all_summarized_files()

            # Sort file paths and apply pagination BEFORE fetching content
            sorted_paths = sorted(summarized_files.keys())
            total = len(sorted_paths)
            page_paths = sorted_paths[offset:offset + limit]

            # Batch fetch full summaries for only the current page (1 query instead of N)
            full_summaries = vector_store.summary_index.get_full_summaries_batch(page_paths)

            # Build summary list from batch results
            summaries = []
            for file_path in page_paths:
                info = summarized_files.get(file_path, {})
                summary_data = full_summaries.get(file_path, {})

                # Get purpose/content from full summary
                summary_content = summary_data.get("purpose", "")

                summaries.append({
                    "file_path": file_path,
                    "pattern": info.get("pattern", ""),
                    "domain": info.get("domain", ""),
                    "model": info.get("model", ""),
                    "summarized_at": info.get("summarized_at", ""),
                    "content_preview": summary_content[:300] + "..." if len(summary_content) > 300 else summary_content,
                    "content": summary_content
                })

            return {
                "codebase": codebase,
                "summaries": summaries,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
            
        except Exception as e:
            logger.error(f"Error listing summaries: {e}")
            return {"error": str(e), "summaries": [], "total": 0}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Import Graph and Heuristic Metadata API
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_import_graph_stats(self, codebase: Optional[str] = None) -> Dict[str, Any]:
        """
        Get import graph statistics for a codebase.
        
        Args:
            codebase: Codebase name (None = all codebases)
            
        Returns:
            Dictionary with import graph statistics
        """
        if codebase:
            if codebase in self._import_graphs:
                return {
                    codebase: self._import_graphs[codebase].get_graph_stats()
                }
            else:
                return {codebase: {"error": "Codebase not found"}}
        else:
            # Return stats for all codebases
            return {
                name: graph.get_graph_stats() 
                for name, graph in self._import_graphs.items()
            }
    
    def get_file_centrality_scores(self, codebase: str, max_files: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get files sorted by centrality score for a codebase.
        
        Args:
            codebase: Codebase name
            max_files: Maximum number of files to return
            
        Returns:
            List of (file_path, centrality_score) tuples, sorted by score descending
        """
        if codebase not in self._import_graphs:
            return []
        
        return self._import_graphs[codebase].get_priority_queue(max_files)
    
    def get_file_dependencies(self, codebase: str, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get dependency information for a specific file.
        
        Args:
            codebase: Codebase name
            file_path: Path to the file
            
        Returns:
            Dictionary with file dependency stats or None if not found
        """
        if codebase not in self._import_graphs:
            return None
        
        return self._import_graphs[codebase].get_file_stats(file_path)
    
    def export_import_graph(self, codebase: str, format: str = 'json') -> Optional[str]:
        """
        Export import graph for visualization.
        
        Args:
            codebase: Codebase name
            format: Export format ('json', 'gexf', 'graphml')
            
        Returns:
            Exported graph data as string or None if export fails
        """
        if codebase not in self._import_graphs:
            return None
        
        return self._import_graphs[codebase].export_graph(format)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Method Call Graph API
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_call_graph(self, codebase: str) -> Optional[MethodCallGraph]:
        """
        Get the method call graph for a codebase.
        
        The call graph tracks method-to-method relationships extracted from
        AST analysis during indexing. Use this for advanced queries like
        finding all callers of a method or tracing call paths.
        
        Args:
            codebase: Name of the codebase
            
        Returns:
            MethodCallGraph if the codebase exists and has a call graph, None otherwise
        """
        return self._call_graphs.get(codebase)
    
    def get_method_callers(self, method_name: str, codebase: str) -> List[Dict[str, Any]]:
        """
        Get all methods that call the specified method.
        
        This answers "what calls X?" queries by looking up the reverse edges
        in the call graph.
        
        Args:
            method_name: The method name to find callers for. Can be:
                - Simple name: "process_data" - matches any method with this name
                - Qualified name: "MyClass.process_data" - matches exact qualified name
            codebase: Name of the codebase to search in
            
        Returns:
            List of dicts with caller information, each containing:
                - qualified_name: Full qualified name (e.g., "MyClass.caller_method")
                - method_name: Just the method name
                - class_name: Containing class (if any)
                - file_path: Path to source file
                - line_number: Line where method is defined
            
        Example:
            >>> callers = service.get_method_callers("fit", "my-codebase")
            >>> for caller in callers:
            ...     print(f"{caller['qualified_name']} in {caller['file_path']}")
        """
        call_graph = self._call_graphs.get(codebase)
        if not call_graph:
            return []
        
        # Find matching methods (support both simple and qualified names)
        matching_qualified_names = []
        
        if '.' in method_name:
            # Already qualified - look for exact match
            if method_name in call_graph.nodes:
                matching_qualified_names.append(method_name)
        else:
            # Simple name - find all methods with this name
            for qname, node in call_graph.nodes.items():
                if node.method_name == method_name:
                    matching_qualified_names.append(qname)
        
        # Collect all callers for matching methods
        all_callers: List[Dict[str, Any]] = []
        seen_callers = set()
        
        for qname in matching_qualified_names:
            callers = call_graph.get_callers(qname)
            for caller in callers:
                if caller.qualified_name not in seen_callers:
                    seen_callers.add(caller.qualified_name)
                    all_callers.append(caller.to_dict())
        
        return all_callers
    
    def get_method_callees(self, method_name: str, codebase: str) -> List[Dict[str, Any]]:
        """
        Get all methods called by the specified method.
        
        This answers "what does X call?" queries by looking up the forward edges
        in the call graph.
        
        Args:
            method_name: The method name to find callees for. Can be:
                - Simple name: "process_data" - matches any method with this name
                - Qualified name: "MyClass.process_data" - matches exact qualified name
            codebase: Name of the codebase to search in
            
        Returns:
            List of dicts with callee information, each containing:
                - qualified_name: Full qualified name (e.g., "MyClass.helper_method")
                - method_name: Just the method name
                - class_name: Containing class (if any)
                - file_path: Path to source file
                - line_number: Line where method is defined
            
        Example:
            >>> callees = service.get_method_callees("process_data", "my-codebase")
            >>> for callee in callees:
            ...     print(f"Calls {callee['qualified_name']}")
        """
        call_graph = self._call_graphs.get(codebase)
        if not call_graph:
            return []
        
        # Find matching methods (support both simple and qualified names)
        matching_qualified_names = []
        
        if '.' in method_name:
            # Already qualified - look for exact match
            if method_name in call_graph.nodes:
                matching_qualified_names.append(method_name)
        else:
            # Simple name - find all methods with this name
            for qname, node in call_graph.nodes.items():
                if node.method_name == method_name:
                    matching_qualified_names.append(qname)
        
        # Collect all callees for matching methods
        all_callees: List[Dict[str, Any]] = []
        seen_callees = set()
        
        for qname in matching_qualified_names:
            callees = call_graph.get_callees(qname)
            for callee in callees:
                if callee.qualified_name not in seen_callees:
                    seen_callees.add(callee.qualified_name)
                    all_callees.append(callee.to_dict())
        
        return all_callees
    
    def get_call_graph_stats(self, codebase: Optional[str] = None) -> Dict[str, Any]:
        """
        Get call graph statistics for a codebase.
        
        Args:
            codebase: Codebase name (None = all codebases)
            
        Returns:
            Dictionary with call graph statistics including:
                - total_methods: Number of method nodes
                - total_edges: Number of call relationships
                - methods_with_callers: Methods that are called by others
                - methods_with_callees: Methods that call others
                - isolated_methods: Methods with no call relationships
        """
        if codebase:
            call_graph = self._call_graphs.get(codebase)
            if call_graph:
                return {codebase: call_graph.get_stats()}
            else:
                return {codebase: {"error": "Codebase not found"}}
        else:
            # Return stats for all codebases
            return {
                name: graph.get_stats() 
                for name, graph in self._call_graphs.items()
            }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary Validation API
    # ─────────────────────────────────────────────────────────────────────────
    
    async def get_summary_statistics_async(
        self,
        codebase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated summary statistics with distribution by pattern, domain, and status.
        
        Args:
            codebase: Optional codebase name. If None, aggregates across all codebases.
            
        Returns:
            Dict with:
                - total_summarized: Total number of summarized files
                - by_pattern: Dict mapping pattern -> count (sorted by count desc)
                - by_domain: Dict mapping domain -> count (sorted by count desc)
                - by_status: Dict mapping status -> count
                - by_codebase: Dict mapping codebase -> count (if codebase is None)
        """
        result = {
            "total_summarized": 0,
            "by_pattern": {},
            "by_domain": {},
            "by_status": {},
            "by_codebase": {}
        }
        
        # Determine which codebases to query
        if codebase:
            vector_store = self._vector_stores.get(codebase)
            if not vector_store:
                return {"error": f"Codebase not found: {codebase}", **result}
            stores_to_query = [(codebase, vector_store)]
        else:
            stores_to_query = list(self._vector_stores.items())
        
        # Aggregate statistics across codebases
        for cb_name, vector_store in stores_to_query:
            try:
                stats = vector_store.summary_index.get_summary_stats()
                
                cb_total = stats.get("total_summarized", 0)
                result["total_summarized"] += cb_total
                result["by_codebase"][cb_name] = cb_total
                
                # Merge pattern counts
                for pattern, count in stats.get("by_pattern", {}).items():
                    result["by_pattern"][pattern] = result["by_pattern"].get(pattern, 0) + count
                
                # Merge domain counts
                for domain, count in stats.get("by_domain", {}).items():
                    result["by_domain"][domain] = result["by_domain"].get(domain, 0) + count
                
                # Merge status counts
                for status, count in stats.get("by_status", {}).items():
                    result["by_status"][status] = result["by_status"].get(status, 0) + count
                    
            except Exception as e:
                logger.warning(f"Error getting summary stats for {cb_name}: {e}")
        
        # Sort by count descending and convert to list of tuples for consistent ordering
        result["by_pattern"] = dict(
            sorted(result["by_pattern"].items(), key=lambda x: x[1], reverse=True)
        )
        result["by_domain"] = dict(
            sorted(result["by_domain"].items(), key=lambda x: x[1], reverse=True)
        )
        
        return result
    
    async def get_validation_queue_async(
        self,
        codebase: str,
        status: str = "unreviewed",
        pattern: str = "",
        domain: str = "",
        offset: int = 0,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get files for summary validation with pagination and filtering.
        
        Args:
            codebase: Codebase name
            status: Filter by validation status: "unreviewed", "approved", "rejected", "all"
            pattern: Filter by code pattern (e.g., "service", "utility")
            domain: Filter by domain (e.g., "api", "database")
            offset: Pagination offset
            limit: Page size (max 50)
            
        Returns:
            Dict with paginated files list, counts, and filter options
        """
        vector_store = self._vector_stores.get(codebase)
        if not vector_store:
            return {"error": f"Codebase not found: {codebase}", "files": []}
        
        # Clamp limit
        limit = min(max(1, limit), 50)
        
        try:
            # Get all summarized files (cached)
            summarized_files = vector_store.summary_index.get_all_summarized_files()
            
            files = []
            reviewed_count = 0
            all_patterns = set()
            all_domains = set()
            
            for file_path, info in summarized_files.items():
                validation_status = info.get("validation_status", "unreviewed")
                file_pattern = info.get("pattern", "")
                file_domain = info.get("domain", "")
                
                # Collect all patterns/domains for filter dropdowns
                if file_pattern:
                    all_patterns.add(file_pattern)
                if file_domain:
                    all_domains.add(file_domain)
                
                # Count reviewed files (before any filtering)
                if validation_status in ["approved", "rejected"]:
                    reviewed_count += 1
                
                # Apply status filter
                status_match = (
                    status == "all" or
                    (status == "unreviewed" and validation_status == "unreviewed") or
                    status == validation_status
                )
                if not status_match:
                    continue
                
                # Apply pattern filter
                if pattern and file_pattern != pattern:
                    continue
                
                # Apply domain filter
                if domain and file_domain != domain:
                    continue
                
                files.append({
                    "path": file_path,
                    "status": validation_status,
                    "pattern": file_pattern,
                    "domain": file_domain,
                    "simple_file": info.get("simple_file", False)
                })
            
            # Sort by path
            files.sort(key=lambda f: f["path"])
            
            # Total before pagination
            total = len(files)
            
            # Apply pagination
            paginated_files = files[offset:offset + limit]
            
            return {
                "codebase": codebase,
                "files": paginated_files,
                "total": total,
                "offset": offset,
                "limit": limit,
                "reviewed_count": reviewed_count,
                "pattern_options": sorted(all_patterns),
                "domain_options": sorted(all_domains)
            }
            
        except Exception as e:
            logger.error(f"Error getting validation queue: {e}")
            return {"error": str(e), "files": []}
    
    async def get_file_summary_async(
        self,
        codebase: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Get the full structured summary for a file.
        
        Returns all Phase 2 fields (how_it_works, key_mechanisms, method_summaries)
        when available, along with the basic summary fields.
        
        Args:
            codebase: Codebase name
            file_path: Relative path to file
            
        Returns:
            Dict with structured summary including all available fields
        """
        vector_store = self._vector_stores.get(codebase)
        if not vector_store:
            return {"error": f"Codebase not found: {codebase}"}

        try:
            # Use the new get_full_summary method which parses stored JSON
            # Try both path separator formats for Windows compatibility
            full_summary = vector_store.summary_index.get_full_summary(file_path)
            if not full_summary:
                full_summary = vector_store.summary_index.get_full_summary(file_path.replace("\\", "/"))
            if not full_summary:
                full_summary = vector_store.summary_index.get_full_summary(file_path.replace("/", "\\"))

            if not full_summary:
                return {"codebase": codebase, "file_path": file_path, "summary": None}
            
            # Build response with all available fields
            summary_response = {
                "purpose": full_summary.get("purpose", ""),
                "pattern": full_summary.get("pattern", ""),
                "domain": full_summary.get("domain", ""),
                "language": full_summary.get("language", ""),
                "key_exports": full_summary.get("key_exports", []),
                "dependencies": full_summary.get("dependencies", []),
                "model": full_summary.get("model_used", full_summary.get("model", "")),
                "summarized_at": full_summary.get("summarized_at", ""),
                "validation_status": full_summary.get("validation_status", "unreviewed"),
                # Simple file tracking
                "simple_file": full_summary.get("simple_file", False),
                "simple_file_reason": full_summary.get("simple_file_reason"),
            }
            
            # Add Phase 2 fields when present
            if full_summary.get("how_it_works"):
                summary_response["how_it_works"] = full_summary["how_it_works"]
            
            if full_summary.get("key_mechanisms"):
                summary_response["key_mechanisms"] = full_summary["key_mechanisms"]
            
            if full_summary.get("method_summaries"):
                summary_response["method_summaries"] = full_summary["method_summaries"]
            
            # Include boolean flags for quick filtering
            summary_response["has_how_it_works"] = full_summary.get("has_how_it_works", False)
            summary_response["has_method_summaries"] = full_summary.get("has_method_summaries", False)
            
            return {
                "codebase": codebase,
                "file_path": file_path,
                "summary": summary_response
            }
            
        except Exception as e:
            logger.error(f"Error getting file summary: {e}")
            return {"error": str(e)}
    
    async def get_file_content_async(
        self,
        codebase: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Get raw file content for validation.
        
        Args:
            codebase: Codebase name
            file_path: Relative path to file
            
        Returns:
            Dict with file content
        """
        # Find the codebase config to get the base path
        codebase_config = None
        for cb in self.config.codebases:
            if cb.name == codebase:
                codebase_config = cb
                break
        
        if not codebase_config:
            return {"error": f"Codebase not found: {codebase}"}
        
        try:
            from pathlib import Path
            full_path = Path(codebase_config.path) / file_path
            
            if not full_path.exists():
                return {"error": f"File not found: {file_path}"}
            
            # Read file with error handling for encoding
            try:
                content = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = full_path.read_text(encoding="latin-1")
            
            # Limit content size for display
            max_size = 50000
            if len(content) > max_size:
                content = content[:max_size] + f"\n\n... (truncated, {len(content)} total characters)"
            
            return {
                "codebase": codebase,
                "file_path": file_path,
                "content": content
            }
            
        except Exception as e:
            logger.error(f"Error reading file content: {e}")
            return {"error": str(e)}
    
    async def validate_summary_async(
        self,
        codebase: str,
        file_path: str,
        status: str
    ) -> Dict[str, Any]:
        """
        Validate (approve/reject) a summary.
        
        Args:
            codebase: Codebase name
            file_path: File path
            status: "approved" or "rejected"
            
        Returns:
            Dict with success status
        """
        vector_store = self._vector_stores.get(codebase)
        if not vector_store:
            return {"error": f"Codebase not found: {codebase}", "success": False}
        
        try:
            # Update the summary index with validation status
            from datetime import datetime
            
            # Get current summary info
            summary_info = vector_store.summary_index.get_summary_info(file_path)
            if not summary_info:
                return {"error": "Summary not found", "success": False}
            
            # Update with validation status
            summary_info["validation_status"] = status
            summary_info["validated_at"] = datetime.now().isoformat()
            
            # Store updated info back
            vector_store.summary_index.update_summary_metadata(file_path, summary_info)
            
            return {"success": True, "status": status}
            
        except Exception as e:
            logger.error(f"Error validating summary: {e}")
            return {"error": str(e), "success": False}
    
    async def regenerate_summary_async(
        self,
        codebase: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Regenerate a summary for a file.
        
        Args:
            codebase: Codebase name
            file_path: File path
            
        Returns:
            Dict with success status and new summary
        """
        if not self._summarizer:
            # Provide detailed diagnostics
            from ..llm.ollama_client import OllamaClient
            
            is_installed, install_msg = OllamaClient.check_ollama_installed()
            if not is_installed:
                return {"error": f"Summarization not available: {install_msg}", "success": False}
            
            is_running, running_msg = OllamaClient.check_ollama_running(self._summarization_config.ollama_url)
            if not is_running:
                return {"error": f"Summarization not available: {running_msg}", "success": False}
            
            return {"error": "Summarizer not initialized. The server may need to be restarted after starting Ollama.", "success": False}
        
        vector_store = self._vector_stores.get(codebase)
        if not vector_store:
            return {"error": f"Codebase not found: {codebase}", "success": False}
        
        try:
            # Get the file content
            content_result = await self.get_file_content_async(codebase, file_path)
            if "error" in content_result:
                return content_result
            
            content = content_result["content"]
            
            # Get heuristic metadata if available
            heuristic_metadata = None
            try:
                heuristic_metadata = self.heuristic_extractor.extract_file_metadata(file_path, content)
            except Exception:
                pass  # Heuristics are optional
            
            # Generate new summary using the correct method signature
            summary = await self._summarizer.summarize_file(file_path, content, heuristic_metadata)
            
            if summary.error:
                return {"error": f"Summarization failed: {summary.error}", "success": False}
            
            # Compute content hash
            content_hash = ChromaVectorStore.compute_content_hash(content)
            
            # Store the new summary
            await self._store_summary(codebase, file_path, summary, content_hash)
            
            return {
                "success": True,
                "summary": {
                    "purpose": summary.purpose,
                    "pattern": summary.pattern,
                    "domain": summary.domain,
                    "exports": ", ".join(summary.key_exports) if summary.key_exports else "",
                    "model": summary.model_used
                }
            }
            
        except Exception as e:
            logger.error(f"Error regenerating summary: {e}")
            return {"error": str(e), "success": False}
    

