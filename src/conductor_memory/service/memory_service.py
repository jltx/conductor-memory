"""
MemoryService - Core memory service with multi-codebase support.

This is the single source of truth for all memory/indexing logic.
Both the HTTP server and stdio MCP server are thin wrappers around this service.
"""

import asyncio
import concurrent.futures
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..config.server import ServerConfig, CodebaseConfig
from ..storage.chroma import ChromaVectorStore
from ..search.chunking import ChunkingManager, ChunkingStrategy
from ..search.boosting import BoostCalculator
from ..search.heuristics import HeuristicExtractor
from ..search.import_graph import ImportGraph
from ..core.models import MemoryChunk, RoleEnum, MemoryType
from ..core.resources import detect_resources, get_optimal_device, SystemResources
from ..embedding.sentence_transformer import SentenceTransformerEmbedder
from ..search.hybrid import HybridSearcher, SearchMode

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
        
        # Initialize vector stores for each enabled codebase
        for codebase in config.get_enabled_codebases():
            self._init_codebase(codebase)
        
        logger.info(f"MemoryService initialized with {len(self._vector_stores)} codebase(s)")
        logger.info(f"Persistent storage at: {self.persist_directory}")
    
    def _init_codebase(self, codebase: CodebaseConfig) -> None:
        """Initialize vector store and status for a codebase"""
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
            
            # Note: File watchers are NOT started here because when called via
            # _run_sync(), we're in a temporary event loop that will close.
            # File watchers should be started via start_file_watchers() in the
            # main server's event loop.
    
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
        min_function_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant memories using semantic similarity, keyword matching, or both.
        
        Args:
            query: Search query text
            codebase: Optional codebase name to search (None = search all)
            max_results: Maximum number of results to return
            project_id: Optional filter by project ID
            search_mode: "semantic", "keyword", "hybrid", or "auto" (default)
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
            
        Returns:
            Dict with results, total_found, query_time_ms, summary, and search_mode_used
        """
        start_time = datetime.now()
        
        try:
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
                query_embedding = self.embedder.generate(query)
                
                if codebase:
                    vector_store = self._vector_stores.get(codebase)
                    if not vector_store:
                        return {
                            "error": f"Codebase not found: {codebase}",
                            "results": [],
                            "total_found": 0
                        }
                    similar_chunks = vector_store.search(query_embedding, max_results * 2)
                else:
                    for codebase_name, vector_store in self._vector_stores.items():
                        chunks = vector_store.search(query_embedding, max_results)
                        similar_chunks.extend(chunks)
                
            else:  # HYBRID
                # Combined semantic + keyword search with RRF fusion
                query_embedding = self.embedder.generate(query)
                
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
                    semantic_chunks = vector_store.search(query_embedding, max_results * 2)
                else:
                    for codebase_name, vector_store in self._vector_stores.items():
                        chunks = vector_store.search(query_embedding, max_results)
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
                similar_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            
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
            
            # Remove duplicates based on content similarity
            deduplicated_chunks = self._deduplicate_chunks(similar_chunks)
            
            # Limit results
            filtered_chunks = deduplicated_chunks[:max_results]
            
            # Format results
            results = []
            for chunk in filtered_chunks:
                results.append({
                    "id": chunk.id,
                    "project_id": chunk.project_id,
                    "role": chunk.role.value if hasattr(chunk.role, 'value') else str(chunk.role),
                    "content": chunk.doc_text[:500] + "..." if len(chunk.doc_text) > 500 else chunk.doc_text,
                    "tags": chunk.tags,
                    "source": chunk.source,
                    "relevance_score": chunk.relevance_score
                })
            
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
            
            # Generate embedding and store
            embedding = self.embedder.generate(content)
            vector_store.add(memory_chunk, embedding)
            
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
        min_function_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Sync wrapper for search_async"""
        return _run_sync(self.search_async(
            query, codebase, max_results, project_id, search_mode, 
            domain_boosts, include_tags, exclude_tags,
            languages, class_names, function_names, annotations,
            has_annotations, has_docstrings, min_class_count, min_function_count
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
        """List all configured codebases with their info"""
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
        bar = "█" * filled + "░" * (bar_width - filled)
        
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
                return
            
            # Process files in batches for better GPU utilization
            total_files = len(all_files)
            await self._index_files_batched(
                all_files, codebase, vector_store, status, total_files
            )
            
            status["status"] = "completed"
            status["progress"] = 1.0
            status["indexed_files_count"] = len(vector_store.get_indexed_files())
            
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
                    
            except asyncio.CancelledError:
                logger.info(f"[{codebase.name}] File watcher stopped")
                break
            except Exception as e:
                logger.error(f"[{codebase.name}] Error in file watcher: {e}")
    
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
