"""
MemoryService - Core memory service with multi-codebase support.

This is the single source of truth for all memory/indexing logic.
Both the HTTP server and stdio MCP server are thin wrappers around this service.
"""

import asyncio
import concurrent.futures
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..config.server import ServerConfig, CodebaseConfig
from ..storage.chroma import ChromaVectorStore
from ..search.chunking import ChunkingManager, ChunkingStrategy
from ..core.models import MemoryChunk, RoleEnum, MemoryType
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
        
        # Shared embedder and chunking manager
        self.embedder = SentenceTransformerEmbedder()
        self.chunking_manager = ChunkingManager(ChunkingStrategy.FUNCTION_CLASS)
        
        # Hybrid search support
        self.hybrid_searcher = HybridSearcher(rrf_k=60, semantic_weight=0.5)
        
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
                indexing_tasks.append(self._index_codebase(codebase))
            else:
                # Start as background task
                asyncio.create_task(self._index_codebase(codebase))
            
            # Start file watcher if enabled
            if self.config.enable_file_watcher:
                self._file_watcher_tasks[codebase.name] = asyncio.create_task(
                    self._watch_for_changes(codebase)
                )
        
        # Wait for all indexing to complete if requested
        if wait_for_indexing and indexing_tasks:
            await asyncio.gather(*indexing_tasks)
    
    async def search_async(
        self,
        query: str,
        codebase: Optional[str] = None,
        max_results: int = 10,
        project_id: Optional[str] = None,
        search_mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Search for relevant memories using semantic similarity, keyword matching, or both.
        
        Args:
            query: Search query text
            codebase: Optional codebase name to search (None = search all)
            max_results: Maximum number of results to return
            project_id: Optional filter by project ID
            search_mode: "semantic", "keyword", "hybrid", or "auto" (default)
            
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
                        if codebase_name in self._hybrid_searchers:
                            try:
                                self._hybrid_searchers[codebase_name].remove_from_index(codebase_name, memory_id)
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
    
    def search(
        self,
        query: str,
        codebase: Optional[str] = None,
        max_results: int = 10,
        project_id: Optional[str] = None,
        search_mode: str = "auto"
    ) -> Dict[str, Any]:
        """Sync wrapper for search_async"""
        return _run_sync(self.search_async(query, codebase, max_results, project_id, search_mode))
    
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
    
    async def _index_codebase(self, codebase: CodebaseConfig) -> None:
        """Index a single codebase - incremental if index exists"""
        codebase_path = Path(codebase.path)
        status = self._indexing_status[codebase.name]
        vector_store = self._vector_stores[codebase.name]
        
        try:
            status["status"] = "indexing"
            status["progress"] = 0.0
            
            logger.info(f"[{codebase.name}] Starting indexing for: {codebase_path}")
            
            # Find all code files
            code_extensions = codebase.get_extension_set()
            logger.info(f"[{codebase.name}] DEBUG: Looking for extensions: {code_extensions}")
            
            code_files = []
            for ext in code_extensions:
                ext_files = list(codebase_path.rglob(f'*{ext}'))
                logger.info(f"[{codebase.name}] DEBUG: Found {len(ext_files)} files with extension {ext}")
                code_files.extend(ext_files)
            
            logger.info(f"[{codebase.name}] DEBUG: Total files found before filtering: {len(code_files)}")
            
            # Log some example files found
            if code_files:
                logger.info(f"[{codebase.name}] DEBUG: First 10 files found:")
                for i, f in enumerate(code_files[:10]):
                    logger.info(f"[{codebase.name}] DEBUG:   {i+1}. {f}")
                
                # Check specifically for src files
                src_files = [f for f in code_files if 'src' in str(f)]
                logger.info(f"[{codebase.name}] DEBUG: Files containing 'src': {len(src_files)}")
                if src_files:
                    logger.info(f"[{codebase.name}] DEBUG: First 5 src files:")
                    for i, f in enumerate(src_files[:5]):
                        logger.info(f"[{codebase.name}] DEBUG:   {i+1}. {f}")
                
                # Check specifically for swing_models.py
                swing_models_files = [f for f in code_files if 'swing_models.py' in str(f)]
                if swing_models_files:
                    logger.info(f"[{codebase.name}] DEBUG: Found swing_models.py: {swing_models_files}")
                else:
                    logger.info(f"[{codebase.name}] DEBUG: swing_models.py NOT found in discovered files")
            
            # Filter out ignored patterns
            files_before_filter = len(code_files)
            code_files_filtered = []
            ignored_files = []
            
            for f in code_files:
                if codebase.should_ignore(str(f)):
                    ignored_files.append(f)
                else:
                    code_files_filtered.append(f)
            
            code_files = code_files_filtered
            logger.info(f"[{codebase.name}] DEBUG: Files after ignore filtering: {len(code_files)} (filtered out: {len(ignored_files)})")
            
            # Log some ignored files to see patterns
            if ignored_files:
                logger.info(f"[{codebase.name}] DEBUG: First 10 ignored files:")
                for i, f in enumerate(ignored_files[:10]):
                    logger.info(f"[{codebase.name}] DEBUG:   IGNORED: {f}")
                
                # Check if any src files were ignored
                ignored_src_files = [f for f in ignored_files if 'src' in str(f)]
                if ignored_src_files:
                    logger.warning(f"[{codebase.name}] DEBUG: {len(ignored_src_files)} src files were IGNORED!")
                    for i, f in enumerate(ignored_src_files[:5]):
                        logger.warning(f"[{codebase.name}] DEBUG:   IGNORED SRC: {f}")
            
            # Check final src files after filtering
            final_src_files = [f for f in code_files if 'src' in str(f)]
            logger.info(f"[{codebase.name}] DEBUG: Final src files after filtering: {len(final_src_files)}")
            
            # Check specifically for swing_models.py after filtering
            final_swing_models = [f for f in code_files if 'swing_models.py' in str(f)]
            if final_swing_models:
                logger.info(f"[{codebase.name}] DEBUG: swing_models.py survived filtering: {final_swing_models}")
            else:
                logger.warning(f"[{codebase.name}] DEBUG: swing_models.py was filtered out or not found!")
            
            # Get existing indexed files
            indexed_files = vector_store.get_indexed_files()
            current_file_paths = set()
            
            # Determine which files need indexing
            files_to_index = []
            files_to_reindex = []
            
            for file_path in code_files:
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
                    logger.warning(f"[{codebase.name}] Error checking file {file_path}: {e}")
            
            # Find and remove deleted files
            deleted_files = set(indexed_files.keys()) - current_file_paths
            for deleted_path in deleted_files:
                vector_store.remove_file_chunks(deleted_path)
                logger.info(f"[{codebase.name}] Removed deleted file from index: {deleted_path}")
            
            # Remove old chunks for files to re-index
            for file_path, _, _, _ in files_to_reindex:
                relative_path = str(file_path.relative_to(codebase_path))
                vector_store.remove_file_chunks(relative_path)
            
            # Combine files to process
            all_files = files_to_index + files_to_reindex
            status["total_files"] = len(all_files)
            
            # Log summary
            existing_count = len(indexed_files) - len(deleted_files)
            logger.info(f"[{codebase.name}] Index status: {existing_count} files already indexed")
            
            if not all_files and not deleted_files:
                logger.info(f"[{codebase.name}] Index is up to date - no changes detected")
                status["status"] = "completed"
                status["progress"] = 1.0
                status["indexed_files_count"] = len(vector_store.get_indexed_files())
                return
            
            if files_to_index:
                logger.info(f"[{codebase.name}] New files to index: {len(files_to_index)}")
            if files_to_reindex:
                logger.info(f"[{codebase.name}] Modified files to re-index: {len(files_to_reindex)}")
            if deleted_files:
                logger.info(f"[{codebase.name}] Deleted files removed: {len(deleted_files)}")
            
            # Process files
            for i, (file_path, mtime, content_hash, content) in enumerate(all_files):
                await self._index_single_file(file_path, content, mtime, content_hash, codebase, vector_store)
                status["files_processed"] = i + 1
                status["progress"] = (i + 1) / len(all_files)
                await asyncio.sleep(0)  # Yield control
            
            status["status"] = "completed"
            status["progress"] = 1.0
            status["indexed_files_count"] = len(vector_store.get_indexed_files())
            
            # Build BM25 index for hybrid search
            self.hybrid_searcher.rebuild_index(codebase.name)
            bm25_stats = self.hybrid_searcher.get_index_stats().get(codebase.name, {})
            
            # Final summary
            logger.info(f"[{codebase.name}] Indexing completed:")
            logger.info(f"[{codebase.name}]   Total files indexed: {status['indexed_files_count']}")
            logger.info(f"[{codebase.name}]   New files added: {len(files_to_index)}")
            logger.info(f"[{codebase.name}]   Modified files re-indexed: {len(files_to_reindex)}")
            logger.info(f"[{codebase.name}]   BM25 index: {bm25_stats.get('document_count', 0)} documents")
            
        except Exception as e:
            logger.error(f"[{codebase.name}] Error during indexing: {e}")
            status["status"] = "error"
            status["error_message"] = str(e)
    
    async def _index_single_file(
        self,
        file_path: Path,
        content: str,
        mtime: float,
        content_hash: str,
        codebase: CodebaseConfig,
        vector_store: ChromaVectorStore
    ) -> None:
        """Index a single file"""
        codebase_path = Path(codebase.path)
        status = self._indexing_status[codebase.name]
        status["current_file"] = str(file_path)
        
        if not content.strip():
            return
        
        relative_path = str(file_path.relative_to(codebase_path))
        chunks = self.chunking_manager.chunk_text(content, relative_path)
        
        chunk_ids = []
        for chunk_text, metadata in chunks:
            if len(chunk_text.strip()) < 50:
                continue
            
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
            
            embedding = self.embedder.generate(chunk_text)
            vector_store.add(memory_chunk, embedding)
            
            # Also add to BM25 index for hybrid search
            self.hybrid_searcher.add_to_index(codebase.name, memory_chunk)
        
        if chunk_ids:
            vector_store.update_file_index(relative_path, mtime, content_hash, chunk_ids)
            logger.debug(f"[{codebase.name}] Indexed {len(chunk_ids)} chunks for: {relative_path}")
    
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
