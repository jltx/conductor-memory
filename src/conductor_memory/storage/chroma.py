"""
Chroma Vector Store Implementation for the Hybrid Local/Cloud LLM Orchestrator

Provides persistent vector storage using Chroma database with efficient similarity search.
"""

import os
import logging
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb.config import Settings

from ..core.vector_store import VectorStore
from ..core.models import MemoryChunk, MemoryType

logger = logging.getLogger(__name__)


class FileIndexMetadata:
    """Tracks metadata for indexed files to support incremental indexing"""
    
    def __init__(self, client: chromadb.ClientAPI, collection_name: str = "file_index_metadata"):
        """
        Initialize file index metadata tracker
        
        Args:
            client: Chroma client instance
            collection_name: Name of the metadata collection
        """
        self.client = client
        self.collection_name = collection_name
        
        # Cache for get_all_indexed_files (invalidated on updates)
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 60.0  # 60 second cache TTL (was 5s, caused event loop blocking)
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing file index metadata collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new file index metadata collection: {collection_name}")
    
    def _invalidate_cache(self):
        """Invalidate the cache after updates"""
        self._cache = None
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get stored metadata for a file"""
        try:
            result = self.collection.get(ids=[file_path], include=["metadatas"])
            if result["ids"]:
                return result["metadatas"][0]
        except Exception as e:
            logger.debug(f"File not found in index: {file_path} - {e}")
        return None
    
    def update_file_info(self, file_path: str, mtime: float, content_hash: str, chunk_ids: List[str]) -> None:
        """Update or insert file metadata"""
        metadata = {
            "mtime": mtime,
            "content_hash": content_hash,
            "chunk_ids": ",".join(chunk_ids),
            "indexed_at": datetime.now().isoformat()
        }
        
        # Use upsert - Chroma handles both insert and update
        try:
            self.collection.upsert(
                ids=[file_path],
                metadatas=[metadata],
                documents=[file_path]  # Required by Chroma
            )
            self._invalidate_cache()
        except Exception as e:
            logger.warning(f"Error updating file index metadata for {file_path}: {e}")
    
    def update_file_info_batch(self, file_infos: List[Dict[str, Any]]) -> None:
        """
        Batch update or insert file metadata.
        Much faster than calling update_file_info() repeatedly.
        
        Automatically splits into smaller batches to respect ChromaDB limits.
        
        Args:
            file_infos: List of dicts with keys: file_path, mtime, content_hash, chunk_ids
        """
        if not file_infos:
            return
        
        # ChromaDB has a max batch size (typically around 5000)
        # Use a safe batch size to avoid hitting limits
        BATCH_SIZE = 1000
        indexed_at = datetime.now().isoformat()
        total_updated = 0
        
        for batch_start in range(0, len(file_infos), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(file_infos))
            batch = file_infos[batch_start:batch_end]
            
            ids = []
            metadatas = []
            documents = []
            
            for info in batch:
                ids.append(info['file_path'])
                documents.append(info['file_path'])
                metadatas.append({
                    "mtime": info['mtime'],
                    "content_hash": info['content_hash'],
                    "chunk_ids": ",".join(info['chunk_ids']),
                    "indexed_at": indexed_at
                })
            
            try:
                self.collection.upsert(
                    ids=ids,
                    metadatas=metadatas,
                    documents=documents
                )
                total_updated += len(ids)
            except Exception as e:
                logger.error(f"Error batch updating file index metadata (batch {batch_start}-{batch_end}): {e}")
                raise  # Re-raise to fail the indexing properly
        
        self._invalidate_cache()
        logger.debug(f"Batch updated file index for {total_updated} files")
    
    def delete_file_info(self, file_path: str) -> List[str]:
        """Delete file metadata and return chunk IDs to remove"""
        chunk_ids = []
        try:
            result = self.collection.get(ids=[file_path], include=["metadatas"])
            if result["ids"] and result["metadatas"]:
                chunk_ids_str = result["metadatas"][0].get("chunk_ids", "")
                if chunk_ids_str:
                    chunk_ids = chunk_ids_str.split(",")
                self.collection.delete(ids=[file_path])
                self._invalidate_cache()
        except Exception as e:
            logger.warning(f"Error deleting file index metadata for {file_path}: {e}")
        return chunk_ids
    
    def get_all_indexed_files(self) -> Dict[str, Dict[str, Any]]:
        """Get all indexed file paths and their metadata (cached)"""
        # Check cache
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        indexed_files = {}
        try:
            result = self.collection.get(include=["metadatas"])
            if result["ids"]:
                for i, file_path in enumerate(result["ids"]):
                    indexed_files[file_path] = result["metadatas"][i]

            # Update cache
            self._cache = indexed_files
            self._cache_time = now
        except Exception as e:
            logger.warning(f"Error getting indexed files: {e}")
        return indexed_files

    def get_indexed_files_paginated(
        self, limit: int = 50, offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated list of indexed files with total count.

        Uses ChromaDB's native limit/offset for efficient pagination without
        loading the entire collection into memory.

        Args:
            limit: Maximum number of files to return
            offset: Number of files to skip

        Returns:
            Tuple of (list of file dicts with path and metadata, total count)
        """
        try:
            # Get total count (fast O(1) operation)
            total = self.collection.count()

            # Get paginated results
            result = self.collection.get(
                limit=limit,
                offset=offset,
                include=["metadatas"]
            )

            files = []
            if result["ids"]:
                for i, file_path in enumerate(result["ids"]):
                    files.append({
                        "path": file_path,
                        **result["metadatas"][i]
                    })

            return files, total

        except Exception as e:
            logger.warning(f"Error getting paginated indexed files: {e}")
            return [], 0


# Schema version for summary index structure.
# Increment this when making breaking changes to summary storage format.
# Version history:
#   1 - Original flat text format
#   2 - Structured JSON storage with Phase 2 fields (how_it_works, method_summaries, etc.)
SUMMARY_SCHEMA_VERSION = "2"


class SummaryIndexMetadata:
    """Tracks metadata for summarized files to support incremental re-summarization"""
    
    # Special ID for storing schema version metadata
    _SCHEMA_VERSION_ID = "__schema_version__"
    
    def __init__(self, client: chromadb.ClientAPI, collection_name: str = "summary_index_metadata"):
        """
        Initialize summary index metadata tracker
        
        Args:
            client: Chroma client instance
            collection_name: Name of the metadata collection
        """
        self.client = client
        self.collection_name = collection_name
        
        # Cache for get_all_summarized_files (invalidated on updates)
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 300.0  # 5 minute cache TTL (summary stats change infrequently)
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing summary index metadata collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new summary index metadata collection: {collection_name}")
    
    def get_schema_version(self) -> Optional[str]:
        """
        Get the stored schema version from the summary index.
        
        Returns:
            The stored schema version string, or None if not set
        """
        try:
            result = self.collection.get(ids=[self._SCHEMA_VERSION_ID], include=["metadatas"])
            if result["ids"] and result["metadatas"]:
                return result["metadatas"][0].get("version")
        except Exception as e:
            logger.debug(f"Error getting schema version: {e}")
        return None
    
    def set_schema_version(self, version: str) -> None:
        """
        Store the schema version in the summary index.
        
        Args:
            version: Schema version string to store
        """
        try:
            self.collection.upsert(
                ids=[self._SCHEMA_VERSION_ID],
                metadatas=[{"version": version, "type": "schema_version"}],
                documents=[f"Schema version: {version}"]
            )
            logger.info(f"Set summary schema version to: {version}")
        except Exception as e:
            logger.warning(f"Error setting schema version: {e}")
    
    def check_schema_version(self) -> tuple[bool, Optional[str]]:
        """
        Check if the stored schema version matches the current version.
        
        Returns:
            Tuple of (is_current: bool, stored_version: str or None)
            - is_current is True if stored version matches SUMMARY_SCHEMA_VERSION
            - stored_version is the version found in storage (None if not set)
        """
        stored_version = self.get_schema_version()
        is_current = stored_version == SUMMARY_SCHEMA_VERSION
        return is_current, stored_version
    
    def clear_all_summaries(self) -> int:
        """
        Clear all summaries from the summary index.
        
        This removes all entries EXCEPT the schema version marker.
        Used for schema migrations or manual invalidation.
        
        Returns:
            Number of summaries cleared
        """
        try:
            # Get all IDs in the collection
            result = self.collection.get(include=[])
            all_ids = result["ids"] if result["ids"] else []
            
            # Filter out the schema version marker
            ids_to_delete = [id for id in all_ids if id != self._SCHEMA_VERSION_ID]
            
            if ids_to_delete:
                # Delete in batches to avoid hitting limits
                BATCH_SIZE = 1000
                for i in range(0, len(ids_to_delete), BATCH_SIZE):
                    batch = ids_to_delete[i:i + BATCH_SIZE]
                    self.collection.delete(ids=batch)
                
                logger.info(f"Cleared {len(ids_to_delete)} summaries from summary index")
            
            self._invalidate_cache()
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Error clearing summaries: {e}")
            return 0
    
    def _invalidate_cache(self):
        """Invalidate the cache after updates"""
        self._cache = None
    
    def get_summary_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get stored summary metadata for a file"""
        try:
            result = self.collection.get(ids=[file_path], include=["metadatas"])
            if result["ids"]:
                return result["metadatas"][0]
        except Exception as e:
            logger.debug(f"Summary not found for file: {file_path} - {e}")
        return None
    
    def update_summary_info(
        self, 
        file_path: str, 
        content_hash: str, 
        summary_chunk_id: str,
        model: str,
        pattern: str = "",
        domain: str = ""
    ) -> None:
        """
        Update summary metadata after summarization (legacy method).
        
        Note: This method only updates specific metadata fields without
        overwriting the document content (which may contain structured JSON
        from store_summary()).
        """
        try:
            # Get existing entry to preserve the document content
            existing = self.collection.get(ids=[file_path], include=["documents", "metadatas"])
            
            # Preserve existing document (may contain structured JSON)
            existing_doc = ""
            existing_meta = {}
            if existing["ids"]:
                existing_doc = existing["documents"][0] if existing["documents"] else file_path
                existing_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            
            # Merge new metadata with existing
            metadata = {**existing_meta}
            metadata.update({
                "content_hash": content_hash,
                "summary_chunk_id": summary_chunk_id,
                "model": model,
                "pattern": pattern,
                "domain": domain,
                "summarized_at": datetime.now().isoformat()
            })
            
            self.collection.upsert(
                ids=[file_path],
                metadatas=[metadata],
                documents=[existing_doc if existing_doc else file_path]
            )
            self._invalidate_cache()
        except Exception as e:
            logger.warning(f"Error updating summary index metadata for {file_path}: {e}")
    
    def delete_summary_info(self, file_path: str) -> Optional[str]:
        """Delete summary metadata and return the summary chunk ID to remove"""
        try:
            result = self.collection.get(ids=[file_path], include=["metadatas"])
            if result["ids"] and result["metadatas"]:
                chunk_id = result["metadatas"][0].get("summary_chunk_id", "")
                self.collection.delete(ids=[file_path])
                self._invalidate_cache()
                return chunk_id if chunk_id else None
        except Exception as e:
            logger.warning(f"Error deleting summary index metadata for {file_path}: {e}")
        return None
    
    def needs_resummarization(self, file_path: str, content_hash: str) -> bool:
        """
        Check if a file needs to be re-summarized based on content hash
        
        Args:
            file_path: Relative path to the file
            content_hash: Hash of current file content
            
        Returns:
            True if file needs re-summarization
        """
        summary_info = self.get_summary_info(file_path)
        if summary_info is None:
            return True  # Not summarized yet
        
        stored_hash = summary_info.get("content_hash", "")
        return content_hash != stored_hash
    
    def update_summary_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Update summary metadata for a file (e.g., validation status).
        
        Preserves the existing document content (which may contain structured JSON).
        
        Args:
            file_path: Relative path to the file
            metadata: Metadata dict to merge with existing metadata
        """
        try:
            # Get existing entry to preserve the document content
            existing = self.collection.get(ids=[file_path], include=["documents", "metadatas"])
            
            # Preserve existing document (may contain structured JSON)
            existing_doc = file_path
            existing_meta = {}
            if existing["ids"]:
                existing_doc = existing["documents"][0] if existing["documents"] else file_path
                existing_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            
            # Merge new metadata with existing
            merged_meta = {**existing_meta, **metadata}
            
            self.collection.upsert(
                ids=[file_path],
                metadatas=[merged_meta],
                documents=[existing_doc]
            )
            self._invalidate_cache()
        except Exception as e:
            logger.warning(f"Error updating summary metadata for {file_path}: {e}")
    
    def store_summary(
        self,
        file_path: str,
        summary_data: Dict[str, Any],
        content_hash: str = ""
    ) -> None:
        """
        Store summary data for a file as structured JSON.
        
        The full summary is stored as JSON in the document field for later retrieval.
        Key fields are stored in metadata for filtering (pattern, domain, model, etc.).
        
        Args:
            file_path: Relative path to the file
            summary_data: Full FileSummary.to_dict() or equivalent dict containing:
                - file_path, language, purpose, pattern, key_exports, dependencies
                - domain, model_used, tokens_used, response_time_ms, is_skeleton, error
                - how_it_works (optional), key_mechanisms (optional), method_summaries (optional)
                - simple_file (optional), simple_file_reason (optional)
            content_hash: Hash of the file content
        """
        import json
        
        # Add timestamp if not present
        if "summarized_at" not in summary_data:
            summary_data["summarized_at"] = datetime.now().isoformat()
        
        # Determine boolean flags for Phase 2 fields
        has_how_it_works = bool(summary_data.get("how_it_works"))
        has_method_summaries = bool(summary_data.get("method_summaries"))
        
        # Key exports as comma-separated string for metadata
        key_exports = summary_data.get("key_exports", [])
        exports_str = ", ".join(key_exports[:10]) if key_exports else ""  # Limit to 10
        
        # Build metadata for filtering
        metadata = {
            "content_hash": content_hash,
            "model": summary_data.get("model_used", summary_data.get("model", "")),
            "pattern": summary_data.get("pattern", ""),
            "domain": summary_data.get("domain", ""),
            "language": summary_data.get("language", ""),
            "exports": exports_str,
            "summarized_at": summary_data.get("summarized_at"),
            "validation_status": summary_data.get("validation_status", "unreviewed"),
            # Phase 2: Boolean flags for filtering
            "has_how_it_works": has_how_it_works,
            "has_method_summaries": has_method_summaries,
            # Simple file tracking
            "simple_file": summary_data.get("simple_file", False),
        }
        
        # Store full summary as JSON in document field
        try:
            summary_json = json.dumps(summary_data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"Error serializing summary to JSON for {file_path}: {e}")
            # Fallback to storing just the purpose as content
            summary_json = summary_data.get("purpose", "")
        
        try:
            self.collection.upsert(
                ids=[file_path],
                metadatas=[metadata],
                documents=[summary_json]
            )
            self._invalidate_cache()
        except Exception as e:
            logger.warning(f"Error storing summary for {file_path}: {e}")
    
    def get_all_summarized_files(self) -> Dict[str, Dict[str, Any]]:
        """Get all summarized file paths and their metadata (cached)"""
        # Check cache
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < self._cache_ttl:
            return self._cache
        
        summarized_files = {}
        try:
            result = self.collection.get(include=["metadatas"])
            if result["ids"]:
                for i, file_path in enumerate(result["ids"]):
                    # Skip the schema version marker entry
                    if file_path == self._SCHEMA_VERSION_ID:
                        continue
                    summarized_files[file_path] = result["metadatas"][i]
            
            # Update cache
            self._cache = summarized_files
            self._cache_time = now
        except Exception as e:
            logger.warning(f"Error getting summarized files: {e}")
        return summarized_files
    
    def get_full_summary(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get the full structured summary for a file.
        
        This retrieves the JSON stored in the document field and parses it,
        along with the metadata fields.
        
        Args:
            file_path: Relative path to the file
            
        Returns:
            Full summary dict including all Phase 2 fields, or None if not found
        """
        import json
        
        try:
            result = self.collection.get(
                ids=[file_path],
                include=["documents", "metadatas"]
            )
            
            if not result["ids"]:
                return None
            
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            document = result["documents"][0] if result["documents"] else ""
            
            # Try to parse the document as JSON (new format)
            summary_data = None
            if document:
                try:
                    summary_data = json.loads(document)
                except json.JSONDecodeError:
                    # Old format: document is plain text, not JSON
                    # Fall back to extracting fields from metadata
                    summary_data = {
                        "purpose": document,  # Old format stored text here
                        "pattern": metadata.get("pattern", ""),
                        "domain": metadata.get("domain", ""),
                        "model_used": metadata.get("model", ""),
                    }
            
            if not summary_data:
                summary_data = {}
            
            # Merge metadata fields (in case they were updated separately)
            summary_data["content_hash"] = metadata.get("content_hash", "")
            summary_data["validation_status"] = metadata.get("validation_status", "unreviewed")
            summary_data["summarized_at"] = metadata.get("summarized_at", "")
            summary_data["summary_chunk_id"] = metadata.get("summary_chunk_id", "")
            
            # Add boolean flags from metadata
            summary_data["has_how_it_works"] = metadata.get("has_how_it_works", False)
            summary_data["has_method_summaries"] = metadata.get("has_method_summaries", False)
            
            return summary_data
            
        except Exception as e:
            logger.debug(f"Error getting full summary for {file_path}: {e}")
            return None

    def get_full_summaries_batch(self, file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get full structured summaries for multiple files in a single query.

        This is much more efficient than calling get_full_summary() in a loop
        as it makes a single ChromaDB query instead of N queries.

        Args:
            file_paths: List of relative file paths

        Returns:
            Dict mapping file_path -> summary data (or empty dict if not found)
        """
        import json

        if not file_paths:
            return {}

        try:
            result = self.collection.get(
                ids=file_paths,
                include=["documents", "metadatas"]
            )

            summaries = {}
            for i, file_path in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                document = result["documents"][i] if result["documents"] else ""

                # Parse document as JSON (new format)
                summary_data = None
                if document:
                    try:
                        summary_data = json.loads(document)
                    except json.JSONDecodeError:
                        # Old format: document is plain text
                        summary_data = {
                            "purpose": document,
                            "pattern": metadata.get("pattern", ""),
                            "domain": metadata.get("domain", ""),
                            "model_used": metadata.get("model", ""),
                        }

                if not summary_data:
                    summary_data = {}

                # Merge metadata fields
                summary_data["content_hash"] = metadata.get("content_hash", "")
                summary_data["validation_status"] = metadata.get("validation_status", "unreviewed")
                summary_data["summarized_at"] = metadata.get("summarized_at", "")
                summary_data["summary_chunk_id"] = metadata.get("summary_chunk_id", "")
                summary_data["has_how_it_works"] = metadata.get("has_how_it_works", False)
                summary_data["has_method_summaries"] = metadata.get("has_method_summaries", False)

                summaries[file_path] = summary_data

            return summaries

        except Exception as e:
            logger.debug(f"Error getting batch summaries: {e}")
            return {}

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        try:
            all_files = self.get_all_summarized_files()
            
            # Count by pattern
            patterns = {}
            domains = {}
            models = {}
            validation_statuses = {}
            simple_count = 0
            llm_count = 0
            
            for file_path, info in all_files.items():
                pattern = info.get("pattern", "unknown")
                domain = info.get("domain", "unknown")
                model = info.get("model", "unknown")
                validation_status = info.get("validation_status", "unreviewed")
                
                patterns[pattern] = patterns.get(pattern, 0) + 1
                domains[domain] = domains.get(domain, 0) + 1
                models[model] = models.get(model, 0) + 1
                validation_statuses[validation_status] = validation_statuses.get(validation_status, 0) + 1
                
                # Count simple vs LLM summarized
                if info.get("simple_file", False):
                    simple_count += 1
                else:
                    llm_count += 1
            
            return {
                "total_summarized": len(all_files),
                "simple_count": simple_count,
                "llm_count": llm_count,
                "by_pattern": patterns,
                "by_domain": domains,
                "by_model": models,
                "by_status": validation_statuses
            }
        except Exception as e:
            logger.warning(f"Error getting summary stats: {e}")
            return {"total_summarized": 0, "simple_count": 0, "llm_count": 0, "error": str(e)}
    
    def get_summary_stats_basic(self) -> Dict[str, Any]:
        """
        Get basic summary statistics (fast, count-only).
        
        This is much faster than get_summary_stats() as it only retrieves
        the total count using collection.count(), not all document metadata.
        
        Use this when you only need the total count, not detailed breakdowns
        by pattern, domain, model, etc.
        
        Returns:
            Dict with total_summarized count only
        """
        try:
            count = self.get_summary_count()
            return {
                "total_summarized": count,
                # These are unknown without full retrieval
                "simple_count": None,
                "llm_count": None,
            }
        except Exception as e:
            logger.warning(f"Error getting basic summary stats: {e}")
            return {"total_summarized": 0, "error": str(e)}
    
    def get_summary_count(self) -> int:
        """
        Get total count of summarized files (fast, uses collection.count()).
        
        This is much faster than get_summary_stats() as it doesn't retrieve
        all document metadata.
        
        Returns:
            Total number of summarized files
        """
        try:
            # collection.count() is very fast - O(1) operation
            count = self.collection.count()
            # Subtract 1 for the schema version marker if it exists
            # (The marker has ID "__schema_version__")
            if count > 0:
                # Check if schema version marker exists
                try:
                    result = self.collection.get(ids=[self._SCHEMA_VERSION_ID], include=[])
                    if result["ids"]:
                        count -= 1
                except Exception:
                    pass
            return max(0, count)
        except Exception as e:
            logger.warning(f"Error getting summary count: {e}")
            return 0


class ChromaVectorStore(VectorStore):
    """
    Chroma-based implementation of VectorStore for persistent vector storage
    and efficient similarity search.
    """

    def __init__(self,
                 collection_name: str = "memory_chunks",
                 persist_directory: Optional[str] = None,
                 host: Optional[str] = None,
                 port: Optional[int] = None):
        """
        Initialize Chroma vector store

        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory for persistent storage (local mode)
            host: Chroma server host (client mode)
            port: Chroma server port (client mode)
        """
        self.collection_name = collection_name

        # Determine mode: local persistence or client connection
        if host and port:
            # Client mode - connect to Chroma server
            self.client = chromadb.HttpClient(host=host, port=port)
            self.is_persistent = False
        else:
            # Local mode with persistence
            persist_dir = persist_directory or "./data/chroma"
            os.makedirs(persist_dir, exist_ok=True)

            self.client = chromadb.PersistentClient(path=persist_dir)
            self.is_persistent = True

        # Get or create collection with optimized HNSW settings
        # HNSW parameters:
        # - hnsw:space: distance metric (cosine for text embeddings)
        # - hnsw:M: max connections per node (higher = better recall, more memory)
        # - hnsw:construction_ef: build-time search width (higher = better index quality)
        # - hnsw:search_ef: query-time search width (higher = better recall, slower)
        collection_metadata = {
            "hnsw:space": "cosine",
            "hnsw:M": 32,  # Default is 16, higher improves recall
            "hnsw:construction_ef": 128,  # Default is 100
            "hnsw:search_ef": 64,  # Default is 10, we need good recall
        }
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing Chroma collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist or other error, create it
            logger.info(f"Creating new Chroma collection '{collection_name}': {e}")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            logger.info(f"Created new Chroma collection: {collection_name}")
        
        # Initialize file index metadata tracker
        self.file_index = FileIndexMetadata(self.client, f"{collection_name}_file_index")
        
        # Initialize summary index metadata tracker (for incremental re-summarization)
        self.summary_index = SummaryIndexMetadata(self.client, f"{collection_name}_summary_index")

    def add(self, chunk: MemoryChunk, embedding: List[float]) -> None:
        """
        Add a memory chunk with its embedding to the vector store

        Args:
            chunk: Memory chunk to store
            embedding: Vector embedding of the chunk
        """
        # Prepare metadata
        metadata = {
            "project_id": chunk.project_id,
            "role": chunk.role.value if hasattr(chunk.role, 'value') else str(chunk.role),
            "source": chunk.source,
            "tags": ",".join(chunk.tags) if chunk.tags else "",
            "relevance_score": chunk.relevance_score,
            "memory_type": chunk.memory_type.value if hasattr(chunk.memory_type, 'value') else str(chunk.memory_type),
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
            "updated_at": chunk.updated_at.isoformat() if chunk.updated_at else None,
            "expires_at": chunk.expires_at.isoformat() if chunk.expires_at else None,
        }

        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        # Add to collection
        self.collection.add(
            ids=[chunk.id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[chunk.doc_text]
        )

        logger.debug(f"Added chunk {chunk.id} to vector store")

    def add_batch(self, chunks: List[MemoryChunk], embeddings: List[List[float]]) -> None:
        """
        Add multiple memory chunks with their embeddings in a single batch.
        Much faster than calling add() repeatedly.

        Args:
            chunks: List of memory chunks to store
            embeddings: List of vector embeddings (same order as chunks)
        """
        if not chunks:
            return
        
        ids = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.doc_text)
            
            metadata = {
                "project_id": chunk.project_id,
                "role": chunk.role.value if hasattr(chunk.role, 'value') else str(chunk.role),
                "source": chunk.source,
                "tags": ",".join(chunk.tags) if chunk.tags else "",
                "relevance_score": chunk.relevance_score,
                "memory_type": chunk.memory_type.value if hasattr(chunk.memory_type, 'value') else str(chunk.memory_type),
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                "updated_at": chunk.updated_at.isoformat() if chunk.updated_at else None,
                "expires_at": chunk.expires_at.isoformat() if chunk.expires_at else None,
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)
        
        # Single batch insert
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.debug(f"Batch added {len(chunks)} chunks to vector store")

    def search(self, query: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """
        Search for similar chunks using vector similarity

        Args:
            query: Query embedding vector
            top_k: Number of top results to return
            filters: Optional metadata filters

        Returns:
            List of similar memory chunks
        """
        # Prepare where clause for filters
        where_clause = None
        if filters:
            where_clause = {}
            for key, value in filters.items():
                if key == "project_id":
                    where_clause["project_id"] = value
                elif key == "role":
                    where_clause["role"] = value
                elif key == "source":
                    where_clause["source"] = value
                elif key == "memory_type":
                    # Handle MemoryType enum or string
                    where_clause["memory_type"] = value.value if hasattr(value, 'value') else str(value)

        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "documents", "distances"]
        )

        # Convert results back to MemoryChunk objects
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                
                # Calculate relevance score from ChromaDB distance
                # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
                # Convert to similarity score (0-1, higher = more similar)
                relevance_score = 0.0
                if results.get("distances") and len(results["distances"]) > 0 and len(results["distances"][0]) > i:
                    distance = results["distances"][0][i]
                    # Convert cosine distance to similarity: similarity = 1 - (distance / 2)
                    # Clamp to [0, 1] range for safety
                    relevance_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                else:
                    # Fallback to stored relevance score if distances not available
                    relevance_score = float(metadata.get("relevance_score", 0.0))

                # Parse memory_type from stored string
                memory_type_str = metadata.get("memory_type", "code")
                try:
                    memory_type = MemoryType(memory_type_str)
                except (ValueError, KeyError):
                    memory_type = MemoryType.CODE
                
                # Reconstruct MemoryChunk from metadata
                chunk = MemoryChunk(
                    id=chunk_id,
                    project_id=metadata.get("project_id", ""),
                    role=metadata.get("role", ""),
                    prompt="",  # Not stored in vector DB
                    response="",  # Not stored in vector DB
                    doc_text=document,
                    embedding_id="",  # Not needed for reconstruction
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    pin=False,  # Not stored
                    relevance_score=relevance_score,
                    created_at=None,  # Would need proper datetime parsing
                    updated_at=None,  # Would need proper datetime parsing
                    source=metadata.get("source", ""),
                    memory_type=memory_type
                )
                chunks.append(chunk)

        logger.debug(f"Vector search returned {len(chunks)} results")
        return chunks

    def delete(self, chunk_id: str) -> None:
        """
        Delete a chunk from the vector store by ID

        Args:
            chunk_id: ID of chunk to delete
        """
        self.collection.delete(ids=[chunk_id])
        logger.debug(f"Deleted chunk {chunk_id} from vector store")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "is_persistent": self.is_persistent,
            "client_type": "http" if hasattr(self.client, '_client') else "persistent"
        }

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection (for testing/reset)
        """
        # Chroma doesn't have a direct clear method, so we delete all
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
                logger.info(f"Cleared {len(all_ids)} documents from collection")
        except Exception as e:
            logger.warning(f"Error clearing collection: {e}")

    def rebuild_from_chunks(self, chunks: List[MemoryChunk], embedder: Any) -> None:
        """
        Rebuild the vector store from a list of memory chunks

        Args:
            chunks: List of memory chunks to index
            embedder: Embedder instance to generate embeddings
        """
        logger.info(f"Rebuilding vector store with {len(chunks)} chunks")

        # Clear existing data
        self.clear_collection()

        # Process in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Generate embeddings for batch
            texts = [chunk.doc_text for chunk in batch]
            embeddings = embedder.generate_batch(texts)

            # Add batch to collection
            ids = [chunk.id for chunk in batch]
            metadatas = []
            documents = []

            for chunk in batch:
                metadata = {
                    "project_id": chunk.project_id,
                    "role": chunk.role.value if hasattr(chunk.role, 'value') else str(chunk.role),
                    "source": chunk.source,
                    "tags": ",".join(chunk.tags) if chunk.tags else "",
                    "relevance_score": chunk.relevance_score,
                }
                metadatas.append(metadata)
                documents.append(chunk.doc_text)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")

        logger.info("Vector store rebuild complete")
    
    def needs_reindex(self, file_path: str, mtime: float, content_hash: str) -> bool:
        """
        Check if a file needs to be re-indexed based on mtime or content hash
        
        Args:
            file_path: Relative path to the file
            mtime: File modification time
            content_hash: Hash of file content
            
        Returns:
            True if file needs re-indexing
        """
        file_info = self.file_index.get_file_info(file_path)
        if file_info is None:
            return True  # New file
        
        # Check if mtime changed
        stored_mtime = file_info.get("mtime", 0)
        if mtime != stored_mtime:
            # Double-check with content hash to avoid unnecessary re-indexing
            stored_hash = file_info.get("content_hash", "")
            if content_hash != stored_hash:
                return True
        
        return False
    
    def update_file_index(self, file_path: str, mtime: float, content_hash: str, chunk_ids: List[str]) -> None:
        """
        Update file index metadata after indexing
        
        Args:
            file_path: Relative path to the file
            mtime: File modification time
            content_hash: Hash of file content
            chunk_ids: List of chunk IDs created for this file
        """
        self.file_index.update_file_info(file_path, mtime, content_hash, chunk_ids)
    
    def update_file_index_batch(self, file_infos: List[Dict[str, Any]]) -> None:
        """
        Batch update file index metadata after indexing.
        Much faster than calling update_file_index() repeatedly.
        
        Args:
            file_infos: List of dicts with keys: file_path, mtime, content_hash, chunk_ids
        """
        self.file_index.update_file_info_batch(file_infos)
    
    def remove_file_chunks(self, file_path: str) -> int:
        """
        Remove all chunks associated with a file
        
        Args:
            file_path: Relative path to the file
            
        Returns:
            Number of chunks removed
        """
        chunk_ids = self.file_index.delete_file_info(file_path)
        if chunk_ids:
            try:
                self.collection.delete(ids=chunk_ids)
                logger.info(f"Removed {len(chunk_ids)} chunks for file: {file_path}")
            except Exception as e:
                logger.warning(f"Error removing chunks for {file_path}: {e}")
        return len(chunk_ids)
    
    def get_indexed_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all indexed files and their metadata
        
        Returns:
            Dictionary mapping file paths to their index metadata
        """
        return self.file_index.get_all_indexed_files()
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def cleanup_orphan_chunks(self, codebase_name: str) -> int:
        """
        Remove chunks that exist in ChromaDB but have no corresponding file index entry.
        This handles recovery from crashes during indexing.
        
        Args:
            codebase_name: Name of the codebase to clean up
            
        Returns:
            Number of orphan chunks removed
        """
        # Get all chunk IDs tracked in file index
        indexed_files = self.file_index.get_all_indexed_files()
        tracked_chunk_ids = set()
        for file_info in indexed_files.values():
            chunk_ids_str = file_info.get("chunk_ids", "")
            if chunk_ids_str:
                tracked_chunk_ids.update(chunk_ids_str.split(","))
        
        # Get all chunk IDs in ChromaDB for this codebase
        try:
            result = self.collection.get(
                where={"project_id": codebase_name},
                include=[]  # We only need IDs
            )
            all_chunk_ids = set(result["ids"]) if result["ids"] else set()
        except Exception as e:
            logger.warning(f"Error getting chunks for cleanup: {e}")
            return 0
        
        # Find orphan chunks (in ChromaDB but not in file index)
        orphan_ids = all_chunk_ids - tracked_chunk_ids
        
        if orphan_ids:
            try:
                self.collection.delete(ids=list(orphan_ids))
                logger.info(f"[{codebase_name}] Cleaned up {len(orphan_ids)} orphan chunks from previous incomplete indexing")
            except Exception as e:
                logger.warning(f"Error cleaning up orphan chunks: {e}")
                return 0
        
        return len(orphan_ids)