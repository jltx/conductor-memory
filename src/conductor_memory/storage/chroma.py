"""
Chroma Vector Store Implementation for the Hybrid Local/Cloud LLM Orchestrator

Provides persistent vector storage using Chroma database with efficient similarity search.
"""

import os
import logging
import hashlib
from typing import List, Optional, Dict, Any
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
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing file index metadata collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new file index metadata collection: {collection_name}")
    
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
        except Exception as e:
            logger.warning(f"Error updating file index metadata for {file_path}: {e}")
    
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
        except Exception as e:
            logger.warning(f"Error deleting file index metadata for {file_path}: {e}")
        return chunk_ids
    
    def get_all_indexed_files(self) -> Dict[str, Dict[str, Any]]:
        """Get all indexed file paths and their metadata"""
        indexed_files = {}
        try:
            result = self.collection.get(include=["metadatas"])
            if result["ids"]:
                for i, file_path in enumerate(result["ids"]):
                    indexed_files[file_path] = result["metadatas"][i]
        except Exception as e:
            logger.warning(f"Error getting indexed files: {e}")
        return indexed_files


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

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing Chroma collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist or other error, create it
            logger.info(f"Creating new Chroma collection '{collection_name}': {e}")
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new Chroma collection: {collection_name}")
        
        # Initialize file index metadata tracker
        self.file_index = FileIndexMetadata(self.client, f"{collection_name}_file_index")

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
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                
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