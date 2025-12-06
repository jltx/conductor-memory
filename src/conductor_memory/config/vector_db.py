"""
Configuration for Vector Database and Embedding components
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class VectorDBConfig:
    """Configuration for vector database connection"""
    # Chroma settings
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    chroma_persist_directory: str = "./data/chroma"
    chroma_collection_name: str = "memory_chunks"

    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None  # Auto-detect
    embedding_cache_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'VectorDBConfig':
        """Create config from environment variables"""
        return cls(
            chroma_host=os.getenv("CHROMA_HOST"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")) if os.getenv("CHROMA_PORT") else None,
            chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION", "memory_chunks"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_device=os.getenv("EMBEDDING_DEVICE"),
            embedding_cache_dir=os.getenv("EMBEDDING_CACHE_DIR")
        )

    def is_server_mode(self) -> bool:
        """Check if running in server/client mode vs local persistence"""
        return self.chroma_host is not None and self.chroma_port is not None