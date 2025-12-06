"""
Conductor Memory - Semantic memory service with codebase indexing for AI agents

This package provides:
- Semantic search across codebases using vector similarity
- Hybrid search combining semantic and keyword (BM25) matching
- Multi-codebase support with per-codebase indexing
- MCP server integration for AI agent tools
- HTTP REST API for programmatic access

Quick Start:
    from conductor_memory import MemoryService, ServerConfig
    
    config = ServerConfig.create_default(codebase_path="/path/to/code")
    service = MemoryService(config)
    service.initialize()
    
    results = service.search("how does authentication work")
"""

from typing import TYPE_CHECKING

# Core types - lightweight, always available
from .core.models import MemoryChunk, RoleEnum, MemoryType, MemoryDB
from .core.vector_store import VectorStore
from .core.embedder import Embedder
from .core.relevance import RelevanceCalculator, CosineRelevanceCalculator

# Configuration
from .config.vector_db import VectorDBConfig
from .config.server import ServerConfig, CodebaseConfig

# Type hints only - not imported at runtime for faster startup
if TYPE_CHECKING:
    from .service.memory_service import MemoryService
    from .storage.chroma import ChromaVectorStore
    from .embedding.sentence_transformer import SentenceTransformerEmbedder
    from .search.hybrid import HybridSearcher, BM25Index, SearchMode, HybridSearchResult


# Lazy loaders for heavy modules
def get_memory_service():
    """Lazy import of MemoryService (loads ChromaDB, SentenceTransformers)"""
    from .service.memory_service import MemoryService
    return MemoryService


def get_chroma_vector_store():
    """Lazy import of ChromaVectorStore"""
    from .storage.chroma import ChromaVectorStore
    return ChromaVectorStore


def get_sentence_transformer_embedder():
    """Lazy import of SentenceTransformerEmbedder (loads TensorFlow)"""
    from .embedding.sentence_transformer import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder


def get_hybrid_searcher():
    """Lazy import of HybridSearcher"""
    from .search.hybrid import HybridSearcher, BM25Index, SearchMode, HybridSearchResult
    return HybridSearcher, BM25Index, SearchMode, HybridSearchResult


# Lazy module for attribute access
class _LazyModule:
    """Wrapper to provide lazy-loaded module attributes"""
    
    @property
    def MemoryService(self):
        return get_memory_service()
    
    @property
    def ChromaVectorStore(self):
        return get_chroma_vector_store()
    
    @property
    def SentenceTransformerEmbedder(self):
        return get_sentence_transformer_embedder()
    
    @property
    def HybridSearcher(self):
        return get_hybrid_searcher()[0]
    
    @property
    def BM25Index(self):
        return get_hybrid_searcher()[1]
    
    @property
    def SearchMode(self):
        return get_hybrid_searcher()[2]
    
    @property
    def HybridSearchResult(self):
        return get_hybrid_searcher()[3]


_lazy = _LazyModule()


def __getattr__(name):
    """Module-level __getattr__ for lazy loading"""
    if hasattr(_lazy, name):
        return getattr(_lazy, name)
    raise AttributeError(f"module 'conductor_memory' has no attribute '{name}'")


__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    
    # Core types
    "MemoryDB", "MemoryChunk", "RoleEnum", "MemoryType",
    "VectorStore", "Embedder", "RelevanceCalculator", "CosineRelevanceCalculator",
    
    # Configuration
    "VectorDBConfig", "ServerConfig", "CodebaseConfig",
    
    # Lazy-loaded (use get_* functions for explicit loading)
    "get_memory_service",
    "get_chroma_vector_store",
    "get_sentence_transformer_embedder",
    "get_hybrid_searcher",
    
    # Backwards compatibility (lazy-loaded via __getattr__)
    "MemoryService",
    "ChromaVectorStore", "SentenceTransformerEmbedder",
    "HybridSearcher", "BM25Index", "SearchMode", "HybridSearchResult",
]
