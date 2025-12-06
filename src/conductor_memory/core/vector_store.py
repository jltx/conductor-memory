"""
Vector Store Interface for the Hybrid Local/Cloud LLM Orchestrator
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from .models import MemoryChunk

class VectorStore(ABC):
    """Abstract interface for vector storage operations"""
    
    @abstractmethod
    def add(self, chunk: MemoryChunk, embedding: List[float]) -> None:
        """Add a memory chunk with its embedding to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query: List[float], top_k: int) -> List[MemoryChunk]:
        """Search for similar chunks using vector similarity"""
        pass
    
    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Delete a chunk from the vector store by ID"""
        pass