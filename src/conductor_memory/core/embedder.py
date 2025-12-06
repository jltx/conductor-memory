"""
Embedder Interface for the Hybrid Local/Cloud LLM Orchestrator
"""

from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):
    """Abstract interface for embedding generation"""
    
    @abstractmethod
    def generate(self, text: str) -> List[float]:
        """Generate a single embedding for the given text"""
        pass
    
    @abstractmethod
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        pass