"""
Relevance Calculator for the Hybrid Local/Cloud LLM Orchestrator
"""

from abc import ABC, abstractmethod
from typing import List
import math

class RelevanceCalculator(ABC):
    """Abstract interface for calculating relevance scores between query and chunk embeddings"""
    
    @abstractmethod
    def calculate(self, query_embedding: List[float], chunk_embeddings: List[List[float]]) -> float:
        """Calculate normalized relevance score (0-1) between query and chunk embeddings"""
        pass

class CosineRelevanceCalculator(RelevanceCalculator):
    """Concrete implementation using cosine similarity for relevance scoring"""
    
    def calculate(self, query_embedding: List[float], chunk_embeddings: List[List[float]]) -> float:
        """
        Calculate cosine similarity between query and all chunk embeddings
        Returns a normalized score between 0 and 1
        """
        if not chunk_embeddings:
            return 0.0
            
        # Calculate cosine similarity for each chunk embedding
        similarities = []
        for chunk_embedding in chunk_embeddings:
            dot_product = sum(q * c for q, c in zip(query_embedding, chunk_embedding))
            query_magnitude = math.sqrt(sum(q * q for q in query_embedding))
            chunk_magnitude = math.sqrt(sum(c * c for c in chunk_embedding))
            
            if query_magnitude == 0 or chunk_magnitude == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_magnitude * chunk_magnitude)
            
            similarities.append(similarity)
        
        # Return the maximum similarity as the relevance score
        max_similarity = max(similarities) if similarities else 0.0
        
        # Normalize to 0-1 range (cosine similarity is already in -1 to 1, so we shift and scale)
        normalized_score = (max_similarity + 1) / 2
        
        return normalized_score