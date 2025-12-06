"""
Search result boosting based on metadata, recency, and query context.
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ..core.models import MemoryChunk, MemoryType
from ..config.server import BoostConfig

logger = logging.getLogger(__name__)


class BoostCalculator:
    """Calculates boost factors for search results"""
    
    def __init__(self, config: BoostConfig):
        self.config = config
    
    def calculate_boost(
        self, 
        chunk: MemoryChunk, 
        query_domain_boosts: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate total boost factor for a chunk.
        
        Args:
            chunk: The memory chunk to boost
            query_domain_boosts: Per-query domain boost overrides
            
        Returns:
            Boost factor (1.0 = no change, >1.0 = boost, <1.0 = penalize)
        """
        boost = 1.0
        
        # Apply domain boost
        domain_boost = self._get_domain_boost(chunk, query_domain_boosts)
        boost *= domain_boost
        
        # Apply memory type boost
        memory_type_boost = self._get_memory_type_boost(chunk)
        boost *= memory_type_boost
        
        # Apply recency boost
        if self.config.recency_enabled:
            recency_boost = self._get_recency_boost(chunk)
            boost *= recency_boost
        
        logger.debug(f"Chunk {chunk.id[:8]}: domain={domain_boost:.2f}, type={memory_type_boost:.2f}, recency={recency_boost:.2f}, total={boost:.2f}")
        
        return boost
    
    def _get_domain_boost(
        self, 
        chunk: MemoryChunk, 
        query_overrides: Optional[Dict[str, float]]
    ) -> float:
        """Get domain-based boost factor"""
        # Extract domain from tags
        domain = None
        for tag in chunk.tags:
            if tag.startswith("domain:"):
                domain = tag[7:]  # Remove "domain:" prefix
                break
        
        if not domain:
            return 1.0
        
        # Use query-specific overrides if provided
        if query_overrides and domain in query_overrides:
            logger.debug(f"Using query override for domain '{domain}': {query_overrides[domain]}")
            return query_overrides[domain]
        
        # Use config defaults
        boost = self.config.domain_boosts.get(domain, 1.0)
        logger.debug(f"Using config default for domain '{domain}': {boost}")
        return boost
    
    def _get_memory_type_boost(self, chunk: MemoryChunk) -> float:
        """Get memory type boost factor"""
        memory_type = chunk.memory_type.value if hasattr(chunk.memory_type, 'value') else str(chunk.memory_type)
        boost = self.config.memory_type_boosts.get(memory_type, 1.0)
        logger.debug(f"Memory type '{memory_type}' boost: {boost}")
        return boost
    
    def _get_recency_boost(self, chunk: MemoryChunk) -> float:
        """
        Calculate recency boost using exponential decay.
        
        Formula: boost = min_boost + (max_boost - min_boost) * exp(-age_days / decay_days)
        """
        if not chunk.created_at:
            logger.debug("No created_at timestamp, using neutral recency boost")
            return 1.0  # No recency info, neutral boost
        
        age_days = (datetime.now() - chunk.created_at).total_seconds() / 86400
        
        # Exponential decay
        decay_factor = math.exp(-age_days / self.config.recency_decay_days)
        
        # Scale between min and max boost
        boost_range = self.config.recency_max_boost - self.config.recency_min_boost
        boost = self.config.recency_min_boost + boost_range * decay_factor
        
        logger.debug(f"Age: {age_days:.1f} days, decay_factor: {decay_factor:.3f}, recency boost: {boost:.2f}")
        
        return boost
    
    def apply_boosts_to_chunks(
        self, 
        chunks: List[MemoryChunk], 
        query_domain_boosts: Optional[Dict[str, float]] = None
    ) -> List[MemoryChunk]:
        """
        Apply boost factors to a list of chunks, modifying their relevance scores.
        
        Args:
            chunks: List of memory chunks to boost
            query_domain_boosts: Per-query domain boost overrides
            
        Returns:
            The same list of chunks with modified relevance scores
        """
        for chunk in chunks:
            original_score = chunk.relevance_score
            boost_factor = self.calculate_boost(chunk, query_domain_boosts)
            chunk.relevance_score *= boost_factor
            
            logger.debug(f"Boosted chunk {chunk.id[:8]}: {original_score:.3f} -> {chunk.relevance_score:.3f} (boost: {boost_factor:.2f})")
        
        return chunks