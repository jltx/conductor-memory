"""
Hybrid Search - Combines semantic (vector) search with keyword (BM25) search.

Uses Reciprocal Rank Fusion (RRF) to combine results from both approaches,
providing better results for queries that benefit from exact keyword matching
while maintaining semantic understanding.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from rank_bm25 import BM25Okapi

from ..core.models import MemoryChunk

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search mode selection"""
    SEMANTIC = "semantic"      # Vector similarity only
    KEYWORD = "keyword"        # BM25 keyword only
    HYBRID = "hybrid"          # Combined (default)
    AUTO = "auto"              # Auto-detect based on query


@dataclass
class HybridSearchResult:
    """Result from hybrid search with score breakdown"""
    chunk: MemoryChunk
    combined_score: float
    semantic_score: float
    keyword_score: float
    semantic_rank: int
    keyword_rank: int


class BM25Index:
    """
    BM25 keyword index for a collection of documents.
    
    Maintains an in-memory BM25 index that can be rebuilt when documents change.
    """
    
    def __init__(self):
        self._documents: Dict[str, str] = {}  # id -> text
        self._chunk_map: Dict[str, MemoryChunk] = {}  # id -> chunk
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []  # Ordered list of doc IDs matching BM25 corpus
        self._tokenized_corpus: List[List[str]] = []
    
    def add_document(self, chunk: MemoryChunk) -> None:
        """Add a document to the index (requires rebuild)"""
        self._documents[chunk.id] = chunk.doc_text
        self._chunk_map[chunk.id] = chunk
        self._bm25 = None  # Invalidate index
    
    def remove_document(self, chunk_id: str) -> None:
        """Remove a document from the index (requires rebuild)"""
        if chunk_id in self._documents:
            del self._documents[chunk_id]
            del self._chunk_map[chunk_id]
            self._bm25 = None  # Invalidate index
    
    def clear(self) -> None:
        """Clear all documents"""
        self._documents.clear()
        self._chunk_map.clear()
        self._bm25 = None
        self._doc_ids.clear()
        self._tokenized_corpus.clear()
    
    def build(self) -> None:
        """Build or rebuild the BM25 index"""
        if not self._documents:
            self._bm25 = None
            return
        
        self._doc_ids = list(self._documents.keys())
        self._tokenized_corpus = [
            self._tokenize(self._documents[doc_id])
            for doc_id in self._doc_ids
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.debug(f"Built BM25 index with {len(self._doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[MemoryChunk, float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples, sorted by score descending
        """
        if self._bm25 is None:
            self.build()
        
        if self._bm25 is None or not self._doc_ids:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        scored_docs = list(zip(self._doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scored_docs[:top_k]:
            if score > 0:  # Only include documents with positive scores
                chunk = self._chunk_map.get(doc_id)
                if chunk:
                    results.append((chunk, score))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Uses a code-aware tokenization that:
        - Splits on whitespace and punctuation
        - Preserves snake_case and camelCase components
        - Lowercases for matching
        - Removes very short tokens
        """
        # Split camelCase: insertBefore -> insert Before
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Split snake_case: insert_before -> insert before
        text = text.replace('_', ' ')
        
        # Split on non-alphanumeric (but keep the tokens)
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        
        # Filter very short tokens (likely noise)
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
    
    @property
    def document_count(self) -> int:
        """Number of documents in the index"""
        return len(self._documents)


class HybridSearcher:
    """
    Combines semantic search results with BM25 keyword search.
    
    Uses Reciprocal Rank Fusion (RRF) for score combination:
    RRF_score = sum(1 / (k + rank)) for each ranking
    
    This approach is robust to different score scales and works well
    in practice for combining heterogeneous retrieval methods.
    """
    
    def __init__(self, rrf_k: int = 60, semantic_weight: float = 0.5):
        """
        Initialize hybrid searcher.
        
        Args:
            rrf_k: RRF constant (default 60, standard value)
            semantic_weight: Weight for semantic results (0-1), keyword gets 1-weight
        """
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self._bm25_indices: Dict[str, BM25Index] = {}  # codebase -> index
    
    def get_or_create_index(self, codebase: str) -> BM25Index:
        """Get or create BM25 index for a codebase"""
        if codebase not in self._bm25_indices:
            self._bm25_indices[codebase] = BM25Index()
        return self._bm25_indices[codebase]
    
    def add_to_index(self, codebase: str, chunk: MemoryChunk) -> None:
        """Add a chunk to the BM25 index for a codebase"""
        index = self.get_or_create_index(codebase)
        index.add_document(chunk)
    
    def remove_from_index(self, codebase: str, chunk_id: str) -> None:
        """Remove a chunk from the BM25 index"""
        if codebase in self._bm25_indices:
            self._bm25_indices[codebase].remove_document(chunk_id)
    
    def clear_index(self, codebase: str) -> None:
        """Clear BM25 index for a codebase"""
        if codebase in self._bm25_indices:
            self._bm25_indices[codebase].clear()
    
    def rebuild_index(self, codebase: str) -> None:
        """Rebuild BM25 index for a codebase"""
        if codebase in self._bm25_indices:
            self._bm25_indices[codebase].build()
    
    def keyword_search(
        self,
        query: str,
        codebase: Optional[str] = None,
        top_k: int = 10
    ) -> List[Tuple[MemoryChunk, float]]:
        """
        Perform keyword-only search using BM25.
        
        Args:
            query: Search query
            codebase: Specific codebase to search (None = all)
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        results = []
        
        if codebase:
            if codebase in self._bm25_indices:
                results = self._bm25_indices[codebase].search(query, top_k)
        else:
            # Search all codebases
            for cb_name, index in self._bm25_indices.items():
                cb_results = index.search(query, top_k)
                results.extend(cb_results)
            
            # Sort combined results and take top_k
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        semantic_results: List[MemoryChunk],
        codebase: Optional[str] = None,
        top_k: int = 10
    ) -> List[HybridSearchResult]:
        """
        Combine semantic search results with keyword search using RRF.
        
        Args:
            query: Search query
            semantic_results: Results from vector similarity search (already ranked)
            codebase: Specific codebase (None = all)
            top_k: Number of final results
            
        Returns:
            List of HybridSearchResult with combined scores
        """
        # Get keyword search results
        keyword_results = self.keyword_search(query, codebase, top_k * 2)
        
        # Build rank maps
        semantic_ranks = {chunk.id: i + 1 for i, chunk in enumerate(semantic_results)}
        keyword_ranks = {chunk.id: i + 1 for i, (chunk, _) in enumerate(keyword_results)}
        
        # Build score maps (normalized)
        semantic_scores = {}
        if semantic_results:
            max_sem = max(c.relevance_score for c in semantic_results) or 1.0
            semantic_scores = {c.id: c.relevance_score / max_sem for c in semantic_results}
        
        keyword_scores = {}
        if keyword_results:
            max_kw = max(score for _, score in keyword_results) or 1.0
            keyword_scores = {chunk.id: score / max_kw for chunk, score in keyword_results}
        
        # Collect all unique chunks
        all_chunks: Dict[str, MemoryChunk] = {}
        for chunk in semantic_results:
            all_chunks[chunk.id] = chunk
        for chunk, _ in keyword_results:
            if chunk.id not in all_chunks:
                all_chunks[chunk.id] = chunk
        
        # Calculate RRF scores
        results = []
        for chunk_id, chunk in all_chunks.items():
            sem_rank = semantic_ranks.get(chunk_id, len(semantic_results) + 100)
            kw_rank = keyword_ranks.get(chunk_id, len(keyword_results) + 100)
            
            # RRF formula with weights
            sem_rrf = self.semantic_weight * (1.0 / (self.rrf_k + sem_rank))
            kw_rrf = self.keyword_weight * (1.0 / (self.rrf_k + kw_rank))
            combined_score = sem_rrf + kw_rrf
            
            results.append(HybridSearchResult(
                chunk=chunk,
                combined_score=combined_score,
                semantic_score=semantic_scores.get(chunk_id, 0.0),
                keyword_score=keyword_scores.get(chunk_id, 0.0),
                semantic_rank=sem_rank if chunk_id in semantic_ranks else -1,
                keyword_rank=kw_rank if chunk_id in keyword_ranks else -1
            ))
        
        # Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)
        
        return results[:top_k]
    
    def detect_search_mode(self, query: str) -> SearchMode:
        """
        Auto-detect the best search mode for a query.
        
        Heuristics:
        - Queries with exact identifiers (CamelCase, snake_case) -> prefer keyword
        - Conceptual questions (how, what, why) -> prefer semantic
        - Mixed or unclear -> hybrid
        
        Args:
            query: The search query
            
        Returns:
            Recommended SearchMode
        """
        # Check for code identifiers
        has_camel_case = bool(re.search(r'[a-z][A-Z]', query))
        has_snake_case = bool(re.search(r'[a-z]_[a-z]', query))
        has_exact_quotes = '"' in query or "'" in query
        
        # Check for conceptual keywords
        conceptual_words = ['how', 'what', 'why', 'when', 'where', 'which', 
                          'explain', 'describe', 'pattern', 'approach', 
                          'related to', 'similar to', 'like']
        query_lower = query.lower()
        is_conceptual = any(word in query_lower for word in conceptual_words)
        
        # Decision logic
        if has_exact_quotes:
            return SearchMode.KEYWORD
        elif (has_camel_case or has_snake_case) and not is_conceptual:
            return SearchMode.KEYWORD
        elif is_conceptual and not (has_camel_case or has_snake_case):
            return SearchMode.SEMANTIC
        else:
            return SearchMode.HYBRID

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 indices"""
        stats = {}
        for codebase, index in self._bm25_indices.items():
            stats[codebase] = {
                "document_count": index.document_count,
                "index_built": index._bm25 is not None
            }
        return stats
