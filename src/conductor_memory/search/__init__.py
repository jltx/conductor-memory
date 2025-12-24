"""Search implementations for conductor-memory"""

from .hybrid import HybridSearcher, BM25Index, SearchMode, HybridSearchResult
from .chunking import ChunkingManager, ChunkingStrategy, ChunkMetadata
from .verification import (
    VerificationStatus,
    VerificationIntent,
    Evidence,
    SubjectInfo,
    VerificationInfo,
    VerificationResult,
    parse_verification_query,
    extract_key_terms,
    is_verification_query,
    find_evidence,
    matches_any,
    calculate_relevance,
)

__all__ = [
    "HybridSearcher", "BM25Index", "SearchMode", "HybridSearchResult",
    "ChunkingManager", "ChunkingStrategy", "ChunkMetadata",
    "VerificationStatus", "VerificationIntent", "Evidence",
    "SubjectInfo", "VerificationInfo", "VerificationResult",
    "parse_verification_query", "extract_key_terms", "is_verification_query",
    "find_evidence", "matches_any", "calculate_relevance",
]
