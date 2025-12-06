# Conductor-Memory Enhancement Implementation Plan

## Overview

This plan implements three major enhancements to the conductor-memory search system:

1. **Metadata-Based Boosting** - Score adjustments based on domain, memory type, and recency
2. **Tag-Based Filtering** - Include/exclude filtering with prefix support
3. **Reranking Stage** - Two-stage retrieval with cross-encoder reranking

## Design Principles

- **Backward Compatible**: Existing API calls continue to work unchanged
- **Configurable**: Hardcoded defaults with optional config overrides
- **Composable**: Users can enable/disable features independently
- **Agent-Friendly**: Simple parameters that agents can use intelligently

---

## 1. Metadata-Based Boosting

### 1.1 Configuration Structure

Add to `ServerConfig` in `config/server.py`:

```python
@dataclass
class BoostConfig:
    """Configuration for search result boosting"""
    
    # Domain-based boosts (applied to code chunks)
    domain_boosts: Dict[str, float] = field(default_factory=lambda: {
        "class": 1.2,      # Classes are often important entry points
        "function": 1.1,   # Functions are core logic
        "imports": 0.9,    # Imports less relevant for most queries
        "test": 0.7,       # Tests usually not what users want
        "private": 0.8,    # Private methods less relevant
        "accessor": 1.0    # Getters/setters neutral
    })
    
    # Memory type boosts
    memory_type_boosts: Dict[str, float] = field(default_factory=lambda: {
        "code": 1.1,           # Code chunks slightly boosted (default)
        "decision": 1.3,       # Architectural decisions highly valuable
        "lesson": 1.2,         # Lessons learned valuable
        "conversation": 1.0    # Conversations neutral
    })
    
    # Recency boost settings
    recency_enabled: bool = True
    recency_decay_days: float = 30.0    # Half-life for recency boost
    recency_max_boost: float = 1.5      # Maximum boost for very recent items
    recency_min_boost: float = 0.8      # Minimum boost for very old items

@dataclass 
class ServerConfig:
    # ... existing fields ...
    
    # Boost configuration
    boost_config: BoostConfig = field(default_factory=BoostConfig)
```

### 1.2 Boost Calculator

Create `src/conductor_memory/search/boosting.py`:

```python
"""
Search result boosting based on metadata, recency, and query context.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ..core.models import MemoryChunk, MemoryType
from ..config.server import BoostConfig

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
            return query_overrides[domain]
        
        # Use config defaults
        return self.config.domain_boosts.get(domain, 1.0)
    
    def _get_memory_type_boost(self, chunk: MemoryChunk) -> float:
        """Get memory type boost factor"""
        memory_type = chunk.memory_type.value if hasattr(chunk.memory_type, 'value') else str(chunk.memory_type)
        return self.config.memory_type_boosts.get(memory_type, 1.0)
    
    def _get_recency_boost(self, chunk: MemoryChunk) -> float:
        """
        Calculate recency boost using exponential decay.
        
        Formula: boost = min_boost + (max_boost - min_boost) * exp(-age_days / decay_days)
        """
        if not chunk.created_at:
            return 1.0  # No recency info, neutral boost
        
        age_days = (datetime.now() - chunk.created_at).total_seconds() / 86400
        
        # Exponential decay
        decay_factor = math.exp(-age_days / self.config.recency_decay_days)
        
        # Scale between min and max boost
        boost_range = self.config.recency_max_boost - self.config.recency_min_boost
        boost = self.config.recency_min_boost + boost_range * decay_factor
        
        return boost
```

### 1.3 Integration into Search

Modify `MemoryService.search_async()` in `service/memory_service.py`:

```python
# Add to imports
from ..search.boosting import BoostCalculator

class MemoryService:
    def __init__(self, config: ServerConfig):
        # ... existing init ...
        self.boost_calculator = BoostCalculator(config.boost_config)
    
    async def search_async(
        self,
        query: str,
        codebase: Optional[str] = None,
        max_results: int = 10,
        project_id: Optional[str] = None,
        search_mode: str = "auto",
        # NEW PARAMETERS
        domain_boosts: Optional[Dict[str, float]] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        use_reranker: bool = True
    ) -> Dict[str, Any]:
        # ... existing search logic until we have similar_chunks ...
        
        # Apply boosting to chunks
        for chunk in similar_chunks:
            boost_factor = self.boost_calculator.calculate_boost(
                chunk, 
                query_domain_boosts=domain_boosts
            )
            chunk.relevance_score *= boost_factor
        
        # Apply tag filtering
        if include_tags or exclude_tags:
            similar_chunks = self._filter_by_tags(
                similar_chunks, 
                include_tags, 
                exclude_tags
            )
        
        # Apply reranking if enabled
        if use_reranker and len(similar_chunks) > 1:
            similar_chunks = await self._rerank_chunks(query, similar_chunks)
        
        # ... rest of existing logic ...
```

---

## 2. Tag-Based Filtering

### 2.1 Tag Filter Implementation

Add to `MemoryService` in `service/memory_service.py`:

```python
def _filter_by_tags(
    self, 
    chunks: List[MemoryChunk], 
    include_tags: Optional[List[str]], 
    exclude_tags: Optional[List[str]]
) -> List[MemoryChunk]:
    """
    Filter chunks by tag inclusion/exclusion with prefix support.
    
    Tag patterns:
    - "exact_tag" - exact match
    - "prefix:*" - matches any tag starting with "prefix:"
    """
    if not include_tags and not exclude_tags:
        return chunks
    
    filtered_chunks = []
    
    for chunk in chunks:
        chunk_tags = set(chunk.tags)
        
        # Check include tags (must match at least one)
        if include_tags:
            include_match = False
            for include_tag in include_tags:
                if self._tag_matches(include_tag, chunk_tags):
                    include_match = True
                    break
            if not include_match:
                continue
        
        # Check exclude tags (must not match any)
        if exclude_tags:
            exclude_match = False
            for exclude_tag in exclude_tags:
                if self._tag_matches(exclude_tag, chunk_tags):
                    exclude_match = True
                    break
            if exclude_match:
                continue
        
        filtered_chunks.append(chunk)
    
    return filtered_chunks

def _tag_matches(self, pattern: str, chunk_tags: set) -> bool:
    """Check if a tag pattern matches any chunk tags"""
    if pattern.endswith("*"):
        # Prefix match
        prefix = pattern[:-1]
        return any(tag.startswith(prefix) for tag in chunk_tags)
    else:
        # Exact match
        return pattern in chunk_tags
```

---

## 3. Reranking Stage

### 3.1 Reranker Selection Strategy

Based on research, implement a tiered approach:

1. **Local Cross-Encoder** (default) - Fast, no API costs
2. **BGE Reranker** (optional) - Better quality, still local
3. **Cohere Rerank** (optional) - Best quality, requires API key

### 3.2 Reranker Configuration

Add to `ServerConfig`:

```python
@dataclass
class RerankerConfig:
    """Configuration for reranking"""
    
    # Reranker type: "cross-encoder", "bge-base", "bge-large", "cohere"
    reranker_type: str = "cross-encoder"
    
    # Model names for each type
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bge_model: str = "BAAI/bge-reranker-base"
    
    # Cohere API settings
    cohere_api_key: Optional[str] = None
    cohere_model: str = "rerank-english-v3.0"
    
    # Retrieval settings
    retrieve_k: int = 20  # Retrieve more candidates for reranking
    rerank_top_k: int = 10  # Final results after reranking
    
    # Performance settings
    reranker_batch_size: int = 32
    reranker_device: str = "auto"  # auto, cpu, cuda, mps

@dataclass
class ServerConfig:
    # ... existing fields ...
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)
```

### 3.3 Reranker Implementation

Create `src/conductor_memory/search/reranking.py`:

```python
"""
Reranking implementation for improved search quality.
"""

import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from ..core.models import MemoryChunk
from ..config.server import RerankerConfig

logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """Abstract base class for rerankers"""
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        chunks: List[MemoryChunk]
    ) -> List[MemoryChunk]:
        """Rerank chunks based on query relevance"""
        pass

class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker using sentence-transformers"""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model = None
    
    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.config.cross_encoder_model,
                    device=self.config.reranker_device
                )
                logger.info(f"Loaded cross-encoder: {self.config.cross_encoder_model}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers"
                )
    
    async def rerank(
        self, 
        query: str, 
        chunks: List[MemoryChunk]
    ) -> List[MemoryChunk]:
        """Rerank using cross-encoder"""
        if len(chunks) <= 1:
            return chunks
        
        self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, chunk.doc_text) for chunk in chunks]
        
        # Get relevance scores
        scores = self._model.predict(pairs)
        
        # Sort by score (descending)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Update relevance scores and return
        reranked_chunks = []
        for chunk, score in scored_chunks:
            chunk.relevance_score = float(score)
            reranked_chunks.append(chunk)
        
        return reranked_chunks

class BGEReranker(BaseReranker):
    """BGE reranker implementation"""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load BGE model"""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.bge_model)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.config.bge_model)
                
                # Move to appropriate device
                if self.config.reranker_device != "cpu":
                    device = torch.device(self.config.reranker_device)
                    self._model.to(device)
                
                logger.info(f"Loaded BGE reranker: {self.config.bge_model}")
            except ImportError:
                raise ImportError(
                    "transformers and torch required for BGE reranking. "
                    "Install with: pip install transformers torch"
                )
    
    async def rerank(
        self, 
        query: str, 
        chunks: List[MemoryChunk]
    ) -> List[MemoryChunk]:
        """Rerank using BGE model"""
        if len(chunks) <= 1:
            return chunks
        
        self._load_model()
        
        import torch
        
        # Prepare inputs
        pairs = [f"{query} [SEP] {chunk.doc_text}" for chunk in chunks]
        
        # Tokenize
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Move to device
        if self.config.reranker_device != "cpu":
            device = torch.device(self.config.reranker_device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get scores
        with torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        # Sort and return
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        reranked_chunks = []
        for chunk, score in scored_chunks:
            chunk.relevance_score = float(score)
            reranked_chunks.append(chunk)
        
        return reranked_chunks

class CohereReranker(BaseReranker):
    """Cohere API-based reranker"""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        if not config.cohere_api_key:
            raise ValueError("Cohere API key required for Cohere reranker")
    
    async def rerank(
        self, 
        query: str, 
        chunks: List[MemoryChunk]
    ) -> List[MemoryChunk]:
        """Rerank using Cohere API"""
        if len(chunks) <= 1:
            return chunks
        
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere required for Cohere reranking. "
                "Install with: pip install cohere"
            )
        
        client = cohere.Client(self.config.cohere_api_key)
        
        # Prepare documents
        documents = [chunk.doc_text for chunk in chunks]
        
        # Call Cohere rerank API
        response = client.rerank(
            model=self.config.cohere_model,
            query=query,
            documents=documents,
            top_k=len(chunks)
        )
        
        # Reorder chunks based on Cohere results
        reranked_chunks = []
        for result in response.results:
            chunk = chunks[result.index]
            chunk.relevance_score = result.relevance_score
            reranked_chunks.append(chunk)
        
        return reranked_chunks

class RerankerFactory:
    """Factory for creating rerankers"""
    
    @staticmethod
    def create_reranker(config: RerankerConfig) -> BaseReranker:
        """Create appropriate reranker based on config"""
        if config.reranker_type == "cross-encoder":
            return CrossEncoderReranker(config)
        elif config.reranker_type in ["bge-base", "bge-large"]:
            return BGEReranker(config)
        elif config.reranker_type == "cohere":
            return CohereReranker(config)
        else:
            raise ValueError(f"Unknown reranker type: {config.reranker_type}")
```

### 3.4 Integration into MemoryService

```python
class MemoryService:
    def __init__(self, config: ServerConfig):
        # ... existing init ...
        self.reranker = RerankerFactory.create_reranker(config.reranker_config) if config.reranker_config.reranker_type != "none" else None
    
    async def _rerank_chunks(
        self, 
        query: str, 
        chunks: List[MemoryChunk]
    ) -> List[MemoryChunk]:
        """Apply reranking to chunks"""
        if not self.reranker or len(chunks) <= 1:
            return chunks
        
        try:
            reranked = await self.reranker.rerank(query, chunks)
            logger.debug(f"Reranked {len(chunks)} chunks")
            return reranked
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, falling back to original order")
            return chunks
```

---

## 4. API Updates

### 4.1 Enhanced Search Parameters

Update `tool_memory_search` in `server/unified.py`:

```python
def tool_memory_search(
    query: str,
    max_results: int = 10,
    project_id: str | None = None,
    codebase: str | None = None,
    min_relevance: float = 0.1,
    # NEW PARAMETERS
    domain_boosts: dict[str, float] | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    use_reranker: bool = True
) -> dict[str, Any]:
    """Search for relevant memories with enhanced filtering and boosting."""
    ensure_initialized()
    if not memory_service:
        return {"error": "Memory service not initialized", "results": []}
    
    return memory_service.search(
        query=query,
        codebase=codebase,
        max_results=max_results,
        project_id=project_id,
        domain_boosts=domain_boosts,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        use_reranker=use_reranker
    )

# Update TOOLS registry
TOOLS["memory_search"]["inputSchema"]["properties"].update({
    "domain_boosts": {
        "type": ["object", "null"],
        "description": "Per-query domain boost overrides (e.g., {'class': 1.5, 'test': 0.5})"
    },
    "include_tags": {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "description": "Include only results with these tags (supports prefix:* patterns)"
    },
    "exclude_tags": {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "description": "Exclude results with these tags (supports prefix:* patterns)"
    },
    "use_reranker": {
        "type": "boolean",
        "default": True,
        "description": "Whether to apply reranking for improved relevance"
    }
})
```

---

## 5. Configuration Examples

### 5.1 Default Configuration

```json
{
  "host": "127.0.0.1",
  "port": 8000,
  "persist_directory": "./data/chroma",
  "boost_config": {
    "domain_boosts": {
      "class": 1.2,
      "function": 1.1,
      "imports": 0.9,
      "test": 0.7,
      "private": 0.8,
      "accessor": 1.0
    },
    "memory_type_boosts": {
      "code": 1.1,
      "decision": 1.3,
      "lesson": 1.2,
      "conversation": 1.0
    },
    "recency_enabled": true,
    "recency_decay_days": 30.0,
    "recency_max_boost": 1.5,
    "recency_min_boost": 0.8
  },
  "reranker_config": {
    "reranker_type": "cross-encoder",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "retrieve_k": 20,
    "rerank_top_k": 10,
    "reranker_device": "auto"
  },
  "codebases": [...]
}
```

### 5.2 High-Quality Configuration (with Cohere)

```json
{
  "reranker_config": {
    "reranker_type": "cohere",
    "cohere_api_key": "your-api-key",
    "cohere_model": "rerank-english-v3.0",
    "retrieve_k": 30,
    "rerank_top_k": 10
  }
}
```

---

## 6. Usage Examples

### 6.1 Basic Enhanced Search

```python
# Search with domain boosting
result = memory_service.search(
    query="authentication implementation",
    domain_boosts={"class": 1.5, "test": 0.3}  # Boost classes, penalize tests
)

# Search with tag filtering
result = memory_service.search(
    query="database connection",
    include_tags=["domain:function", "ext:.py"],  # Only Python functions
    exclude_tags=["domain:test"]  # Exclude test code
)

# Search without reranking (faster)
result = memory_service.search(
    query="quick lookup",
    use_reranker=False
)
```

### 6.2 Agent-Friendly Patterns

```python
# For code review: boost recent changes
result = memory_service.search(
    query="error handling patterns",
    domain_boosts={"function": 1.3, "class": 1.2},
    exclude_tags=["domain:test", "domain:imports"]
)

# For documentation: include lessons and decisions
result = memory_service.search(
    query="API design decisions",
    include_tags=["memory_type:decision", "memory_type:lesson"]
)

# For debugging: focus on recent code changes
result = memory_service.search(
    query="authentication bug",
    domain_boosts={"function": 1.4, "private": 1.2},  # Private methods often have bugs
    include_tags=["domain:*"],  # Only code, not conversations
    exclude_tags=["domain:test"]
)
```

---

## 7. Implementation Timeline

### Phase 1: Core Boosting (Week 1) ✅ COMPLETE
- [x] Add `BoostConfig` to server config
- [x] Implement `BoostCalculator` class
- [x] Integrate boosting into `MemoryService.search_async()`
- [x] Add basic tests (test_phase1.py)
- [x] Update API with `domain_boosts` parameter
- [x] Bug fix: BM25 index rebuild performance (was rebuilding every search)
- [x] Bug fix: ChromaDB batch size limit (update_file_info_batch)

### Phase 2: Tag Filtering (Week 1) ✅ COMPLETE
- [x] Implement `_filter_by_tags()` method in `memory_service.py`
- [x] Implement `_tag_matches()` helper for exact and prefix matching
- [x] Add `include_tags`/`exclude_tags` parameters to `search_async()` and `search()`
- [x] Update API in `unified.py` (HTTP + MCP schemas)
- [x] Support prefix matching (`domain:*`, `ext:*`, etc.)
- [x] Add filtering tests (`test_phase2_filtering.py` - all passing)

### Phase 3: Multi-Language AST Chunking (Week 2) ✅ IMPLEMENTATION COMPLETE
- [x] Add tree-sitter dependencies to `requirements.txt` (10 language modules)
- [x] Create `parsers/` subpackage with base interfaces
- [x] Implement `language_configs.py` with comprehensive 9-language support
- [x] Implement `domain_detector.py` with intelligent classification
- [x] Implement `TreeSitterParser` core class with class summary generation
- [x] Update `ChunkMetadata` to include `parent_class` field
- [x] Integrate TreeSitterParser into `ChunkingManager` with fallback
- [x] Update memory service to handle `parent:` tags
- [x] Create basic unit tests (`test_tree_sitter_basic.py`)

**Next Steps**: Install dependencies and test with real codebases

### Phase 3: Reranking (Week 2)
- [ ] Implement `BaseReranker` and `CrossEncoderReranker`
- [ ] Add `RerankerConfig` to server config
- [ ] Integrate reranking into search pipeline
- [ ] Add `use_reranker` parameter
- [ ] Performance testing

### Phase 4: Advanced Rerankers (Week 3)
- [ ] Implement `BGEReranker`
- [ ] Implement `CohereReranker` (optional)
- [ ] Add reranker selection logic
- [ ] Comprehensive testing

### Phase 5: Documentation & Optimization (Week 4)
- [ ] Update API documentation
- [ ] Add configuration examples
- [ ] Performance optimization
- [ ] Integration testing with real codebases

---

## 8. Testing Strategy

### 8.1 Unit Tests
- Boost calculation accuracy
- Tag filtering logic
- Reranker implementations
- Configuration parsing

### 8.2 Integration Tests
- End-to-end search with all features
- Performance benchmarks
- Memory usage monitoring
- Error handling

### 8.3 Quality Metrics
- Search relevance improvement (before/after)
- Latency impact measurement
- Memory usage impact
- Agent usability testing

---

## 9. Dependencies to Add

Update `requirements.txt`:

```
# Existing dependencies...

# For reranking
sentence-transformers>=2.2.0  # Already present
transformers>=4.21.0  # For BGE reranker
torch>=1.12.0  # For BGE reranker
cohere>=4.0.0  # Optional, for Cohere reranker
```

---

## 10. Backward Compatibility

All existing API calls will continue to work unchanged:

```python
# This still works exactly as before
result = memory_service.search(query="test query")

# New features are opt-in
result = memory_service.search(
    query="test query",
    domain_boosts={"class": 1.5},  # NEW
    include_tags=["ext:.py"],      # NEW
    use_reranker=True              # NEW (default)
)
```

---

## 11. Performance Considerations

### Expected Latency Impact:
- **Boosting**: +1-5ms (negligible)
- **Tag Filtering**: +1-10ms (depends on result count)
- **Cross-Encoder Reranking**: +50-200ms (local)
- **BGE Reranking**: +100-300ms (local)
- **Cohere Reranking**: +200-500ms (API call)

### Memory Impact:
- **Boosting**: Minimal
- **Tag Filtering**: Minimal
- **Reranking**: +100-500MB for model loading (one-time)

### Mitigation Strategies:
- Lazy model loading
- Configurable batch sizes
- Optional reranking (can be disabled)
- Caching for repeated queries

---

This implementation plan provides a comprehensive enhancement to conductor-memory while maintaining backward compatibility and agent-friendly usage patterns. The modular design allows users to enable only the features they need, balancing quality improvements with performance requirements.