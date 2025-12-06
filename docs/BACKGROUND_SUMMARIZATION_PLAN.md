# Background Summarization Implementation Plan

## Overview

Add a background summarization system that:
1. Runs heuristic extraction immediately during indexing (fast first pass)
2. Calculates file centrality from import graph
3. Queues files for LLM summarization in priority order
4. Runs LLM summarization in background without blocking queries
5. Stores structured summaries in ChromaDB alongside code chunks
6. Enhances search results with summaries

## Configuration

**Model:** `qwen2.5-coder:7b-instruct-q4_K_M`

**New config section in `config.json`:**
```json
{
  "summarization": {
    "enabled": true,
    "llm_enabled": true,
    "ollama_url": "http://localhost:11434",
    "model": "qwen2.5-coder:7b-instruct-q4_K_M",
    "rate_limit_seconds": 0.5,
    "max_file_lines": 600,
    "max_file_tokens": 4000,
    "skip_patterns": ["**/test/**", "**/*_test.*", "**/vendor/**"],
    "priority_patterns": ["**/src/**", "**/lib/**"]
  }
}
```

## Supported Languages

All 9 languages currently supported by tree-sitter:
- Python, Java, Ruby, Go, C, C#, Kotlin, Swift, Objective-C

## Phase 1: Heuristic Extraction (Instant Value)

**Goal:** Extract structured metadata from all files during indexing without LLM calls.

**What to Extract:**
| Element | Source | Storage |
|---------|--------|---------|
| Class/Interface names + signatures | Tree-sitter AST | ChromaDB metadata |
| Function/method signatures | Tree-sitter AST | ChromaDB metadata |
| Docstrings/KDoc/Javadoc | Tree-sitter AST | Separate chunk with `domain:docstring` |
| Annotations | Tree-sitter AST | ChromaDB metadata field |
| Import statements | Tree-sitter AST | Build import graph |

**Files to Modify:**
- `search/parsers/base.py` - Add `extract_heuristics()` method to parser interface
- `search/parsers/tree_sitter.py` - Implement heuristic extraction per language
- `search/parsers/language_configs.py` - Add annotation extraction queries
- `service/memory_service.py` - Call heuristic extraction during indexing

**New Files:**
- `search/heuristics.py` - HeuristicExtractor class
- `search/import_graph.py` - ImportGraph class for centrality calculation

## Phase 2: Import Graph & Centrality

**Goal:** Build a dependency graph to prioritize "hub" files for summarization.

**Algorithm:**
1. During heuristic pass, record all imports per file
2. Build directed graph: file → imported files
3. Calculate PageRank or in-degree centrality
4. Store centrality scores in SQLite or ChromaDB metadata

**Data Structure:**
```python
class ImportGraph:
    def add_file(self, file_path: str, imports: List[str])
    def calculate_centrality(self) -> Dict[str, float]
    def get_priority_queue(self) -> List[str]  # Sorted by centrality
```

**Files to Create:**
- `search/import_graph.py`

## Phase 3: Ollama Integration

**Goal:** Integrate with Ollama for local LLM summarization.

**Model:** `qwen2.5-coder:7b-instruct-q4_K_M`

**Large File Handling:** For files > threshold (600 lines or 4000 tokens), extract skeleton only:
- Class/interface signatures
- Method signatures
- Docstrings/comments
- Annotations
- No implementation details

**Prompt Template:**
```
Analyze this {language} file and provide a structured summary.

File: {file_path}
```{language}
{file_content_or_skeleton}
```

Respond with JSON only:
{
  "purpose": "1-2 sentence description of what this file does",
  "pattern": "architectural pattern (e.g., Repository, ViewModel, Controller, Utility)",
  "key_exports": ["list", "of", "main", "public", "APIs"],
  "dependencies": ["key", "external", "dependencies"],
  "domain": "business domain (e.g., authentication, payments, ui)"
}
```

**Files to Create:**
- `llm/ollama_client.py` - OllamaClient class with async support
- `llm/summarizer.py` - FileSummarizer using OllamaClient

## Phase 4: Background Task System

**Goal:** Run summarization without blocking queries.

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│                   MemoryService                      │
├─────────────────────────────────────────────────────┤
│  _query_active: asyncio.Event                        │
│  _summarizer_task: asyncio.Task                      │
│  _summary_queue: PriorityQueue[Tuple[float, str]]   │
└─────────────────────────────────────────────────────┘
```

**Cooperative Yielding Logic:**
```python
async def _background_summarizer(self):
    while True:
        # Yield to queries
        if self._query_active.is_set():
            await asyncio.sleep(0.1)
            continue
        
        # Get next file from priority queue
        file_path = await self._summary_queue.get()
        
        # Generate summary via Ollama
        summary = await self._summarizer.summarize(file_path)
        
        # Store in ChromaDB
        await self._store_summary(file_path, summary)
        
        # Rate limit
        await asyncio.sleep(0.5)
```

**Query Path Modification:**
```python
async def search_async(self, ...):
    self._query_active.set()
    try:
        # existing search logic
    finally:
        self._query_active.clear()
```

**Files to Modify:**
- `service/memory_service.py` - Add background task management
- `server/sse.py` - Expose summarization status endpoint

**Files to Create:**
- `service/background_tasks.py` - BackgroundSummarizer class

## Phase 5: Summary Storage & Retrieval

**Goal:** Store summaries in ChromaDB and include in search results.

**Storage Format:**
New chunks with special tags:
```python
{
    "id": "summary_{file_hash}",
    "doc_text": json.dumps(summary_json),
    "tags": ["summary", "file:{file_path}", "codebase:{name}"],
    "source": "llm_summarization",
    "memory_type": "summary"
}
```

**Search Enhancement:**
Add `include_summaries` parameter (default: True):
- When True: Include summary chunks alongside code chunks
- When "only": Return only summaries, not code
- When False: Current behavior

**Files to Modify:**
- `service/memory_service.py` - `search_async()` and `store_async()`
- `server/sse.py` - Add parameter to `memory_search` tool
- `storage/chroma.py` - Add summary-specific queries

## Phase 6: Status & Control APIs

**Goal:** Expose summarization progress and control.

**New MCP Tools:**
```python
@mcp.tool()
async def memory_summarization_status() -> dict:
    """Get background summarization progress"""
    return {
        "is_running": bool,
        "files_queued": int,
        "files_completed": int,
        "current_file": str | None,
        "estimated_time_remaining": str
    }

@mcp.tool()
async def memory_summarization_control(action: str) -> dict:
    """Control summarization: pause, resume, prioritize"""
    # action: "pause" | "resume" | "prioritize:{file_path}"
```

**Files to Modify:**
- `server/sse.py` - Add new tools
- `server/stdio.py` - Add new tools

## File Summary

**New Files (8):**
| File | Purpose |
|------|---------|
| `search/heuristics.py` | HeuristicExtractor class |
| `search/import_graph.py` | ImportGraph with centrality calculation |
| `llm/__init__.py` | Package init |
| `llm/ollama_client.py` | Async Ollama API client |
| `llm/summarizer.py` | FileSummarizer orchestrating LLM calls |
| `service/background_tasks.py` | BackgroundSummarizer task manager |
| `config/summarization.py` | Summarization settings (model, prompts, rate limits) |
| `config/summarization_config.py` | SummarizationConfig dataclass |

**Modified Files (7):**
| File | Changes |
|------|---------|
| `search/parsers/base.py` | Add heuristic extraction interface |
| `search/parsers/tree_sitter.py` | Implement heuristic extraction |
| `search/parsers/language_configs.py` | Add annotation extraction queries |
| `service/memory_service.py` | Background task management, query yielding |
| `storage/chroma.py` | Summary storage/retrieval methods |
| `server/sse.py` | New tools, search parameter |
| `server/stdio.py` | New tools, search parameter |

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Heuristic Extraction | Medium | None |
| Phase 2: Import Graph | Small | Phase 1 |
| Phase 3: Ollama Integration | Small | None |
| Phase 4: Background Tasks | Medium | Phase 2, 3 |
| Phase 5: Storage & Retrieval | Medium | Phase 4 |
| Phase 6: Status APIs | Small | Phase 4 |

**Total:** ~2-3 days of implementation

## Implementation Order

1. Phase 1 & 2 (Heuristics + Import Graph) - Immediate value
2. Phase 3 (Ollama Integration) - Can be tested standalone
3. Phase 4 (Background Tasks) - Core orchestration
4. Phase 5 (Storage & Retrieval) - User-facing enhancement
5. Phase 6 (Status APIs) - Polish and control

## Success Criteria

- Heuristic extraction provides instant structured metadata for all supported languages
- Import graph correctly identifies central/hub files
- Background summarization runs without blocking queries
- Search results include relevant summaries alongside code chunks
- System handles 12,000+ files efficiently with configurable resource usage