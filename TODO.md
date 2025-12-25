# Conductor Memory - TODO

## ✅ COMPLETED PHASES

### Phase 4: Background Summarization - COMPLETED
- LLM-powered file summarization using Ollama
- Priority queue based on file centrality
- Query yielding to avoid search latency impact
- Graceful startup with health checks

### Phase 5: Summary Integration - COMPLETED  
- `include_summaries` parameter for enhanced search results
- `boost_summarized` parameter for 15% relevance boost
- Structured summary data with purpose, pattern, exports
- Full integration with MCP tools and HTTP API

### Phase 6: Incremental Re-summarization & Web UI - COMPLETED
- Smart detection of file changes for efficient updates
- Real-time web dashboard with progress tracking
- Enhanced status reporting with timing estimates
- Callback-based startup system

### Configuration

Add to `~/.conductor-memory/config.json`:
```json
{
  "summarization": {
    "enabled": true,
    "llm_enabled": true,
    "ollama_url": "http://localhost:11434",
    "model": "qwen2.5-coder:1.5b",
    "rate_limit_seconds": 0.5,
    "timeout_seconds": 30.0,
    "max_file_lines": 600,
    "max_file_tokens": 4000
  }
}
```

### Usage

1. Ensure Ollama is running: `ollama serve`
2. Pull the model: `ollama pull qwen2.5-coder:1.5b`
3. Add summarization config to `~/.conductor-memory/config.json`
4. Restart conductor-memory
5. Check progress with `memory_summarization_status` tool or logs

The summarizer will:
- Start automatically after indexing completes
- Process files in priority order (most central files first)
- Pause during active searches to avoid latency impact
- Store summaries as searchable, pinned memory chunks
- Log progress every 10 files

---

## Summary Integration in Search (Phase 5) - COMPLETED

Phase 5 integrates LLM-generated file summaries into the search experience.

### What's Implemented
- [x] `include_summaries` parameter for `memory_search` MCP tool
  - When `True`, results include `file_summary` dict with:
    - `purpose` - What the file does
    - `pattern` - Architectural pattern (Repository, Controller, etc.)
    - `key_exports` - Main public APIs/functions
    - `dependencies` - External dependencies
    - `language`, `domain`
  - Each result includes `has_summary` boolean
- [x] `boost_summarized` parameter for relevance boosting
  - Files with LLM summaries get 15% relevance boost (default enabled)
  - Rationale: Summarized files are "better understood" by the system
- [x] Helper methods in `MemoryService`:
  - `_extract_file_path_from_chunk()` - Extract file path from chunk tags
  - `_get_file_summaries_async()` - Lookup summaries for result files
  - `_parse_summary_text()` - Parse stored summary format
  - `_apply_summary_boost_async()` - Apply boost to summarized files
  - `_get_summarized_files_async()` - Get set of files with summaries
- [x] Tests in `tests/test_phase5_summary_integration.py`

### Usage

```python
# Search with summaries included
results = memory_search(
    query="authentication flow",
    include_summaries=True,    # Include file summaries in results
    boost_summarized=True      # Boost files that have summaries (default)
)

# Each result will have:
# {
#   "id": "...",
#   "content": "...",
#   "has_summary": True,
#   "file_summary": {
#     "purpose": "Handles user authentication",
#     "pattern": "Controller",
#     "key_exports": ["login", "logout"],
#     "dependencies": ["jwt", "bcrypt"]
#   }
# }
```

---

## Incremental Re-summarization & Web UI (Phase 6) - COMPLETED

Phase 6 adds incremental re-summarization when files change and a web dashboard.

### What's Implemented

#### Incremental Re-summarization
- [x] `SummaryIndexMetadata` class in `storage/chroma.py`
  - Tracks content hash for each summarized file
  - Enables detection of changed files needing re-summarization
  - Methods: `get_summary_info()`, `update_summary_info()`, `needs_resummarization()`
- [x] Updated `_store_summary()` to track content hashes
- [x] Updated `_background_summarizer()` to skip unchanged files
- [x] File watcher integration:
  - Modified files are queued for re-summarization with high priority
  - Deleted files have their summaries removed
- [x] New methods in `MemoryService`:
  - `queue_file_for_summarization()` - Queue single file for (re-)summarization
  - `remove_file_summary()` - Remove summary for deleted file

#### Web Dashboard
- [x] Simple HTML dashboard at `http://localhost:9820/`
  - Real-time summarization progress
  - Indexing statistics per codebase
  - Current file being processed
  - Error display
  - Auto-refresh every 5 seconds
- [x] REST API endpoints:
  - `GET /api/summarization` - Summarization status JSON
  - `GET /api/status` - Full service status JSON
  - `GET /health` - Health check endpoint

#### Enhanced Status Reporting
- [x] `get_summarization_status()` now includes:
  - `total_summarized` - Persistent count of summarized files
  - `files_skipped` - Files skipped (unchanged)
  - `by_codebase` - Breakdown by codebase with pattern/domain stats

### Usage

1. Start the SSE server:
   ```bash
   python -m conductor_memory.server.sse --port 9820
   ```

2. Open the dashboard: http://localhost:9820/

3. API endpoints:
   ```bash
   # Get summarization status
   curl http://localhost:9820/api/summarization
   
   # Get full service status
   curl http://localhost:9820/api/status
   ```

4. File changes are automatically detected and queued for re-summarization

---

## Other TODOs

### Minor Fixes
- [x] PowerShell conda activation error - EXTERNAL: User environment issue, not in codebase.
      Scripts use direct Python paths to avoid conda activation. Users experiencing this can run:
      `conda init powershell --reverse && conda init powershell` to reinitialize cleanly.
- [x] Log rotation for service logs - Uses RotatingFileHandler with 10MB max, 5 backups (~50MB total)

### ✅ Issues Resolved During Testing
- [x] **Heuristic Metadata Storage**: Fixed - Now properly storing `class_count:N`, `function_count:N`, `lang:python` tags
- [x] **Phase 5 Summary Integration**: Fixed - `include_summaries=True` returns structured summary data
- [x] **Phase 5 Summary Boost**: Fixed - 15% boost correctly applied (0.012454 → 0.014322)
- [x] **Callback-based Startup**: Fixed - Summarizer starts immediately when indexing completes
- [x] **Time Estimation**: Added - Real-time progress tracking with completion estimates
- [x] **Tag Filtering**: Working - `include_tags` with proper tag names returns results

### 🔧 Known Minor Issues
- [x] Keyword search mode returns 0 results for exact class names (semantic/hybrid work fine)
  - **Fixed**: Changed from `BM25Okapi` to `BM25Plus` algorithm which handles small corpora correctly
  - Root cause: `BM25Okapi` returns 0 or negative scores for terms in small document sets due to IDF calculation
  - Also fixed: `store()` now adds chunks to BM25 index for keyword/hybrid search
- [x] Search result inconsistency: Same wildcard query may return different results on repeated calls
  - **Fixed**: Added secondary sort keys (chunk ID, doc_id, etc.) as tiebreakers in all score-based sorts
  - Root cause: When multiple chunks had identical relevance scores, Python's stable sort preserved 
    non-deterministic dict iteration order, causing result order to vary between calls
  - Files fixed: `hybrid.py` (3 sorts), `memory_service.py` (3 sorts), `verification.py` (1 sort), `import_graph.py` (1 sort)

## 🚀 NEXT PHASE PRIORITIES

### Phase 7: Conversation Memory Integration
- [ ] **Context Persistence**: Automatically store important conversation context
- [ ] **Decision Tracking**: Link code changes to architectural decisions  
- [ ] **Learning Integration**: Connect debugging sessions to code improvements
- [ ] **Session Management**: Track conversation threads and outcomes

### Phase 8: Enhanced Search Intelligence
- [ ] **Semantic Code Relationships**: Understand function call graphs and dependencies
- [ ] **Pattern Recognition**: Identify similar code patterns across codebases
- [ ] **Change Impact Analysis**: Predict which files might be affected by changes
- [ ] **Smart Suggestions**: Recommend related files/functions based on context

### Phase 9: Advanced Summarization
- [ ] **Cross-File Context**: Summaries that understand file relationships
- [ ] **Custom Prompts**: Domain-specific summarization strategies
- [ ] **Quality Metrics**: Validation and scoring of summary quality
- [ ] **Multi-Model Support**: OpenAI, Anthropic for cloud options

### Dashboard UX Enhancements - COMPLETED
See `docs/plans/DASHBOARD_UX_ENHANCEMENTS.md` for details.
- [x] Fix summarization stats terminology (remove "session" concept)
- [x] Add action buttons (queue, invalidate, reindex)
- [x] Show simple file indicators in Browse/Summaries tabs
- [x] Show Phase 2 fields (how_it_works, mechanisms, method summaries)
- [x] Rename Validate tab to Summaries
- [x] Add summary statistics view
- [x] Show summary preview in search results

### Future Dashboard Enhancements (Deferred)
- [ ] **Call Graph Explorer**: Visualize method-to-method relationships
- [ ] **Dependency Graph**: Visualize file import relationships
- [ ] **File Centrality View**: Show most important files by PageRank score

### Minor Enhancements
- [x] Fix keyword search for exact class names
- [ ] Add summary diff view when files change
- [ ] Implement batch re-summarization command
