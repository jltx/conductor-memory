# Conductor Memory - TODO

## Background Summarization (Phase 4) - COMPLETED

The background summarization system has been implemented and wired up.

### What's Implemented
- [x] `llm/ollama_client.py` - Async Ollama API client
- [x] `llm/lmstudio_client.py` - LMStudio client (alternative)
- [x] `llm/summarizer.py` - FileSummarizer with skeleton extraction for large files
- [x] `llm/base.py` - Base LLM client interface
- [x] `config/summarization.py` - SummarizationConfig dataclass
- [x] `config/server.py` - Integrated summarization_config into ServerConfig
- [x] `search/import_graph.py` - Import graph with centrality calculation
- [x] `search/heuristics.py` - Heuristic extraction (classes, functions, annotations)
- [x] `service/memory_service.py` - Background summarizer with:
  - Priority queue based on file centrality
  - Query yielding (pauses during active searches)
  - Progress logging every 10 files
  - Summary storage as pinned memory chunks
  - Graceful startup with Ollama health check
- [x] `server/stdio.py` - Added `memory_summarization_status` MCP tool

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
- [ ] Fix PowerShell conda activation error in PATH (harmless but noisy)
- [ ] Consider adding log rotation for `logs/conductor-memory.log`

### Future Enhancements (Phase 7+)
- [ ] Support for more LLM backends (OpenAI, Anthropic for cloud option)
- [ ] Summary quality metrics and validation
- [ ] Batch re-summarization command for forcing refresh
- [ ] Summary diff view when files change
