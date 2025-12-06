# MCP Memory Server

A standalone memory service for AI agents with codebase indexing, semantic search, and LLM-powered summarization.

## Features

### Intelligent Memory Management
- **Semantic Search**: Vector similarity search using sentence-transformers
- **Hybrid Search**: Combines semantic and BM25 keyword matching with RRF
- **Context Storage**: Store decisions, lessons learned, and conversation context
- **Persistent Storage**: ChromaDB vector database with incremental indexing

### Codebase Indexing
- **Automatic Indexing**: Indexes codebases on startup with smart chunking
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, Go, Rust, etc.
- **File Watching**: Automatically re-indexes changed files
- **Incremental Updates**: Only processes changed files (tracks mtime + content hash)

### LLM Summarization
- **Background Processing**: Summarizes files using local LLM (Ollama)
- **Incremental Re-summarization**: Re-summarizes when files change
- **Search Integration**: Summaries boost search relevance

### Web Dashboard
- **Real-time Monitoring**: View indexing and summarization progress
- **REST API**: JSON endpoints for status and debugging

## Quick Start

### 1. Create Configuration

Create `~/.conductor-memory/config.json`:

```json
{
  "host": "127.0.0.1",
  "persist_directory": "~/.conductor-memory",
  "codebases": [
    {
      "name": "my-project",
      "path": "/path/to/your/project",
      "extensions": [".py", ".js", ".ts"],
      "ignore_patterns": ["__pycache__", ".git", "node_modules"]
    }
  ],
  "enable_file_watcher": true,
  "summarization": {
    "enabled": true,
    "llm_enabled": true,
    "ollama_url": "http://localhost:11434",
    "model": "qwen2.5-coder:1.5b"
  }
}
```

### 2. Start the Server

```bash
python -m conductor_memory.server.sse --port 9820
```

### 3. Configure Your AI Client

**OpenCode** (`opencode.json`):
```json
{
  "mcp": {
    "memory": {
      "type": "remote",
      "url": "http://localhost:9820/sse"
    }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "memory": {
      "url": "http://localhost:9820/sse"
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_search` | Search with hybrid semantic+keyword matching |
| `memory_store` | Store context for later retrieval |
| `memory_store_decision` | Store architectural decisions (auto-pinned) |
| `memory_store_lesson` | Store debugging insights and lessons |
| `memory_status` | Check indexing status |
| `memory_summarization_status` | Check LLM summarization progress |
| `memory_prune` | Remove obsolete memories |
| `memory_delete` | Delete specific memory by ID |

### Search Parameters

```python
memory_search(
    query="authentication flow",
    max_results=10,
    codebase="my-project",      # Optional: filter by codebase
    search_mode="auto",          # auto, semantic, keyword, hybrid
    include_summaries=True,      # Include LLM summaries in results
    boost_summarized=True        # Boost files with summaries
)
```

## REST API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web dashboard |
| `GET /health` | Health check |
| `GET /api/status` | Full service status |
| `GET /api/summarization` | Summarization progress |
| `GET /sse` | MCP SSE endpoint |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   AI Client     │────▶│   SSE Server    │────▶│ MemoryService   │
│ (OpenCode/etc)  │     │   (FastMCP)     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Web Dashboard  │     │  ChromaDB +     │
                        │  + REST API     │     │  Embeddings     │
                        └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Background     │
                                                │  Summarizer     │
                                                │  (Ollama)       │
                                                └─────────────────┘
```

## Performance

- **Indexing**: ~50-100 files/second
- **Search**: ~10-50ms latency
- **Memory**: ~100MB base + ~1MB per 1000 chunks
- **Summarization**: ~1-2 files/second (depends on LLM)

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
netstat -ano | findstr :9820

# Try different port
python -m conductor_memory.server.sse --port 9821
```

### No summarization progress
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull the model if needed
ollama pull qwen2.5-coder:1.5b
```

### Search returns no results
```bash
# Check indexing status
curl http://localhost:9820/api/status

# Check if files are indexed
# Look for "indexed_files_count" in response
```
