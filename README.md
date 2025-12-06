# Conductor Memory

A standalone semantic memory service with codebase indexing for AI agents. Provides intelligent context retrieval through hybrid semantic + keyword search.

## Features

- **Semantic Search**: Vector similarity search using sentence-transformers
- **Hybrid Search**: Combines semantic and BM25 keyword search with Reciprocal Rank Fusion
- **Codebase Indexing**: AST-aware chunking for Python, with support for 15+ languages
- **Multi-Codebase Support**: Index and search across multiple projects
- **Incremental Indexing**: Only re-indexes changed files (tracks mtime + content hash)
- **Background Summarization**: LLM-powered file summaries using Ollama
- **MCP Integration**: Model Context Protocol server for AI agent tools
- **Web Dashboard**: Real-time monitoring of indexing and summarization progress

## Installation

```bash
pip install conductor-memory
```

Or from source:
```bash
git clone https://github.com/yourusername/conductor-memory
cd conductor-memory
pip install -e .
```

## Quick Start

### 1. Create a Configuration File

Create `~/.conductor-memory/config.json`:

```json
{
  "host": "127.0.0.1",
  "persist_directory": "~/.conductor-memory",
  "codebases": [
    {
      "name": "my-project",
      "path": "/path/to/your/project",
      "extensions": [".py", ".js", ".ts", ".md"],
      "ignore_patterns": ["__pycache__", ".git", "node_modules", "venv"]
    }
  ],
  "embedding_model": "all-MiniLM-L12-v2",
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
# Start the SSE server
python -m conductor_memory.server.sse --port 9820

# Or use the start script (Windows)
.\scripts\start.ps1
```

The server will be available at:
- **Dashboard**: http://localhost:9820/
- **MCP SSE endpoint**: http://localhost:9820/sse
- **Health check**: http://localhost:9820/health

### 3. Configure Your AI Client

Add to your `opencode.json`:
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

Or for Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "memory": {
      "url": "http://localhost:9820/sse"
    }
  }
}
```

### 4. Use the API

```python
import requests

# Search for relevant code
response = requests.post("http://localhost:9820/search", json={
    "query": "how does authentication work",
    "max_results": 5
})
results = response.json()

for r in results["results"]:
    print(f"Score: {r['relevance_score']:.3f}")
    print(f"Content: {r['content'][:200]}...")
```

## Configuration

### Default Data Directory

By default, conductor-memory stores data in `~/.conductor-memory/`:
- ChromaDB vector database
- `config.json` - Server configuration

### Environment Variables

```bash
# Override config file location
CONDUCTOR_MEMORY_CONFIG=/path/to/config.json

# Embedding settings
EMBEDDING_MODEL=all-MiniLM-L12-v2
EMBEDDING_DEVICE=cuda  # Use GPU if available
```

### Codebase Configuration Options

| Field | Description | Default |
|-------|-------------|---------|
| `name` | Unique identifier for this codebase | Required |
| `path` | Absolute path to codebase root | Required |
| `extensions` | File extensions to index | Common code extensions |
| `ignore_patterns` | Patterns to exclude | `__pycache__`, `.git`, etc. |
| `enabled` | Whether to index this codebase | `true` |
| `description` | Human-readable description | `""` |

### Summarization Configuration

| Field | Description | Default |
|-------|-------------|---------|
| `enabled` | Enable background summarization | `false` |
| `llm_enabled` | Enable LLM calls | `false` |
| `ollama_url` | Ollama server URL | `http://localhost:11434` |
| `model` | LLM model for summarization | `qwen2.5-coder:1.5b` |
| `rate_limit_seconds` | Delay between LLM calls | `0.5` |
| `timeout_seconds` | LLM request timeout | `30.0` |

## Search Modes

The `search_mode` parameter controls how queries are processed:

| Mode | Best For | Example Query |
|------|----------|---------------|
| `auto` (default) | Auto-detects based on query | Any query |
| `semantic` | Conceptual questions | "how does authentication work" |
| `keyword` | Exact identifiers | "calculate_position_size", "SwingDetector" |
| `hybrid` | Mixed queries | "position sizing with Kelly criterion" |

### Auto-Detection Heuristics

- **snake_case** identifiers → keyword mode
- **CamelCase** identifiers → keyword mode
- **Quoted strings** → keyword mode
- **Question words** (how, what, why) → semantic mode
- **Mixed/unclear** → hybrid mode

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `memory_search` | Search codebase with hybrid semantic+keyword matching |
| `memory_store` | Store important context for later retrieval |
| `memory_store_decision` | Store architectural decisions (auto-pinned) |
| `memory_store_lesson` | Store debugging insights and lessons learned |
| `memory_status` | Check indexing status and memory system health |
| `memory_summarization_status` | Check LLM summarization progress |
| `memory_prune` | Remove obsolete memories based on age/relevance |
| `memory_delete` | Delete a specific memory by ID |

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/health` | GET | Health check |
| `/api/status` | GET | Full service status |
| `/api/summarization` | GET | Summarization progress |
| `/sse` | GET | MCP SSE endpoint |

## Architecture

```
conductor-memory/
├── src/conductor_memory/
│   ├── core/           # Data models and interfaces
│   ├── storage/        # ChromaDB vector store
│   ├── embedding/      # Sentence transformer embedder
│   ├── search/         # Hybrid search, BM25, chunking
│   ├── llm/            # Ollama client, summarizer
│   ├── config/         # Configuration classes
│   ├── service/        # MemoryService orchestrator
│   └── server/         # SSE server with web dashboard
├── scripts/            # Startup scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Performance

- **Indexing Speed**: ~50-100 files/second
- **Search Latency**: ~10-50ms for typical queries
- **Memory Usage**: ~100MB base + ~1MB per 1000 code chunks

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_search_quality.py -v
```

## License

MIT License - see LICENSE file for details.
