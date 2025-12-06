# Conductor Memory

A production-ready semantic memory service with intelligent codebase indexing for AI agents. Provides context-aware search through hybrid semantic + keyword matching with advanced heuristic filtering.

## Features

### 🔍 **Advanced Search**
- **Hybrid Search**: Combines semantic vectors + BM25 keyword search with Reciprocal Rank Fusion
- **Heuristic Filtering**: Filter by classes, functions, annotations, file types, and complexity metrics
- **Smart Mode Detection**: Auto-detects optimal search strategy (semantic/keyword/hybrid)
- **Summary Integration**: Include LLM-generated file summaries in search results with relevance boosting

### 📁 **Intelligent Indexing**
- **Multi-Language Support**: AST-aware parsing for 15+ programming languages via tree-sitter
- **Multi-Codebase**: Index and search across multiple projects simultaneously
- **Incremental Updates**: Only re-indexes changed files (mtime + content hash tracking)
- **Metadata Extraction**: Automatic extraction of classes, functions, imports, and complexity metrics

### 🤖 **LLM Integration**
- **Background Summarization**: Automatic file summarization using Ollama with callback-based startup
- **Time Estimation**: Real-time progress tracking with completion time estimates
- **Incremental Re-summarization**: Smart detection of file changes for efficient updates
- **Pattern Recognition**: Identifies architectural patterns and domains in code

### 🔧 **Production Ready**
- **MCP Integration**: Full Model Context Protocol server for AI agent tools
- **Web Dashboard**: Real-time monitoring with enhanced status reporting
- **RESTful API**: Complete HTTP API for integration
- **Robust Configuration**: Flexible config system with validation and hot-reloading

## Installation

### Quick Install (Recommended)

```bash
pip install conductor-memory
```

### Platform-Specific Installation

#### Windows
```cmd
# Using pip
pip install conductor-memory

# Or use the convenience script
curl -O https://raw.githubusercontent.com/jltx/conductor-memory/main/scripts/install-windows.ps1
powershell -ExecutionPolicy Bypass -File install-windows.ps1
```

#### macOS/Linux
```bash
# Using pip
pip install conductor-memory

# Or use the install script (Linux)
curl -sSL https://raw.githubusercontent.com/jltx/conductor-memory/main/scripts/install-linux.sh | bash
```

### Development Installation
```bash
git clone https://github.com/jltx/conductor-memory
cd conductor-memory
pip install -e .
```

📖 **For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

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

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_search` | Advanced search with filtering and summary integration | `include_summaries`, `boost_summarized`, `include_tags`, `min_class_count` |
| `memory_store` | Store important context for later retrieval | `content`, `tags`, `pin` |
| `memory_store_decision` | Store architectural decisions (auto-pinned) | `content`, `tags` |
| `memory_store_lesson` | Store debugging insights and lessons learned | `content`, `tags` |
| `memory_status` | Check indexing status and memory system health | - |
| `memory_summarization_status` | Check LLM summarization progress with timing estimates | - |
| `memory_queue_codebase_summarization` | Queue specific codebase for LLM summarization | `codebase`, `only_missing` |
| `memory_reindex_codebase` | Force reindexing of a specific codebase | `codebase` |
| `memory_prune` | Remove obsolete memories based on age/relevance | `max_age_days` |
| `memory_delete` | Delete a specific memory by ID | `memory_id` |

### Advanced Search Parameters

- **`include_summaries`**: Include LLM-generated file summaries in results
- **`boost_summarized`**: Apply 15% relevance boost to files with summaries  
- **`include_tags`/`exclude_tags`**: Filter by metadata tags (supports wildcards)
- **`min_class_count`/`min_function_count`**: Filter by code complexity
- **`languages`**: Filter by programming language
- **`search_mode`**: Force specific search strategy

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

- **Indexing Speed**: ~50-100 files/second with heuristic metadata extraction
- **Search Latency**: ~10-50ms for typical queries (hybrid search with filtering)
- **Summarization Speed**: ~0.6s per file average (with qwen2.5-coder:1.5b)
- **Memory Usage**: ~100MB base + ~1MB per 1000 code chunks + summary metadata
- **Scalability**: Tested with 400+ files across multiple codebases

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
