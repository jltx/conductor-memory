# Conductor Memory

A standalone semantic memory service with codebase indexing for AI agents. Provides intelligent context retrieval through hybrid semantic + keyword search.

## Features

- **Semantic Search**: Vector similarity search using sentence-transformers
- **Hybrid Search**: Combines semantic and BM25 keyword search with Reciprocal Rank Fusion
- **Codebase Indexing**: AST-aware chunking for Python, with support for 15+ languages
- **Multi-Codebase Support**: Index and search across multiple projects
- **Incremental Indexing**: Only re-indexes changed files (tracks mtime + content hash)
- **MCP Integration**: Model Context Protocol servers for AI agent tools
- **HTTP REST API**: FastAPI-based server with automatic OpenAPI docs

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
  "http_port": 9800,
  "tcp_port": 9801,
  "persist_directory": "~/.conductor-memory/data",
  "codebases": [
    {
      "name": "my-project",
      "path": "/path/to/your/project",
      "extensions": [".py", ".js", ".ts", ".md"],
      "ignore_patterns": ["__pycache__", ".git", "node_modules", "venv"]
    }
  ],
  "embedding_model": "all-MiniLM-L12-v2"
}
```

### 2. Start the Server

```bash
# Start the SSE server (recommended for multi-client setups)
python -m conductor_memory.server.sse --config ~/.conductor-memory/config.json

# Or use stdio mode (spawned by MCP clients like OpenCode/Claude)
python -m conductor_memory.server.stdio --config ~/.conductor-memory/config.json

# Or start the unified HTTP + TCP server
python scripts/start_server.py --config ~/.conductor-memory/config.json
```

### 3. Use the API

```python
import requests

# Search for relevant code
response = requests.post("http://localhost:9800/search", json={
    "query": "how does authentication work",
    "max_results": 5
})
results = response.json()

for r in results["results"]:
    print(f"Score: {r['relevance_score']:.3f}")
    print(f"Content: {r['content'][:200]}...")
    print()
```

### 4. Programmatic Usage

```python
from conductor_memory import MemoryService, ServerConfig

# Load config and initialize
config = ServerConfig.from_file("~/.conductor-memory/config.json")
service = MemoryService(config)
service.initialize()  # Blocks until indexing completes

# Search
results = service.search("authentication middleware", max_results=5)
for r in results["results"]:
    print(r["content"])

# Store a memory
service.store(
    content="Important architectural decision about auth...",
    tags=["architecture", "auth"],
    memory_type="decision"
)
```

## Configuration

### Default Data Directory

By default, conductor-memory stores data in `~/.conductor-memory/`:
- `data/` - ChromaDB vector database
- `config.json` - Server configuration
- `memory_server.log` - Server logs

### Environment Variables

```bash
# Override config file location
CONDUCTOR_MEMORY_CONFIG=/path/to/config.json

# ChromaDB settings
CHROMA_PERSIST_DIR=~/.conductor-memory/data
CHROMA_COLLECTION=memory_chunks

# Embedding model
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

## MCP Integration

Conductor Memory supports two transport modes:
- **stdio**: Spawned as a subprocess (simpler, one process per client)
- **SSE (Server-Sent Events)**: Runs as a persistent HTTP server (shared across clients)

### Starting the SSE Server

For SSE mode, start the server first:

```bash
# Using the Python module directly
python -m conductor_memory.server.sse --config ~/.conductor-memory/config.json

# Or with custom host/port
python -m conductor_memory.server.sse --host 127.0.0.1 --port 9820

# Windows batch script
scripts\start_mcp_sse.bat
```

The SSE server runs on `http://127.0.0.1:9820/sse` by default.

### OpenCode Configuration

Add to your `opencode.json`:

**Option 1: stdio mode** (spawns a new process):
```json
{
  "mcp": {
    "memory": {
      "command": ["python", "-m", "conductor_memory.server.stdio", "--config", "~/.conductor-memory/config.json"],
      "enabled": true
    }
  }
}
```

**Option 2: SSE mode** (connects to running server):
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

### Claude Code Configuration

Add to your `claude_desktop_config.json` (typically at `%APPDATA%\Claude\claude_desktop_config.json` on Windows or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

**Option 1: stdio mode**:
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "conductor_memory.server.stdio", "--config", "~/.conductor-memory/config.json"]
    }
  }
}
```

**Option 2: SSE mode** (requires starting the server first):
```json
{
  "mcpServers": {
    "memory": {
      "url": "http://localhost:9820/sse"
    }
  }
}
```

### Other MCP Clients

Any MCP-compatible client can connect using either transport:

**stdio**: Spawn `python -m conductor_memory.server.stdio --config <path>`

**SSE**: Connect to `http://<host>:<port>/sse` (default: `http://localhost:9820/sse`)

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `memory_search` | Search codebase with hybrid semantic+keyword matching |
| `memory_store` | Store important context for later retrieval |
| `memory_store_decision` | Store architectural decisions (auto-pinned) |
| `memory_store_lesson` | Store debugging insights and lessons learned |
| `memory_status` | Check indexing status and memory system health |
| `memory_prune` | Remove obsolete memories based on age/relevance |
| `memory_delete` | Delete a specific memory by ID |

## API Reference

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Server status with indexing progress |
| `/codebases` | GET | List all codebases |
| `/codebases/{name}/status` | GET | Codebase-specific status |
| `/codebases/{name}/reindex` | POST | Trigger re-indexing |
| `/search` | POST | Semantic/hybrid search |
| `/store` | POST | Store memory |
| `/prune` | POST | Prune memories |

### Search Request

```json
{
  "query": "how does X work",
  "codebase": "my-project",  // optional, searches all if omitted
  "max_results": 10,
  "search_mode": "auto"  // auto, semantic, keyword, hybrid
}
```

### Store Request

```json
{
  "content": "Important information...",
  "codebase": "my-project",
  "tags": ["architecture", "decision"],
  "memory_type": "decision",  // code, conversation, decision, lesson
  "pin": true  // Prevents pruning
}
```

## Architecture

```
conductor-memory/
├── src/conductor_memory/
│   ├── core/           # Data models and interfaces
│   ├── storage/        # ChromaDB vector store
│   ├── embedding/      # Sentence transformer embedder
│   ├── search/         # Hybrid search, BM25, chunking
│   ├── config/         # Configuration classes
│   ├── service/        # MemoryService orchestrator
│   ├── server/         # HTTP, stdio, SSE servers
│   └── client/         # HTTP client tools
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
