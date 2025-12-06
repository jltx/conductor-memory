# MCP Memory Server

A standalone memory service for agent-based tools with codebase indexing and semantic search capabilities.

## Features

### 🧠 **Intelligent Memory Management**
- **Semantic Search**: Find relevant context using vector similarity
- **Conversation History**: Store and index chat conversations
- **Context Pruning**: Automatically remove obsolete memories
- **Persistent Storage**: Uses Chroma vector database with sentence transformers

### 📁 **Codebase Indexing**
- **Automatic Indexing**: Indexes entire codebases on startup
- **Smart Chunking**: Function/class-based chunking for code files
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, etc.
- **Incremental Updates**: Efficient batch processing with progress tracking

### 🔧 **MCP Tool Integration**
- **Standardized Tools**: Ready-to-use MCP tools for agents
- **RESTful API**: FastAPI-based server with automatic documentation
- **Flexible Configuration**: Environment-based configuration
- **Production Ready**: CORS support, error handling, logging

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Basic usage (current directory as codebase)
python start_memory_server.py

# Specify codebase path
python start_memory_server.py --codebase /path/to/your/code

# Custom port and host
python start_memory_server.py --port 8080 --host 0.0.0.0

# Development mode with auto-reload
python start_memory_server.py --dev
```

### 3. Access the API

- **Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Core Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Search memories using semantic similarity |
| `/store` | POST | Store new memory chunks |
| `/prune` | POST | Prune obsolete memories |
| `/status` | GET | Get server status and indexing progress |
| `/health` | GET | Health check |

### Memory Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memories/{id}` | GET | Get specific memory by ID |
| `/memories/{id}` | DELETE | Delete specific memory |

## MCP Tools

The server provides standardized MCP tools for agent integration:

### `memory_search`
Search for relevant context using semantic similarity.

```python
from src.mcp_memory_tools import MemorySearchTool

tool = MemorySearchTool()
result = tool.execute({
    "query": "binary search algorithm",
    "max_results": 5,
    "project_id": "my_project"
})
```

### `memory_store`
Store new memories (conversations, code, documentation).

```python
from src.mcp_memory_tools import MemoryStoreTool

tool = MemoryStoreTool()
result = tool.execute({
    "project_id": "my_project",
    "role": "user",
    "prompt": "How do I implement quicksort?",
    "tags": ["algorithm", "sorting"]
})
```

### `memory_prune`
Prune obsolete memories based on age and relevance.

```python
from src.mcp_memory_tools import MemoryPruneTool

tool = MemoryPruneTool()
result = tool.execute({
    "project_id": "my_project",
    "max_age_days": 30
})
```

### `memory_status`
Check server status and indexing progress.

```python
from src.mcp_memory_tools import MemoryStatusTool

tool = MemoryStatusTool()
result = tool.execute({})
```

## Configuration

### Environment Variables

```bash
# Memory server URL (for tools)
MCP_MEMORY_SERVER_URL=http://localhost:8000

# Chroma vector database
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=memory_chunks

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

### Server Configuration

```bash
# Start with custom configuration
python src/mcp_memory_server.py \
    --port 8000 \
    --host 127.0.0.1 \
    --codebase-path /path/to/code \
    --log-level INFO
```

## Usage Examples

### 1. Agent Integration

```python
from src.mcp_memory_tools import create_memory_tools

# Create all memory tools
tools = create_memory_tools()

# Use in your agent
search_tool = tools["memory_search"]
result = search_tool.execute({
    "query": "authentication middleware",
    "project_id": "web_app"
})

print(result.output)  # Human-readable results
print(result.structured_data)  # Raw API response
```

### 2. Direct API Usage

```python
import requests

# Search for relevant code
response = requests.post("http://localhost:8000/search", json={
    "query": "database connection pooling",
    "max_results": 3
})

results = response.json()
for memory in results["results"]:
    print(f"Found: {memory['source']} - {memory['tags']}")
    print(f"Content: {memory['doc_text'][:200]}...")
```

### 3. Store Conversation

```python
import requests

# Store user message
requests.post("http://localhost:8000/store", json={
    "project_id": "chat_session_1",
    "role": "user",
    "prompt": "How do I optimize database queries?",
    "tags": ["database", "performance"]
})

# Store assistant response
requests.post("http://localhost:8000/store", json={
    "project_id": "chat_session_1",
    "role": "assistant",
    "response": "Here are several strategies for optimizing database queries...",
    "tags": ["database", "performance", "optimization"]
})
```

## Architecture

### Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent/Tool    │───▶│  MCP Memory     │───▶│  Memory Core    │
│                 │    │  Server (API)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  FastAPI        │    │  Chroma Vector  │
                       │  REST Server    │    │  Database       │
                       └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Initialization**: Server starts and indexes codebase
2. **Storage**: Memories stored with embeddings in vector database
3. **Search**: Queries converted to embeddings for similarity search
4. **Retrieval**: Relevant memories returned with context summaries
5. **Pruning**: Obsolete memories removed based on age/relevance

## Performance

### Benchmarks

- **Indexing Speed**: ~50-100 files/second (depends on file size)
- **Search Latency**: ~10-50ms for typical queries
- **Memory Usage**: ~100MB base + ~1MB per 1000 code chunks
- **Storage**: ~1KB per code chunk in vector database

### Optimization Tips

1. **Batch Size**: Adjust batch size for indexing based on available memory
2. **Embedding Model**: Use smaller models for faster inference
3. **Pruning**: Regular pruning keeps database size manageable
4. **Caching**: Enable response caching for repeated queries

## Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check if port is available
netstat -an | findstr :8000

# Try different port
python start_memory_server.py --port 8001
```

**Indexing fails:**
```bash
# Check file permissions
# Ensure codebase path exists and is readable
# Check logs for specific error messages
```

**Search returns no results:**
```bash
# Check if indexing completed
curl http://localhost:8000/status

# Verify memories were stored
curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d '{"query":"test","min_relevance":0.0}'
```

### Logging

Enable debug logging for troubleshooting:

```bash
python start_memory_server.py --log-level DEBUG
```

## Integration with Conductor

The MCP Memory Server integrates seamlessly with the Conductor orchestrator:

```python
# In Conductor's tool registry
from src.mcp_memory_tools import create_memory_tools

# Add memory tools to orchestrator
memory_tools = create_memory_tools()
for name, tool in memory_tools.items():
    orchestrator.tool_registry.register(tool)
```

## Development

### Running Tests

```bash
# Test memory components
python test_mcp_memory_server.py

# Test with specific codebase
python test_mcp_memory_server.py --codebase /path/to/code
```

### Adding New Features

1. **New API Endpoints**: Add to `mcp_memory_server.py`
2. **New MCP Tools**: Add to `mcp_memory_tools.py`
3. **New Memory Types**: Extend `MemoryChunk` in `memory_db.py`
4. **New Chunking Strategies**: Add to `chunking_strategy.py`

## License

This MCP Memory Server is part of the Conductor project and follows the same licensing terms.

---

**Ready to enhance your agent's memory capabilities!** 🚀

The MCP Memory Server provides a robust foundation for building intelligent agents that can remember, learn, and provide contextually relevant responses based on both codebase knowledge and conversation history.