# Conductor Memory Architecture

## Overview

Conductor Memory is a semantic memory service designed for AI agent integration. It provides context retrieval through hybrid search combining vector similarity and keyword matching.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Conductor Memory                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────┐  ┌─────────────┐                      │
│  │   SSE Server (port 9820)     │  │ stdio MCP   │                      │
│  │  - HTTP REST API             │  │  Server     │                      │
│  │  - SSE MCP endpoint (/sse)   │  │             │                      │
│  │  - Web Dashboard (/)         │  │             │                      │
│  └──────────────┬───────────────┘  └──────┬──────┘                      │
│                 │                         │                              │
│                 └────────────┬────────────┘                              │
│                              │                                           │
│                     ┌────────┴────────┐                                 │
│                     │  MemoryService  │                                 │
│                     │  (Orchestrator) │                                 │
│                     └────────┬────────┘                                 │
│                              │                                           │
│         ┌────────────────────┼────────────────────┐                     │
│         │                    │                    │                     │
│  ┌──────┴──────┐      ┌──────┴──────┐     ┌──────┴──────┐              │
│  │   Vector    │      │   Hybrid    │     │  Chunking   │              │
│  │   Store     │      │  Searcher   │     │   Manager   │              │
│  │  (Chroma)   │      │ (BM25+Vec)  │     │    (AST)    │              │
│  └──────┬──────┘      └──────┬──────┘     └─────────────┘              │
│         │                    │                                          │
│  ┌──────┴──────┐      ┌──────┴──────┐                                  │
│  │  Embedder   │      │  BM25 Index │                                  │
│  │ (SentTrans) │      │ (rank_bm25) │                                  │
│  └─────────────┘      └─────────────┘                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼ (optional)
     ┌──────────────────────────┐    ┌──────────────────────────┐
     │   ChromaDB (Persistent)  │    │   PostgreSQL (Metadata)  │
     │   ~/.conductor-memory/   │    │   Fast dashboard queries │
     │   - Vector embeddings    │    │   - File index metadata  │
     │   - Full-text content    │    │   - Summary metadata     │
     └──────────────────────────┘    │   - Materialized stats   │
                                     └──────────────────────────┘
```

## Component Details

### Core Layer (`core/`)

**models.py** - Data structures
- `MemoryChunk`: Primary data unit with content, metadata, embeddings
- `RoleEnum`: Message roles (user, assistant, system, tool)
- `MemoryType`: Memory categories (code, conversation, decision, lesson)
- `MemoryDB`: Abstract interface for memory storage

**vector_store.py** - Abstract vector storage interface
- `add()`, `search()`, `delete()` operations
- Implementations must provide similarity search

**embedder.py** - Abstract embedding interface
- `generate()`: Single text embedding
- `generate_batch()`: Batch embedding for efficiency

### Storage Layer (`storage/`)

**chroma.py** - ChromaDB implementation
- Persistent vector storage with cosine similarity
- File index metadata for incremental indexing
- `needs_reindex()`: Check if file changed (mtime + content hash)
- `update_file_index()`: Track indexed files
- Modes: `embedded` (SQLite) or `http` (standalone server)

**postgres.py** - PostgreSQL metadata store (optional)
- Fast metadata queries for dashboard operations
- Sub-millisecond counts via materialized views
- Paginated file/summary listing with filtering
- Automatic fallback to ChromaDB if unavailable
- Requires: `pip install conductor-memory[postgres]`

### Embedding Layer (`embedding/`)

**sentence_transformer.py** - SentenceTransformer implementation
- Default model: `all-MiniLM-L12-v2` (384 dimensions)
- GPU support via `device` parameter
- Batch processing for efficiency

### Search Layer (`search/`)

**hybrid.py** - Hybrid search engine
- `BM25Index`: In-memory keyword index per codebase
- `HybridSearcher`: Combines semantic + keyword with RRF
- `detect_search_mode()`: Auto-detect best mode from query

**chunking.py** - Code chunking strategies
- `ChunkingStrategy`: Enum of strategies (AST, paragraph, etc.)
- `ChunkingManager`: Orchestrates chunking with AST parsing
- Python AST-aware chunking preserves function/class boundaries
- Extracts metadata: decorators, arguments, domain tags

### Configuration Layer (`config/`)

**server.py** - Server configuration
- `ServerConfig`: Main config (host, port, codebases)
- `CodebaseConfig`: Per-codebase settings
- JSON/YAML file loading

**vector_db.py** - Vector DB configuration
- Chroma connection settings
- Embedding model configuration

### Service Layer (`service/`)

**memory_service.py** - Main orchestrator
- Manages multiple codebases
- Coordinates indexing, search, storage
- Provides both sync and async APIs
- Background file watching for auto-reindex

Key methods:
- `initialize()`: Start indexing, returns when complete
- `search()`: Hybrid search across codebases
- `store()`: Store new memories
- `prune()`: Remove old memories
- `reset_all()`: Clear all indexed data

### Server Layer (`server/`)

**sse.py** - Main SSE MCP server (port 9820)
- HTTP REST API endpoints
- SSE MCP endpoint at `/sse`
- Web dashboard at `/`
- Long-running service for production use

**stdio.py** - stdio MCP server
- For spawned process integration
- Uses FastMCP SDK
- Ideal for local development

**windows_service.py** - Windows Service wrapper
- Runs SSE server as Windows service
- Auto-start on boot support

### Client Layer (`client/`)

**tools.py** - HTTP client tools
- `MemorySearchTool`, `MemoryStoreTool`, etc.
- Compatible with agent tool registries

**base.py** - Tool base classes
- `Tool`: Abstract base for all tools
- `ToolResponse`: Standardized response format

## Data Flow

### Indexing Flow

1. **File Discovery**: Walk codebase, filter by extensions/ignore patterns
2. **Change Detection**: Check mtime/content hash against stored metadata
3. **Chunking**: AST parsing for Python, naive chunking for others
4. **Embedding**: Generate vectors via SentenceTransformer
5. **Storage**: Add to ChromaDB collection + BM25 index
6. **Metadata Update**: Store file index info for incremental updates

### Search Flow

1. **Mode Detection**: Analyze query for identifiers vs concepts
2. **Vector Search**: Generate query embedding, search ChromaDB
3. **Keyword Search**: Tokenize query, search BM25 index
4. **Fusion**: Combine results with Reciprocal Rank Fusion
5. **Deduplication**: Remove near-duplicate chunks
6. **Response**: Format results with metadata

## Multi-Codebase Architecture

Each codebase gets:
- Separate ChromaDB collection (`codebase_{name}`)
- Separate file index metadata collection
- Separate BM25 index instance

Cross-codebase search:
- Queries all vector stores in parallel
- Merges and re-ranks results
- Maintains per-codebase tags for filtering

## Memory Types

| Type | Purpose | Auto-Pin |
|------|---------|----------|
| `code` | Indexed source code chunks | No |
| `conversation` | Chat/session context | No |
| `decision` | Architectural decisions | Yes |
| `lesson` | Debugging insights | Yes |

Pinned memories are protected from automatic pruning.

## File Watching

Background task per codebase:
1. Sleep for `watch_interval` seconds (default 5)
2. Check all files for mtime changes
3. Re-verify with content hash (avoid false positives)
4. Index new files, re-index modified, remove deleted
5. Rebuild BM25 index after changes

## Performance Optimizations

- **Lazy loading**: Heavy dependencies loaded on first use
- **Batch processing**: Embeddings generated in batches of 32
- **Incremental indexing**: Only changed files re-processed
- **Content hashing**: Avoids re-indexing unchanged files with new mtime
- **In-memory BM25**: Fast keyword search without external service
- **Thread pool executor**: File watcher operations run in executor to avoid blocking event loop
- **Metadata caching**: 60-second TTL cache for file index metadata

## Storage Strategy

### Default: ChromaDB Only

ChromaDB handles both vectors and metadata:
- Vector embeddings for semantic search
- File content and chunked text
- File index metadata (mtime, hash, etc.)

This works well for small-to-medium codebases (<1000 files).

### Optional: ChromaDB + PostgreSQL

For large codebases, add PostgreSQL for metadata:

| Operation | ChromaDB | PostgreSQL |
|-----------|----------|------------|
| Vector search | ✓ | - |
| File content storage | ✓ | - |
| Embedding generation | ✓ | - |
| Dashboard file counts | Slow | ✓ Fast |
| Paginated file lists | Slow | ✓ Fast |
| Summary filtering | Slow | ✓ Fast |

PostgreSQL uses:
- **Materialized views** for instant count queries
- **Proper indexes** for filtered pagination
- **Async connection pooling** via asyncpg

Configuration:
```json
{
  "postgres_url": "postgresql://user:pass@host:5432/conductor_memory"
}
```

If PostgreSQL is unavailable, the system automatically falls back to ChromaDB.

## Extension Points

To add new functionality:

1. **New vector store**: Implement `VectorStore` interface
2. **New embedder**: Implement `Embedder` interface
3. **New chunking strategy**: Add to `ChunkingStrategy` enum
4. **New search mode**: Extend `HybridSearcher`
5. **New transport**: Add server module following existing patterns
