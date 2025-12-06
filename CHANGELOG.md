# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2024-12-06

### Added
- New MCP tool `memory_queue_codebase_summarization` to manually queue codebases for summarization
- Method `queue_codebase_for_summarization()` in MemoryService for programmatic access
- Test script `test_queue_summarization.py` for testing codebase queueing
- Support for queuing only missing files or re-summarizing entire codebases

### Fixed
- Issue where new codebases added after initial startup weren't automatically queued for summarization
- Summarization now properly initializes for all codebases on server restart

## [1.0.0] - 2024-12-06

### Added
- MIT License for open source distribution
- Comprehensive installation guide (INSTALL.md)
- Platform-specific installation scripts for Windows and Linux
- GitHub Actions workflow for automated releases and PyPI publishing
- Cross-platform packaging configuration
- Initial release of conductor-memory
- Semantic memory service with codebase indexing for AI agents
- Hybrid search combining semantic vectors + BM25 keyword search
- Heuristic filtering by classes, functions, annotations, and file types
- Multi-language AST-aware parsing for 15+ programming languages
- Multi-codebase support with incremental indexing
- LLM integration for background file summarization via Ollama
- MCP (Model Context Protocol) server integration
- Web dashboard for real-time monitoring
- RESTful API for integration
- Advanced search parameters and relevance boosting
- Production-ready configuration system

### Changed
- Updated project URLs in pyproject.toml to point to correct GitHub repository

### Added
- Initial release of conductor-memory
- Semantic memory service with codebase indexing for AI agents
- Hybrid search combining semantic vectors + BM25 keyword search
- Heuristic filtering by classes, functions, annotations, and file types
- Multi-language AST-aware parsing for 15+ programming languages
- Multi-codebase support with incremental indexing
- LLM integration for background file summarization via Ollama
- MCP (Model Context Protocol) server integration
- Web dashboard for real-time monitoring
- RESTful API for integration
- Advanced search parameters and relevance boosting
- Production-ready configuration system

### Features
- **Advanced Search**: Hybrid semantic + keyword matching with Reciprocal Rank Fusion
- **Intelligent Indexing**: Multi-language support, incremental updates, metadata extraction
- **LLM Integration**: Background summarization with time estimation and progress tracking
- **Production Ready**: MCP integration, web dashboard, RESTful API, robust configuration

### Supported Languages
- Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Ruby, Kotlin, Swift, Objective-C

### MCP Tools
- `memory_search` - Advanced search with filtering and summary integration
- `memory_store` - Store important context for later retrieval
- `memory_store_decision` - Store architectural decisions (auto-pinned)
- `memory_store_lesson` - Store debugging insights and lessons learned
- `memory_status` - Check indexing status and memory system health
- `memory_summarization_status` - Check LLM summarization progress
- `memory_reindex_codebase` - Force reindexing of specific codebase
- `memory_prune` - Remove obsolete memories based on age/relevance
- `memory_delete` - Delete specific memory by ID