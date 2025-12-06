#!/usr/bin/env python3
"""
MCP Memory Server - SSE (Server-Sent Events) version for remote connections

This server runs as an HTTP service that OpenCode can connect to remotely.
No need to spawn a process - just start this server once and connect from any project.

Usage:
    # Start the server (do this once, or add to Windows startup)
    python src/mcp_memory_sse.py
    
    # Then in any project's opencode.json:
    {
      "mcp": {
        "conductor_memory": {
          "type": "remote",
          "url": "http://localhost:9820/sse"
        }
      }
    }
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Any

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MCP SDK imports
from mcp.server.fastmcp import FastMCP

from ..config.server import ServerConfig
from ..service.memory_service import MemoryService

# Configure logging to stderr (stdout may be used by transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Global service instance (initialized in main)
memory_service: MemoryService | None = None

# Create the MCP server with SSE settings
mcp = FastMCP(
    "Conductor Memory",
    instructions="""
    Memory tools for semantic search and context retrieval. Provides:
    - Semantic search across multiple codebases and conversation history
    - Storage of conversation memories and code snippets
    - Multi-codebase support with per-codebase or cross-codebase search
    
    Use memory_search to find relevant code or past conversations.
    Use memory_store to save important context for later retrieval.
    Use memory_status to check indexing progress for all codebases.
    """
)


@mcp.tool()
async def memory_search(
    query: str,
    max_results: int = 10,
    project_id: str | None = None,
    codebase: str | None = None,
    min_relevance: float = 0.1,
    search_mode: str = "auto",
    # Tag filtering
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    # Heuristic filtering (Phase 2)
    languages: list[str] | None = None,
    class_names: list[str] | None = None,
    function_names: list[str] | None = None,
    annotations: list[str] | None = None,
    has_annotations: bool | None = None,
    has_docstrings: bool | None = None,
    min_class_count: int | None = None,
    min_function_count: int | None = None
) -> dict[str, Any]:
    """
    Search for relevant memories using semantic similarity, keyword matching, or both.
    
    Args:
        query: Search query for semantic similarity
        max_results: Maximum number of results to return (default 10)
        project_id: Optional filter by project ID
        codebase: Optional codebase name to search (None = search all codebases)
        min_relevance: Minimum relevance score 0-1 (default 0.1)
        search_mode: Search mode - "auto" (default), "semantic", "keyword", or "hybrid"
        include_tags: Include only results matching these tags (supports prefix:* patterns)
        exclude_tags: Exclude results matching these tags (supports prefix:* patterns)
        languages: Filter by programming languages (e.g., ['python', 'java'])
        class_names: Filter by class names (e.g., ['UserService', 'TestClass'])
        function_names: Filter by function names (e.g., ['process_data', 'validate'])
        annotations: Filter by annotations (e.g., ['@Test', '@Component'])
        has_annotations: Filter files that have/don't have annotations
        has_docstrings: Filter files that have/don't have docstrings
        min_class_count: Minimum number of classes in file
        min_function_count: Minimum number of functions in file
    
    Returns:
        Dictionary with search results and metadata including search_mode_used
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "results": []}
    
    # Use async method directly to avoid asyncio.run() conflict
    return await memory_service.search_async(
        query=query,
        codebase=codebase,
        max_results=max_results,
        project_id=project_id,
        search_mode=search_mode,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        languages=languages,
        class_names=class_names,
        function_names=function_names,
        annotations=annotations,
        has_annotations=has_annotations,
        has_docstrings=has_docstrings,
        min_class_count=min_class_count,
        min_function_count=min_function_count
    )


@mcp.tool()
async def memory_store(
    content: str,
    project_id: str = "default",
    codebase: str | None = None,
    role: str = "user",
    tags: list[str] | None = None,
    pin: bool = False,
    source: str = "opencode"
) -> dict[str, Any]:
    """
    Store a new memory chunk for later retrieval.
    
    Args:
        content: The text content to store
        project_id: Project identifier (default "default")
        codebase: Codebase to store in (default: first configured codebase)
        role: Role of the memory - user, assistant, system, tool (default "user")
        tags: Optional list of tags for categorization
        pin: Pin this memory to prevent pruning (default False)
        source: Source of the memory (default "opencode")
    
    Returns:
        Dictionary with stored memory details
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    
    # Use async method directly to avoid asyncio.run() conflict
    return await memory_service.store_async(
        content=content,
        project_id=project_id,
        codebase=codebase,
        role=role,
        tags=tags,
        pin=pin,
        source=source,
        memory_type="conversation"
    )


@mcp.tool()
async def memory_store_decision(
    content: str,
    tags: list[str] | None = None,
    project_id: str = "default",
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Store an architectural decision for later retrieval.
    Decisions are automatically pinned and tagged as 'decision'.
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    
    all_tags = list(tags or [])
    for default_tag in ["decision", "architecture"]:
        if default_tag not in all_tags:
            all_tags.append(default_tag)
    
    # Use async method directly to avoid asyncio.run() conflict
    return await memory_service.store_async(
        content=content,
        project_id=project_id,
        codebase=codebase,
        role="assistant",
        tags=all_tags,
        pin=True,
        source="opencode",
        memory_type="decision"
    )


@mcp.tool()
async def memory_store_lesson(
    content: str,
    tags: list[str] | None = None,
    project_id: str = "default",
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Store a debugging insight or lesson learned for later retrieval.
    Lessons are automatically pinned and tagged as 'lesson'.
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    
    all_tags = list(tags or [])
    for default_tag in ["lesson", "debugging"]:
        if default_tag not in all_tags:
            all_tags.append(default_tag)
    
    # Use async method directly to avoid asyncio.run() conflict
    return await memory_service.store_async(
        content=content,
        project_id=project_id,
        codebase=codebase,
        role="assistant",
        tags=all_tags,
        pin=True,
        source="opencode",
        memory_type="lesson"
    )


@mcp.tool()
async def memory_status() -> dict[str, Any]:
    """
    Get the current status of the memory system.
    
    Returns:
        Dictionary with memory system status including indexing progress
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    # get_status is sync and doesn't use asyncio.run(), so it's safe
    return memory_service.get_status()


@mcp.tool()
async def memory_prune(
    project_id: str | None = None,
    max_age_days: int = 30
) -> dict[str, Any]:
    """
    Prune obsolete memories based on age and relevance.
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "pruned": 0, "kept": 0, "total_processed": 0}
    
    # Use async method directly to avoid asyncio.run() conflict
    return await memory_service.prune_async(
        project_id=project_id,
        max_age_days=max_age_days
    )


@mcp.tool()
async def memory_delete(
    memory_id: str,
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Delete a specific memory by ID.
    
    Use this to remove outdated decisions or lessons when they are superseded.
    Unlike prune, this can delete pinned memories (decisions, lessons).
    
    Args:
        memory_id: The ID of the memory to delete (returned when storing)
        codebase: Optional codebase to delete from (searches all if not specified)
    
    Returns:
        Dictionary with deletion result
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    
    return await memory_service.delete_async(
        memory_id=memory_id,
        codebase=codebase
    )


@mcp.tool()
async def memory_import_graph_stats(
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Get import graph statistics for codebases.
    
    Args:
        codebase: Optional codebase name (None = all codebases)
    
    Returns:
        Dictionary with import graph statistics including file counts, edges, and centrality info
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    return memory_service.get_import_graph_stats(codebase)


@mcp.tool()
async def memory_file_centrality(
    codebase: str,
    max_files: int = 20
) -> dict[str, Any]:
    """
    Get files sorted by centrality score (importance in dependency graph).
    
    Files with higher centrality are more "central" to the codebase and are
    imported by many other files, making them good candidates for LLM summarization.
    
    Args:
        codebase: Codebase name
        max_files: Maximum number of files to return (default 20)
    
    Returns:
        Dictionary with list of files and their centrality scores
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    try:
        priority_queue = memory_service.get_file_centrality_scores(codebase, max_files)
        return {
            "codebase": codebase,
            "files": [
                {"file_path": file_path, "centrality_score": score}
                for file_path, score in priority_queue
            ],
            "total_files": len(priority_queue)
        }
    except Exception as e:
        return {"error": f"Failed to get centrality scores: {e}"}


@mcp.tool()
async def memory_file_dependencies(
    codebase: str,
    file_path: str
) -> dict[str, Any]:
    """
    Get dependency information for a specific file.
    
    Args:
        codebase: Codebase name
        file_path: Path to the file
    
    Returns:
        Dictionary with file dependency information including imports and imported_by lists
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    try:
        file_stats = memory_service.get_file_dependencies(codebase, file_path)
        if file_stats:
            return file_stats
        else:
            return {"error": f"File not found in import graph: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to get file dependencies: {e}"}


def main():
    """Main entry point for SSE MCP server"""
    global memory_service

    parser = argparse.ArgumentParser(description="MCP Memory Server (SSE/HTTP)")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9820, help="Port to listen on")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Load configuration - check multiple locations in priority order
    DEFAULT_HOME_CONFIG = Path.home() / ".conductor-memory" / "config.json"
    LEGACY_CONFIG = Path("memory_server_config.json")

    config_path = args.config

    # If no explicit config, check in priority order:
    # 1. CONDUCTOR_MEMORY_CONFIG environment variable
    # 2. ~/.conductor-memory/config.json (documented default)
    # 3. ./memory_server_config.json (legacy/backwards compat)
    if not config_path:
        env_config = os.environ.get("CONDUCTOR_MEMORY_CONFIG")
        if env_config and Path(env_config).exists():
            config_path = env_config
            logger.info(f"Using config from CONDUCTOR_MEMORY_CONFIG: {config_path}")
        elif DEFAULT_HOME_CONFIG.exists():
            config_path = str(DEFAULT_HOME_CONFIG)
            logger.info(f"Using default config: {config_path}")
        elif LEGACY_CONFIG.exists():
            config_path = str(LEGACY_CONFIG)
            logger.info(f"Using legacy config file: {config_path}")

    if config_path and Path(config_path).exists():
        try:
            config = ServerConfig.from_file(config_path)
            logger.info(f"Loaded config from: {config_path}")
            logger.info(f"Configured {len(config.codebases)} codebase(s)")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    else:
        logger.warning(f"No config file found. Create {DEFAULT_HOME_CONFIG} or use --config")
        config = ServerConfig()
    
    # Create and initialize MemoryService
    memory_service = MemoryService(config)
    
    if config.get_enabled_codebases():
        logger.info("Initializing and indexing codebases...")
        memory_service.initialize()
        logger.info("Indexing complete")
    
    # Log ready status
    status = memory_service.get_status()
    total_files = sum(cb.get("indexed_files_count", 0) for cb in status.get("codebases", {}).values())
    logger.info(f"=== READY === Memory server initialized with {total_files} indexed files")
    
    # Configure FastMCP settings for SSE
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    
    logger.info(f"Starting MCP Memory Server (SSE) on http://{args.host}:{args.port}/sse")
    logger.info("Configure OpenCode with:")
    logger.info(f'  "type": "remote", "url": "http://{args.host}:{args.port}/sse"')
    
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
