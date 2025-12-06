#!/usr/bin/env python3
"""
MCP Memory Server - stdio wrapper for OpenCode integration

Thin wrapper around MemoryService for stdio-based MCP interface.
Supports multi-codebase indexing via config file.

Usage:
    # Single codebase (backward compatible)
    python -m src.mcp_memory_stdio --codebase-path /path/to/code
    
    # Multiple codebases via config file (recommended)
    python -m src.mcp_memory_stdio --config memory_server_config.json
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# MCP SDK imports
from mcp.server.fastmcp import FastMCP

from ..config.server import ServerConfig
from ..service.memory_service import MemoryService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance (initialized in main)
memory_service: MemoryService | None = None

# Create the MCP server
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
def memory_search(
    query: str,
    max_results: int = 10,
    project_id: str | None = None,
    codebase: str | None = None,
    min_relevance: float = 0.1,
    search_mode: str = "auto"
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
            - auto: Automatically detect based on query (exact identifiers -> keyword, conceptual -> semantic)
            - semantic: Vector similarity only (best for conceptual queries like "how does X work")
            - keyword: BM25 keyword matching only (best for exact names like "calculate_score")
            - hybrid: Combines both with Reciprocal Rank Fusion (balanced approach)
    
    Returns:
        Dictionary with search results and metadata including search_mode_used
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "results": []}
    
    return memory_service.search(
        query=query,
        codebase=codebase,
        max_results=max_results,
        project_id=project_id,
        search_mode=search_mode
    )


@mcp.tool()
def memory_store(
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
    
    return memory_service.store(
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
def memory_store_decision(
    content: str,
    tags: list[str] | None = None,
    project_id: str = "default",
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Store an architectural decision for later retrieval.
    Decisions are automatically pinned and tagged as 'decision'.
    
    Use this when:
    - Choosing between alternative approaches
    - Establishing patterns that should be followed elsewhere
    - Making tradeoffs that future developers should understand
    - Deprecating an approach in favor of a new one
    
    Format your content like:
    ```
    DECISION: [One-line summary]
    CONTEXT: [What prompted this decision]
    ALTERNATIVES: [Other options considered]
    RATIONALE: [Why this choice was made]
    CONSEQUENCES: [Implications and tradeoffs]
    ```
    
    Args:
        content: The decision description (use format above)
        tags: Optional additional tags (auto-adds 'decision', 'architecture')
        project_id: Project identifier (default "default")
        codebase: Codebase to store in (default: first configured)
    
    Returns:
        Dictionary with stored memory details
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    
    # Build tags with defaults
    all_tags = list(tags or [])
    for default_tag in ["decision", "architecture"]:
        if default_tag not in all_tags:
            all_tags.append(default_tag)
    
    return memory_service.store(
        content=content,
        project_id=project_id,
        codebase=codebase,
        role="assistant",
        tags=all_tags,
        pin=True,  # Decisions are always pinned
        source="opencode",
        memory_type="decision"
    )


@mcp.tool()
def memory_store_lesson(
    content: str,
    tags: list[str] | None = None,
    project_id: str = "default",
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Store a debugging insight or lesson learned for later retrieval.
    Lessons are automatically pinned and tagged as 'lesson'.
    
    Use this when:
    - Solving a tricky bug that wasn't obvious
    - Discovering a gotcha or non-obvious behavior
    - Finding a workaround for a limitation
    - Learning something about the codebase that would help future work
    
    Format your content like:
    ```
    LESSON: [One-line summary]
    PROBLEM: [What went wrong or was confusing]
    SOLUTION: [What fixed it or the correct approach]
    CONTEXT: [Where/when this applies]
    ```
    
    Args:
        content: The lesson description (use format above)
        tags: Optional additional tags (auto-adds 'lesson', 'debugging')
        project_id: Project identifier (default "default")
        codebase: Codebase to store in (default: first configured)
    
    Returns:
        Dictionary with stored memory details
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    
    # Build tags with defaults
    all_tags = list(tags or [])
    for default_tag in ["lesson", "debugging"]:
        if default_tag not in all_tags:
            all_tags.append(default_tag)
    
    return memory_service.store(
        content=content,
        project_id=project_id,
        codebase=codebase,
        role="assistant",
        tags=all_tags,
        pin=True,  # Lessons are always pinned
        source="opencode",
        memory_type="lesson"
    )


@mcp.tool()
def memory_status() -> dict[str, Any]:
    """
    Get the current status of the memory system.
    
    Returns:
        Dictionary with memory system status including indexing progress
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    return memory_service.get_status()


@mcp.tool()
def memory_prune(
    project_id: str | None = None,
    max_age_days: int = 30
) -> dict[str, Any]:
    """
    Prune obsolete memories based on age and relevance.
    
    This will NOT prune:
    - Pinned memories (decisions, lessons)
    - Code chunks from codebase indexing
    - Memories younger than max_age_days
    
    Args:
        project_id: Optional project ID to filter pruning
        max_age_days: Maximum age in days for memories (default 30)
    
    Returns:
        Dictionary with pruning results
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "pruned": 0, "kept": 0, "total_processed": 0}
    
    return memory_service.prune(
        project_id=project_id,
        max_age_days=max_age_days
    )


@mcp.tool()
def memory_delete(
    memory_id: str,
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Delete a specific memory by ID.
    
    Use this to remove outdated decisions or lessons when they are superseded.
    Unlike prune, this can delete pinned memories (decisions, lessons).
    
    Typical workflow for superseding a decision:
    1. Search for the old decision to get its ID
    2. Store the new decision with memory_store_decision
    3. Delete the old decision with memory_delete
    
    Args:
        memory_id: The ID of the memory to delete (returned when storing)
        codebase: Optional codebase to delete from (searches all if not specified)
    
    Returns:
        Dictionary with deletion result including success status
    """
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    
    return memory_service.delete(
        memory_id=memory_id,
        codebase=codebase
    )


def main():
    """Main entry point for stdio MCP server"""
    global memory_service
    
    parser = argparse.ArgumentParser(description="MCP Memory Server (stdio)")
    parser.add_argument("--codebase-path", type=str, help="Path to codebase (single codebase mode)")
    parser.add_argument("--codebase-name", type=str, default="default", help="Name for codebase")
    parser.add_argument("--config", type=str, help="Config file path (default: memory_server_config.json if exists)")
    parser.add_argument("--persist-dir", type=str, default="./data/chroma", help="Persistent storage dir")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--reset", action="store_true", help="Clear all indexed data and reindex from scratch")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Default config file name
    DEFAULT_CONFIG = "memory_server_config.json"
    
    # Determine config file path
    config_path = args.config
    if not config_path and not args.codebase_path:
        # Auto-detect config file in current directory
        if Path(DEFAULT_CONFIG).exists():
            config_path = DEFAULT_CONFIG
            logger.info(f"Auto-detected config file: {DEFAULT_CONFIG}")
    
    # Load configuration
    if config_path:
        try:
            config = ServerConfig.from_file(config_path)
            logger.info(f"Loaded config from: {config_path}")
            logger.info(f"Configured {len(config.codebases)} codebase(s)")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    elif args.codebase_path:
        config = ServerConfig.create_default(
            codebase_path=args.codebase_path,
            codebase_name=args.codebase_name
        )
        config.persist_directory = args.persist_dir
        logger.info(f"Single codebase mode: {args.codebase_path}")
    else:
        logger.warning(f"No config file found. Create {DEFAULT_CONFIG} or use --codebase-path")
        config = ServerConfig()
        config.persist_directory = args.persist_dir
    
    # Create and initialize MemoryService (blocking - indexes before accepting requests)
    memory_service = MemoryService(config)
    
    # Handle --reset flag
    if args.reset:
        logger.info("Reset flag detected - clearing all indexed data...")
        memory_service.reset_all()
        logger.info("All data cleared. Will perform full reindex.")
    
    if config.get_enabled_codebases():
        logger.info("Initializing and indexing codebases...")
        memory_service.initialize()  # Sync version - blocks until complete
        logger.info("Indexing complete")
    
    # Log ready status with summary
    status = memory_service.get_status()
    total_files = sum(cb.get("indexed_files_count", 0) for cb in status.get("codebases", {}).values())
    logger.info(f"=== READY === Memory server initialized with {total_files} indexed files across {status.get('total_codebases', 0)} codebase(s)")
    
    logger.info("Starting MCP Memory Server (stdio transport)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
