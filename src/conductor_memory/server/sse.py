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
from starlette.responses import JSONResponse

from ..config.server import ServerConfig
from ..service.memory_service import MemoryService
from .chroma_manager import ChromaServerManager

# Configure logging to stderr (stdout may be used by transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Global service instance (initialized in main)
memory_service: MemoryService | None = None
chroma_manager: ChromaServerManager | None = None

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


# Health check endpoint for service monitoring
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for service readiness detection."""
    if memory_service:
        status = memory_service.get_status()
        return JSONResponse({
            "status": "healthy",
            "codebases": len(status.get("codebases", {})),
            "indexed_files": sum(
                cb.get("indexed_files_count", 0) 
                for cb in status.get("codebases", {}).values()
            )
        })
    return JSONResponse({"status": "initializing"}, status_code=503)


# Summarization status API endpoint
@mcp.custom_route("/api/summarization", methods=["GET"])
async def summarization_status_api(request):
    """Get summarization status as JSON (fast, uses cached counts)."""
    if memory_service:
        # Use basic method to avoid expensive ChromaDB queries on every poll
        return JSONResponse(memory_service.get_summarization_status_basic())
    return JSONResponse({"error": "Service not initialized"}, status_code=503)


# Indexing status API endpoint
@mcp.custom_route("/api/status", methods=["GET"])
async def indexing_status_api(request):
    """Get full service status as JSON."""
    if memory_service:
        return JSONResponse(memory_service.get_status())
    return JSONResponse({"error": "Service not initialized"}, status_code=503)


# Ultra-fast counts endpoint for dashboard polling
@mcp.custom_route("/api/status/counts", methods=["GET"])
async def status_counts_api(request):
    """Get lightweight status counts (optimized for polling)."""
    if memory_service:
        return JSONResponse(memory_service.get_status_counts())
    return JSONResponse({"error": "Service not initialized"}, status_code=503)


# ChromaDB status API endpoint
@mcp.custom_route("/api/chroma-status", methods=["GET"])
async def chroma_status_api(request):
    """Get ChromaDB connection status."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    config = memory_service.config
    status = {
        "mode": config.chroma_mode,
        "host": config.chroma_host if config.chroma_mode == "http" else None,
        "port": config.chroma_port if config.chroma_mode == "http" else None,
    }
    
    if config.chroma_mode == "http":
        # Check if ChromaDB server is healthy
        if chroma_manager:
            status["process_running"] = chroma_manager.is_running()
            status["server_healthy"] = await chroma_manager.is_healthy()
        else:
            status["process_running"] = False
            status["server_healthy"] = False
        
        # Try to get collection info
        try:
            # Get first vector store to check connection
            if memory_service._vector_stores:
                store = next(iter(memory_service._vector_stores.values()))
                status["collections"] = len(store.client.list_collections())
                status["connected"] = True
        except Exception as e:
            status["connected"] = False
            status["error"] = str(e)
    else:
        status["persist_directory"] = config.persist_directory
        status["connected"] = True
    
    return JSONResponse(status)


# Search API endpoint for web dashboard
@mcp.custom_route("/api/search", methods=["POST"])
async def api_search(request):
    """Search endpoint for the web dashboard."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        body = await request.json()
        
        result = await memory_service.search_async(
            query=body.get("query", ""),
            codebase=body.get("codebase"),
            max_results=body.get("max_results", 10),
            search_mode=body.get("search_mode", "auto"),
            include_tags=body.get("include_tags"),
            exclude_tags=body.get("exclude_tags"),
            languages=body.get("languages"),
            class_names=body.get("class_names"),
            function_names=body.get("function_names"),
            has_annotations=body.get("has_annotations"),
            has_docstrings=body.get("has_docstrings"),
            min_class_count=body.get("min_class_count"),
            min_function_count=body.get("min_function_count"),
            calls=body.get("calls"),
            accesses=body.get("accesses"),
            subscripts=body.get("subscripts"),
            include_summaries=body.get("include_summaries", False),
            boost_summarized=body.get("boost_summarized", True)
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# List indexed files API endpoint
@mcp.custom_route("/api/files", methods=["GET"])
async def api_list_files(request):
    """List indexed files with pagination and filtering."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        
        # Parse has_summary filter
        has_summary = None
        has_summary_param = params.get("has_summary")
        if has_summary_param == "yes":
            has_summary = True
        elif has_summary_param == "no":
            has_summary = False
        
        result = await memory_service.list_indexed_files_async(
            codebase=params.get("codebase"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0)),
            search_filter=params.get("filter"),
            has_summary=has_summary,
            pattern=params.get("pattern") or None,
            domain=params.get("domain") or None,
            language=params.get("language") or None
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Get file details API endpoint
@mcp.custom_route("/api/file-details", methods=["GET"])
async def api_file_details(request):
    """Get details for a specific indexed file."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        codebase = params.get("codebase")
        file_path = params.get("path")
        
        if not codebase or not file_path:
            return JSONResponse({"error": "codebase and path are required"}, status_code=400)
        
        result = await memory_service.get_file_details_async(codebase, file_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# List memories API endpoint
@mcp.custom_route("/api/memories", methods=["GET"])
async def api_list_memories(request):
    """List stored memories (conversations, decisions, lessons)."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        result = await memory_service.list_memories_async(
            memory_type=params.get("type"),
            codebase=params.get("codebase"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0))
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# List summaries API endpoint
@mcp.custom_route("/api/summaries", methods=["GET"])
async def api_list_summaries(request):
    """List files with LLM-generated summaries."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        result = await memory_service.list_summaries_async(
            codebase=params.get("codebase"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0))
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# List available codebases API endpoint
@mcp.custom_route("/api/codebases", methods=["GET"])
async def api_list_codebases(request):
    """List all configured codebases (fast, uses cached counts)."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        # Use basic method to avoid expensive ChromaDB queries
        codebases = memory_service.list_codebases_basic()
        return JSONResponse({"codebases": codebases})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Summary statistics API endpoint
@mcp.custom_route("/api/summary-stats", methods=["GET"])
async def api_summary_stats(request):
    """Get aggregated summary statistics by pattern, domain, and validation status."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        codebase = params.get("codebase") or None  # Empty string -> None for all codebases
        
        result = await memory_service.get_summary_statistics_async(codebase=codebase)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Validation queue API endpoint
@mcp.custom_route("/api/validation-queue", methods=["GET"])
async def api_validation_queue(request):
    """Get files for summary validation with pagination and filtering."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        codebase = params.get("codebase")
        status = params.get("status", "unreviewed")
        pattern = params.get("pattern", "")
        domain = params.get("domain", "")
        sort = params.get("sort", "recent")
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 20))

        if not codebase:
            return JSONResponse({"error": "codebase is required"}, status_code=400)

        result = await memory_service.get_validation_queue_async(
            codebase=codebase,
            status=status,
            pattern=pattern,
            domain=domain,
            offset=offset,
            limit=limit,
            sort=sort
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# File content API endpoint (for validation)
@mcp.custom_route("/api/file-content", methods=["GET"])
async def api_file_content(request):
    """Get raw file content for validation."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        codebase = params.get("codebase")
        file_path = params.get("path")
        
        if not codebase or not file_path:
            return JSONResponse({"error": "codebase and path are required"}, status_code=400)
        
        result = await memory_service.get_file_content_async(codebase, file_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Summary-only API endpoint (lightweight, for validation)
@mcp.custom_route("/api/file-summary", methods=["GET"])
async def api_file_summary(request):
    """Get just the summary for a file (fast, no chunks)."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        codebase = params.get("codebase")
        file_path = params.get("path")
        
        if not codebase or not file_path:
            return JSONResponse({"error": "codebase and path are required"}, status_code=400)
        
        result = await memory_service.get_file_summary_async(codebase, file_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Summary validation API endpoint
@mcp.custom_route("/api/summary/validate", methods=["POST"])
async def api_validate_summary(request):
    """Validate (approve/reject) a summary."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        body = await request.json()
        codebase = body.get("codebase")
        file_path = body.get("file_path")
        status = body.get("status")  # "approved" or "rejected"
        
        if not all([codebase, file_path, status]):
            return JSONResponse({"error": "codebase, file_path, and status are required"}, status_code=400)
        
        result = await memory_service.validate_summary_async(codebase, file_path, status)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Summary regeneration API endpoint
@mcp.custom_route("/api/summary/regenerate", methods=["POST"])
async def api_regenerate_summary(request):
    """Regenerate a summary for a file."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        body = await request.json()
        codebase = body.get("codebase")
        file_path = body.get("file_path")
        
        if not codebase or not file_path:
            return JSONResponse({"error": "codebase and file_path are required"}, status_code=400)
        
        result = await memory_service.regenerate_summary_async(codebase, file_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Queue summarization API endpoint (for dashboard action buttons)
@mcp.custom_route("/api/queue-summarization", methods=["POST"])
async def api_queue_summarization(request):
    """Queue a codebase for summarization."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        body = await request.json()
        codebase = body.get("codebase")
        only_missing = body.get("only_missing", True)
        
        if not codebase:
            return JSONResponse({"error": "codebase is required"}, status_code=400)
        
        result = await memory_service.queue_codebase_for_summarization(codebase, only_missing)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Invalidate summaries API endpoint (for dashboard action buttons)
@mcp.custom_route("/api/invalidate-summaries", methods=["POST"])
async def api_invalidate_summaries(request):
    """Invalidate all summaries to force re-summarization."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        result = await memory_service.invalidate_summaries_async()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Reindex codebase API endpoint (for dashboard action buttons)
@mcp.custom_route("/api/reindex", methods=["POST"])
async def api_reindex_codebase(request):
    """Reindex a specific codebase."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        body = await request.json()
        codebase = body.get("codebase")
        
        if not codebase:
            return JSONResponse({"error": "codebase is required"}, status_code=400)
        
        result = await memory_service.reindex_codebase_async(codebase)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# Web Dashboard with Search and Browse
@mcp.custom_route("/", methods=["GET"])
async def web_dashboard(request):
    """Full-featured HTML dashboard with Status, Search, and Browse tabs."""
    from starlette.responses import HTMLResponse
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conductor Memory - Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }
        
        /* Header */
        .header {
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .header h1 { color: #00d4ff; font-size: 20px; }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 5px;
        }
        .tab {
            padding: 8px 16px;
            background: transparent;
            border: 1px solid #0f3460;
            border-radius: 4px;
            color: #888;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .tab:hover { border-color: #00d4ff; color: #ccc; }
        .tab.active { background: #0f3460; color: #00d4ff; border-color: #00d4ff; }
        
        /* Main content */
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px 40px;
            height: calc(100vh - 60px);
            overflow-y: auto;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        /* Full height tabs - summaries and browse */
        #summaries-tab.active, #browse-tab.active {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 100px);
            overflow: hidden;
        }
        #summaries-tab h2, #browse-tab h2 { flex-shrink: 0; }
        .container:has(#summaries-tab.active), .container:has(#browse-tab.active) {
            overflow: hidden;
        }
        
        /* Cards */
        h2 { color: #888; font-size: 14px; text-transform: uppercase; margin: 20px 0 10px; }
        .card { 
            background: #16213e; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 15px;
            border: 1px solid #0f3460;
        }
        
        /* Stats grid */
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
        .stat { text-align: center; padding: 12px; background: #0f3460; border-radius: 6px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 11px; color: #888; margin-top: 4px; }
        
        /* Status colors */
        .status-running { color: #00ff88; }
        .status-idle { color: #888; }
        .status-error { color: #ff6b6b; }
        
        /* Progress bar */
        .progress-bar { height: 6px; background: #0f3460; border-radius: 3px; overflow: hidden; margin-top: 10px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00d4ff, #00ff88); transition: width 0.5s ease; }
        
        /* Current file indicator */
        .current-file { 
            font-family: monospace; font-size: 12px; color: #00d4ff;
            padding: 6px 10px; background: #0f3460; border-radius: 4px;
            margin-top: 10px; word-break: break-all;
        }
        
        /* Codebase list */
        .codebase { 
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 0; border-bottom: 1px solid #0f3460;
        }
        .codebase:last-child { border-bottom: none; }
        .codebase-name { font-weight: bold; }
        .codebase-stats { color: #888; font-size: 13px; }
        
        /* Error message */
        .error-msg { 
            color: #ff6b6b; font-size: 13px; margin-top: 10px;
            padding: 8px 12px; background: rgba(255, 107, 107, 0.1); border-radius: 4px;
        }
        
        /* Search tab styles */
        .search-form { display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }
        .search-input { 
            flex: 1; min-width: 200px; padding: 10px 14px; 
            background: #0f3460; border: 1px solid #1a3a6e; border-radius: 6px;
            color: #eee; font-size: 14px;
        }
        .search-input:focus { outline: none; border-color: #00d4ff; }
        .search-btn {
            padding: 10px 20px; background: #00d4ff; border: none; border-radius: 6px;
            color: #1a1a2e; font-weight: bold; cursor: pointer; transition: background 0.2s;
        }
        .search-btn:hover { background: #00b8e6; }
        .search-btn:disabled { background: #555; cursor: not-allowed; }
        
        /* Search options row */
        .search-options { 
            display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; align-items: center;
        }
        .search-options label { font-size: 13px; color: #888; }
        .search-options select, .search-options input[type="number"] {
            padding: 6px 10px; background: #0f3460; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #eee; font-size: 13px;
        }
        
        /* Advanced filters toggle */
        .advanced-toggle {
            color: #00d4ff; cursor: pointer; font-size: 13px; 
            display: flex; align-items: center; gap: 5px;
        }
        .advanced-filters { 
            display: none; padding: 15px; background: #0f3460; 
            border-radius: 6px; margin-bottom: 15px;
        }
        .advanced-filters.open { display: block; }
        .filter-row { display: flex; gap: 15px; margin-bottom: 10px; flex-wrap: wrap; }
        .filter-group { display: flex; flex-direction: column; gap: 4px; }
        .filter-group label { font-size: 12px; color: #888; }
        .filter-group input { 
            padding: 6px 10px; background: #16213e; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #eee; font-size: 13px; width: 150px;
        }
        .checkbox-group { display: flex; align-items: center; gap: 6px; }
        .checkbox-group input { width: auto; }
        
        /* Search type filters */
        .search-type-filters {
            display: flex; gap: 10px; align-items: center;
            margin-bottom: 12px; padding: 10px 0;
            border-bottom: 1px solid #0f3460;
        }
        .type-filter-label {
            font-size: 13px; color: #888; margin-right: 5px;
        }
        .type-checkbox {
            cursor: pointer; display: flex; align-items: center;
        }
        .type-checkbox input {
            display: none;
        }
        .type-checkbox .type-badge {
            padding: 4px 10px; border-radius: 12px; font-size: 12px;
            border: 1px solid transparent; transition: all 0.2s;
        }
        .type-checkbox input:checked + .type-badge {
            border-color: currentColor;
        }
        .type-checkbox input:not(:checked) + .type-badge {
            opacity: 0.4;
        }
        .type-code .type-badge { background: rgba(0, 255, 136, 0.15); color: #00ff88; }
        .type-decision .type-badge { background: rgba(0, 136, 255, 0.15); color: #0088ff; }
        .type-lesson .type-badge { background: rgba(255, 170, 0, 0.15); color: #ffaa00; }
        .type-conversation .type-badge { background: rgba(136, 136, 136, 0.15); color: #888; }
        
        /* Search results */
        .results-header { 
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 10px; color: #888; font-size: 13px;
        }
        .result-item {
            background: #16213e; border-radius: 8px; padding: 0; margin-bottom: 12px;
            border: 1px solid #0f3460; overflow: hidden;
        }
        .result-item-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 12px; border-bottom: 1px solid #0f3460;
        }
        .result-type-badge {
            display: flex; align-items: center; gap: 8px;
        }
        .result-type-icon {
            width: 10px; height: 10px; border-radius: 50%;
        }
        .result-type-icon.code { background: #00ff88; }
        .result-type-icon.decision { background: #0088ff; }
        .result-type-icon.lesson { background: #ffaa00; }
        .result-type-icon.conversation { background: #888; }
        .result-type-label {
            font-size: 11px; text-transform: uppercase; font-weight: bold;
            letter-spacing: 0.5px;
        }
        .result-type-label.code { color: #00ff88; }
        .result-type-label.decision { color: #0088ff; }
        .result-type-label.lesson { color: #ffaa00; }
        .result-type-label.conversation { color: #888; }
        .result-pinned {
            font-size: 10px; color: #555; margin-left: 5px;
        }
        .result-score-bar {
            display: flex; align-items: center; gap: 8px;
        }
        .score-segments {
            display: flex; gap: 2px;
        }
        .score-segment {
            width: 8px; height: 12px; border-radius: 2px;
            background: #0f3460;
        }
        .score-segment.filled { background: #00d4ff; }
        .score-segment.high { background: #00ff88; }
        .score-value {
            font-size: 12px; color: #888; font-family: monospace;
        }
        .result-file { 
            font-family: monospace; color: #00d4ff; font-size: 13px;
            padding: 8px 12px; background: #0f3460;
        }
        .result-body {
            padding: 12px;
        }
        .result-tags { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }
        .tag { 
            background: #0f3460; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; color: #888;
        }
        .result-content {
            font-family: monospace; font-size: 12px; color: #ccc;
            white-space: pre-wrap; overflow: hidden;
            max-height: 80px; position: relative;
        }
        .result-content.expanded { max-height: none; }
        .result-actions {
            display: flex; justify-content: space-between; align-items: center;
            margin-top: 10px;
        }
        .expand-btn {
            color: #00d4ff; cursor: pointer; font-size: 12px;
        }
        .copy-btn {
            padding: 4px 10px; background: #0f3460; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #888; cursor: pointer; font-size: 11px;
            transition: all 0.2s;
        }
        .copy-btn:hover {
            border-color: #00d4ff; color: #00d4ff;
        }
        .copy-btn.copied {
            border-color: #00ff88; color: #00ff88;
        }
        
        /* Search result summary preview */
        .result-summary-preview {
            margin: 8px 12px 0 12px;
            padding: 10px 12px;
            background: #0f3460;
            border-radius: 6px;
            border-left: 3px solid #00d4ff;
        }
        .result-summary-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
            margin-bottom: 6px;
        }
        .result-summary-purpose {
            font-size: 12px;
            color: #ccc;
            line-height: 1.4;
            flex: 1;
        }
        .result-summary-badges {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            flex-shrink: 0;
        }
        .result-summary-badges .pattern-badge {
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            font-weight: bold;
            text-transform: uppercase;
            background: rgba(0, 212, 255, 0.15);
            color: #00d4ff;
        }
        .result-summary-badges .domain-badge {
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            background: rgba(0, 255, 136, 0.15);
            color: #00ff88;
        }
        .result-summary-meta {
            display: flex;
            gap: 12px;
            font-size: 11px;
            color: #888;
        }
        .result-summary-meta .has-indicator {
            color: #00d4ff;
        }
        .result-summary-meta .has-indicator::before {
            content: "✓ ";
        }
        
        /* Browse tab styles */
        .browse-controls {
            display: flex; gap: 15px; margin-bottom: 10px; flex-wrap: wrap; align-items: center;
            flex-shrink: 0;
        }
        .browse-controls select {
            padding: 8px 12px; background: #0f3460; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #eee; font-size: 14px;
        }
        .filter-input {
            padding: 8px 12px; background: #0f3460; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #eee; font-size: 14px; width: 200px;
        }
        
        /* Quick Filters */
        .browse-quick-filters {
            display: flex; gap: 12px; margin-bottom: 15px; flex-wrap: wrap;
            align-items: center; padding: 10px 15px;
            background: #0f3460; border-radius: 6px;
            flex-shrink: 0;
        }
        .browse-quick-filters label {
            font-size: 12px; color: #888;
            display: flex; align-items: center; gap: 6px;
        }
        .browse-quick-filters select {
            padding: 5px 8px; background: #16213e; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #eee; font-size: 12px;
        }
        .browse-quick-filters select:focus {
            outline: none; border-color: #00d4ff;
        }
        .browse-quick-filters select.filter-active {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
        }
        .filter-reset-btn {
            color: #ff6b6b; cursor: pointer; font-size: 12px;
            padding: 4px 8px; border-radius: 4px;
            transition: background 0.2s;
        }
        .filter-reset-btn:hover {
            background: rgba(255, 107, 107, 0.1);
        }
        
        /* Master-Detail Split Layout */
        .browse-split-container {
            display: grid;
            grid-template-columns: 30% 70%;
            gap: 20px;
            flex: 1;
            min-height: 0;
        }
        .browse-list-panel {
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .browse-list-header {
            padding: 12px 15px;
            border-bottom: 1px solid #0f3460;
            font-size: 13px;
            color: #888;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .browse-view-toggle {
            display: flex;
            gap: 4px;
        }
        .view-toggle-btn {
            padding: 4px 8px;
            background: transparent;
            border: 1px solid #1a3a6e;
            border-radius: 4px;
            color: #555;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        .view-toggle-btn:hover {
            border-color: #00d4ff;
            color: #888;
        }
        .view-toggle-btn.active {
            background: #0f3460;
            border-color: #00d4ff;
            color: #00d4ff;
        }
        /* Tree view styles */
        .tree-node {
            user-select: none;
        }
        .tree-folder {
            padding: 6px 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            color: #888;
            transition: background 0.15s;
        }
        .tree-folder:hover {
            background: rgba(0, 212, 255, 0.05);
        }
        .tree-folder-icon {
            font-size: 12px;
            width: 16px;
            text-align: center;
            transition: transform 0.15s;
        }
        .tree-folder.expanded .tree-folder-icon {
            transform: rotate(90deg);
        }
        .tree-folder-name {
            font-family: monospace;
            font-size: 12px;
        }
        .tree-folder-count {
            font-size: 10px;
            color: #555;
            margin-left: auto;
        }
        .tree-children {
            display: none;
            padding-left: 16px;
        }
        .tree-children.expanded {
            display: block;
        }
        .tree-file {
            padding: 5px 10px 5px 26px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            color: #ccc;
            font-family: monospace;
            font-size: 12px;
            transition: background 0.15s;
        }
        .tree-file:hover {
            background: rgba(0, 212, 255, 0.05);
        }
        .tree-file.selected {
            background: rgba(0, 212, 255, 0.15);
            border-left: 2px solid #00d4ff;
        }
        .tree-file-icon {
            font-size: 10px;
            color: #555;
        }
        .tree-file-meta {
            margin-left: auto;
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .browse-list-content {
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }
        .browse-detail-panel {
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .browse-detail-header {
            padding: 12px 15px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .browse-detail-title {
            font-family: monospace;
            color: #00d4ff;
            font-size: 14px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .browse-detail-content {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            min-height: 0;
        }
        .browse-detail-empty {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #555;
            font-size: 14px;
        }
        
        /* File list items */
        .file-list-item {
            padding: 10px 15px;
            border-bottom: 1px solid #0f3460;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.15s;
        }
        .file-list-item:hover {
            background: rgba(0, 212, 255, 0.05);
        }
        .file-list-item.selected {
            background: rgba(0, 212, 255, 0.15);
            border-left: 3px solid #00d4ff;
            padding-left: 12px;
        }
        .file-list-item:focus {
            outline: none;
            background: rgba(0, 212, 255, 0.1);
        }
        .file-item-path {
            font-family: monospace;
            font-size: 13px;
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex: 1;
        }
        .file-list-item.selected .file-item-path {
            color: #00d4ff;
        }
        .file-item-meta {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-shrink: 0;
            margin-left: 10px;
        }
        .file-item-chunks {
            background: #0f3460;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            color: #888;
        }
        .file-item-summary-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4a2d2d;
        }
        .file-item-summary-indicator.has-summary {
            background: #00ff88;
        }
        
        /* Summarization type badges */
        .summary-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 500;
            cursor: help;
        }
        .summary-badge.simple {
            background: rgba(147, 112, 219, 0.2);
            color: #b794f4;
            border: 1px solid rgba(147, 112, 219, 0.3);
        }
        .summary-badge.llm {
            background: rgba(72, 187, 120, 0.2);
            color: #68d391;
            border: 1px solid rgba(72, 187, 120, 0.3);
        }
        .summary-badge.pending {
            background: rgba(237, 137, 54, 0.2);
            color: #f6ad55;
            border: 1px solid rgba(237, 137, 54, 0.3);
        }
        .summary-badge-icon {
            font-size: 10px;
        }
        
        /* Detail panel sections */
        .detail-meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        .detail-meta-item {
            background: #0f3460;
            padding: 10px;
            border-radius: 6px;
        }
        .detail-meta-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .detail-meta-value {
            font-family: monospace;
            font-size: 13px;
            color: #eee;
            word-break: break-all;
        }
        
        /* Pagination for split view */
        .browse-list-footer {
            padding: 10px 15px;
            border-top: 1px solid #0f3460;
            display: flex;
            justify-content: center;
            gap: 5px;
        }
        
        /* Table styles */
        .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .data-table th { 
            text-align: left; padding: 10px; background: #0f3460;
            color: #888; font-weight: normal; text-transform: uppercase; font-size: 11px;
        }
        .data-table td { padding: 10px; border-bottom: 1px solid #0f3460; }
        .data-table tr:hover { background: rgba(0, 212, 255, 0.05); }
        .data-table tr.selected { background: rgba(0, 212, 255, 0.1); }
        .clickable { cursor: pointer; }
        .badge {
            display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; background: #16213e;
        }
        .badge-decision { background: #2d4a3e; color: #00ff88; }
        .badge-lesson { background: #4a3a2d; color: #ffaa00; }
        .badge-conversation { background: #2d3a4a; color: #00d4ff; }
        .badge-yes { background: #2d4a3e; color: #00ff88; }
        .badge-no { background: #4a2d2d; color: #888; }
        
        /* Pagination */
        .pagination { 
            display: flex; justify-content: center; gap: 5px; margin-top: 15px;
        }
        .page-btn {
            padding: 6px 12px; background: #0f3460; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #888; cursor: pointer; font-size: 13px;
        }
        .page-btn:hover { border-color: #00d4ff; color: #eee; }
        .page-btn.active { background: #00d4ff; color: #1a1a2e; border-color: #00d4ff; }
        .page-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        /* Legacy details panel (kept for backwards compat, unused in new layout) */
        .details-panel { display: none; }
        .details-panel.open { display: none; }
        
        /* Validate Tab Styles - 3-column layout */
        .validate-filter-bar {
            display: flex;
            gap: 15px;
            padding: 12px 15px;
            background: #16213e;
            border-radius: 8px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            align-items: center;
            flex-shrink: 0;
        }
        .validate-filter-bar label {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: #888;
        }
        .validate-filter-bar select {
            padding: 6px 10px;
            background: #0f3460;
            border: 1px solid #1a3a6e;
            border-radius: 4px;
            color: #eee;
            font-size: 13px;
        }
        .validate-filter-bar select:hover {
            border-color: #00d4ff;
        }
        .validate-progress {
            margin-left: auto;
            padding: 6px 12px;
            background: #0f3460;
            border-radius: 4px;
            font-size: 13px;
            color: #00d4ff;
        }
        .validate-refresh-btn {
            padding: 6px 10px;
            background: #0f3460;
            border: 1px solid #1a3a6e;
            border-radius: 4px;
            color: #888;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .validate-refresh-btn:hover {
            border-color: #00d4ff;
            color: #00d4ff;
        }
        .validate-filter-reset {
            color: #ff6b6b;
            cursor: pointer;
            font-size: 12px;
        }
        .validate-filter-reset:hover {
            color: #ff8888;
        }
        
        .validate-split-container {
            display: grid;
            grid-template-columns: 25% 1fr 1fr;
            gap: 15px;
            flex: 1;
            min-height: 0;
            margin-bottom: 15px;
        }
        
        /* File list panel (left) */
        .validate-file-list {
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .validate-file-list-header {
            padding: 10px 12px;
            border-bottom: 1px solid #0f3460;
            font-size: 11px;
            color: #888;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .validate-file-list-content {
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }
        .validate-file-list-footer {
            padding: 8px;
            border-top: 1px solid #0f3460;
            display: flex;
            justify-content: center;
            gap: 3px;
            flex-wrap: wrap;
        }
        .validate-file-list-footer .page-btn {
            padding: 4px 8px;
            font-size: 11px;
        }
        
        /* File list items */
        .validate-file-item {
            padding: 8px 12px;
            border-bottom: 1px solid #0f3460;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            transition: background 0.15s;
        }
        .validate-file-item:hover {
            background: rgba(0, 212, 255, 0.05);
        }
        .validate-file-item.selected {
            background: rgba(0, 212, 255, 0.15);
            border-left: 3px solid #00d4ff;
            padding-left: 9px;
        }
        .validate-file-item-path {
            font-family: monospace;
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex: 1;
        }
        .validate-file-item.selected .validate-file-item-path {
            color: #00d4ff;
        }
        
        /* Status icons */
        .validate-status-icon {
            margin-left: 8px;
            font-size: 11px;
            flex-shrink: 0;
        }
        .validate-status-icon.approved { color: #00ff88; }
        .validate-status-icon.rejected { color: #ff6b6b; }
        .validate-status-icon.unreviewed { color: #555; }
        
        /* Simple file styling */
        .validate-file-item.simple-file {
            opacity: 0.75;
        }
        .validate-file-item.simple-file:hover {
            opacity: 1;
        }
        .validate-simple-badge {
            margin-right: 6px;
            font-size: 10px;
            flex-shrink: 0;
        }
        .validate-simple-header-badge {
            display: inline-block;
            background: #1a4a3a;
            color: #4ade80;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            margin-left: 8px;
        }
        
        /* Content panels (middle and right) */
        .validate-panel {
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .validate-panel-header {
            padding: 10px 15px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .validate-panel-title {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }
        .validate-file-path {
            font-family: monospace;
            font-size: 11px;
            color: #00d4ff;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 300px;
        }
        .validate-model {
            font-size: 11px;
            color: #555;
        }
        .validate-panel-content {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            color: #ccc;
            min-height: 0;
        }
        .validate-source .validate-panel-content {
            background: #0f3460;
            color: #aaa;
        }
        
        /* Action bar */
        .validate-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
            flex-shrink: 0;
        }
        .validate-action-group {
            display: flex;
            gap: 10px;
        }
        .validate-btn {
            padding: 8px 16px;
            border: 1px solid #1a3a6e;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
            background: #0f3460;
            color: #888;
        }
        .validate-btn:hover:not(:disabled) {
            border-color: #00d4ff;
            color: #eee;
        }
        .validate-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .validate-btn-approve {
            background: rgba(0, 255, 136, 0.1);
            border-color: rgba(0, 255, 136, 0.3);
            color: #00ff88;
        }
        .validate-btn-approve:hover:not(:disabled) {
            background: rgba(0, 255, 136, 0.2);
            border-color: #00ff88;
        }
        .validate-btn-reject {
            background: rgba(255, 107, 107, 0.1);
            border-color: rgba(255, 107, 107, 0.3);
            color: #ff6b6b;
        }
        .validate-btn-reject:hover:not(:disabled) {
            background: rgba(255, 107, 107, 0.2);
            border-color: #ff6b6b;
        }
        .validate-btn-regenerate {
            background: rgba(0, 212, 255, 0.1);
            border-color: rgba(0, 212, 255, 0.3);
            color: #00d4ff;
        }
        .validate-btn-regenerate:hover:not(:disabled) {
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
        }
        .chunk-list { max-height: 400px; overflow-y: auto; }
        .chunk-item {
            background: #0f3460; border-radius: 4px; padding: 10px; margin-bottom: 8px;
        }
        .chunk-header { 
            display: flex; justify-content: space-between; margin-bottom: 6px;
            font-size: 12px; color: #888;
        }
        .chunk-content {
            font-family: monospace; font-size: 12px; color: #ccc;
            white-space: pre-wrap; overflow: hidden; max-height: 60px;
        }
        .chunk-content.expanded { max-height: none; }
        
        /* Summary section */
        .summary-section {
            background: #0f3460; border-radius: 6px; padding: 15px;
            margin-bottom: 15px; border-left: 3px solid #00ff88;
        }
        .summary-title { color: #00ff88; font-size: 13px; margin-bottom: 8px; }
        .summary-content { font-size: 13px; color: #ccc; white-space: pre-wrap; }
        
        /* Phase 2: Collapsible sections */
        .collapsible-section {
            margin-top: 6px;
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 6px;
        }
        .collapsible-header {
            display: flex;
            align-items: center;
            cursor: pointer;
            color: #00ff88;
            font-size: 13px;
            user-select: none;
        }
        .collapsible-header:hover { color: #00ffaa; }
        .collapsible-arrow {
            margin-right: 6px;
            transition: transform 0.2s;
            font-size: 10px;
        }
        .collapsible-header.expanded .collapsible-arrow {
            transform: rotate(90deg);
        }
        .collapsible-body {
            display: none;
            margin-top: 4px;
            padding: 6px;
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
            font-size: 13px;
            color: #ccc;
            white-space: pre-wrap;
        }
        .collapsible-body.show { display: block; }
        
        /* Phase 2: Mechanism tags */
        .mechanism-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }
        .mechanism-tag {
            background: #1a4a7a;
            color: #88ccff;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        
        /* Phase 2: Method summaries list */
        .method-summaries-list {
            margin-top: 8px;
            font-size: 12px;
        }
        .method-item {
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex;
            gap: 8px;
        }
        .method-item:last-child { border-bottom: none; }
        .method-name {
            color: #ffcc00;
            font-family: monospace;
            font-weight: 500;
            flex-shrink: 0;
        }
        .method-desc {
            color: #aaa;
        }
        
        /* Summary Statistics Section */
        .summary-stats-container {
            margin-bottom: 15px;
            flex-shrink: 0;
        }
        .summary-stats-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: #16213e;
            border: 1px solid #0f3460;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            user-select: none;
        }
        .summary-stats-header:hover {
            background: #1a2a4e;
        }
        .summary-stats-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #00d4ff;
        }
        .summary-stats-arrow {
            transition: transform 0.2s;
            font-size: 10px;
        }
        .summary-stats-header.collapsed .summary-stats-arrow {
            transform: rotate(-90deg);
        }
        .summary-stats-total {
            font-size: 12px;
            color: #888;
        }
        .summary-stats-body {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            padding: 15px;
            background: #0f3460;
            border: 1px solid #0f3460;
            border-top: none;
            border-radius: 0 0 8px 8px;
        }
        .summary-stats-body.hidden {
            display: none;
        }
        .stats-column {
            background: #16213e;
            border-radius: 6px;
            padding: 12px;
        }
        .stats-column-title {
            font-size: 12px;
            color: #888;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #0f3460;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stats-list {
            max-height: 180px;
            overflow-y: auto;
        }
        .stats-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 8px;
            margin-bottom: 4px;
            border-radius: 4px;
            font-size: 13px;
        }
        .stats-item:hover {
            background: rgba(0, 212, 255, 0.1);
        }
        .stats-item-label {
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 120px;
        }
        .stats-item-count {
            color: #00d4ff;
            font-weight: 500;
            flex-shrink: 0;
            margin-left: 8px;
        }
        .stats-item.status-approved .stats-item-label { color: #00ff88; }
        .stats-item.status-rejected .stats-item-label { color: #ff6b6b; }
        .stats-item.status-unreviewed .stats-item-label { color: #888; }
        .stats-item.status-approved .stats-item-count { color: #00ff88; }
        .stats-item.status-rejected .stats-item-count { color: #ff6b6b; }
        .stats-loading {
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 12px;
        }
        @media (max-width: 900px) {
            .summary-stats-body {
                grid-template-columns: 1fr;
            }
            .stats-list {
                max-height: 120px;
            }
        }
        
        /* Loading & empty states */
        .loading { text-align: center; padding: 40px; color: #888; }
        .empty-state { text-align: center; padding: 40px; color: #888; }
        
        /* Refresh note */
        .refresh-note { text-align: center; color: #555; font-size: 12px; margin-top: 20px; }
        
        /* Responsive Design */
        @media (max-width: 1200px) {
            .container { padding: 20px; }
            .browse-split-container {
                grid-template-columns: 35% 65%;
            }
            .validate-split-container {
                grid-template-columns: 30% 1fr 1fr;
            }
        }
        @media (max-width: 900px) {
            .browse-split-container {
                grid-template-columns: 1fr;
                min-height: auto;
            }
            .validate-split-container {
                grid-template-columns: 1fr;
                grid-template-rows: auto 1fr 1fr;
                min-height: auto;
            }
            .browse-list-panel,
            .browse-detail-panel {
                min-height: 300px;
            }
            .validate-file-list {
                max-height: 200px;
            }
            .validate-panel {
                min-height: 250px;
            }
            .validate-actions {
                flex-wrap: wrap;
                gap: 10px;
            }
            .validate-action-group {
                width: 100%;
                justify-content: center;
            }
        }
        @media (max-width: 600px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
            .tabs {
                width: 100%;
                justify-content: space-between;
            }
            .tab {
                padding: 6px 10px;
                font-size: 12px;
            }
            .browse-controls,
            .browse-quick-filters,
            .validate-filter-bar {
                flex-direction: column;
                align-items: stretch;
            }
            .browse-controls label,
            .browse-quick-filters label,
            .validate-controls label {
                width: 100%;
            }
            .browse-controls select,
            .browse-quick-filters select,
            .validate-controls select,
            .filter-input {
                width: 100%;
            }
            .search-type-filters {
                flex-wrap: wrap;
            }
            .result-item-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
            }
        }
        
        /* Keyboard shortcut hints */
        .keyboard-hint {
            font-size: 10px;
            color: #555;
            background: #0f3460;
            padding: 2px 6px;
            border-radius: 3px;
            margin-left: 5px;
        }
        
        /* Action buttons card */
        .action-buttons {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .action-btn {
            padding: 12px 20px;
            border: 1px solid #1a3a6e;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            background: #0f3460;
            color: #eee;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .action-btn:hover:not(:disabled) {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
        }
        .action-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .action-btn.loading {
            pointer-events: none;
        }
        .action-btn .spinner {
            display: none;
            width: 14px;
            height: 14px;
            border: 2px solid #555;
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        .action-btn.loading .spinner {
            display: inline-block;
        }
        .action-btn.loading .btn-icon {
            display: none;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Modal styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.open {
            display: flex;
        }
        .modal {
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
            padding: 20px;
            min-width: 350px;
            max-width: 450px;
        }
        .modal-title {
            font-size: 16px;
            color: #eee;
            margin-bottom: 15px;
        }
        .modal-body {
            margin-bottom: 20px;
        }
        .modal-body p {
            color: #888;
            font-size: 13px;
            margin-bottom: 12px;
        }
        .modal-body select {
            width: 100%;
            padding: 10px 12px;
            background: #0f3460;
            border: 1px solid #1a3a6e;
            border-radius: 4px;
            color: #eee;
            font-size: 14px;
        }
        .modal-body select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .modal-body label {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ccc;
            font-size: 13px;
            margin-top: 12px;
            cursor: pointer;
        }
        .modal-body label input[type="checkbox"] {
            width: 16px;
            height: 16px;
        }
        .modal-footer {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        .modal-btn {
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .modal-btn-cancel {
            background: transparent;
            border: 1px solid #1a3a6e;
            color: #888;
        }
        .modal-btn-cancel:hover {
            border-color: #555;
            color: #ccc;
        }
        .modal-btn-confirm {
            background: #00d4ff;
            border: none;
            color: #1a1a2e;
            font-weight: 500;
        }
        .modal-btn-confirm:hover {
            background: #00b8e6;
        }
        .modal-btn-confirm:disabled {
            background: #555;
            cursor: not-allowed;
        }
        .modal-btn-danger {
            background: #ff6b6b;
            border: none;
            color: #fff;
            font-weight: 500;
        }
        .modal-btn-danger:hover {
            background: #ff5252;
        }
        
        /* Toast notifications */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1100;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .toast {
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 13px;
            animation: slideIn 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .toast-success {
            background: rgba(0, 255, 136, 0.15);
            border: 1px solid rgba(0, 255, 136, 0.3);
            color: #00ff88;
        }
        .toast-error {
            background: rgba(255, 107, 107, 0.15);
            border: 1px solid rgba(255, 107, 107, 0.3);
            color: #ff6b6b;
        }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Conductor Memory</h1>
        <div class="tabs">
            <button class="tab active" data-tab="status">Status</button>
            <button class="tab" data-tab="search">Search</button>
            <button class="tab" data-tab="browse">Browse</button>
            <button class="tab" data-tab="summaries">Summaries</button>
        </div>
    </div>
    
    <div class="container">
        <!-- STATUS TAB -->
        <div id="status-tab" class="tab-content active">
            <div id="status-loading" class="loading">Loading...</div>
            <div id="status-content" style="display: none;">
                <h2>Indexing Status</h2>
                <div class="card">
                    <div class="stats" id="indexing-stats"></div>
                </div>
                
                <h2>Summarization Status</h2>
                <div class="card">
                    <div class="stats" id="summary-stats"></div>
                    <div id="progress-section"></div>
                    <div id="current-file-section"></div>
                    <div id="error-section"></div>
                </div>
                
                <h2>Codebases</h2>
                <div class="card" id="codebases"></div>
                
                <h2>Actions</h2>
                <div class="card">
                    <div class="action-buttons">
                        <button class="action-btn" id="btn-queue-summarization" onclick="openQueueSummarizationModal()">
                            <span class="btn-icon">📋</span>
                            <span class="spinner"></span>
                            Queue Summarization
                        </button>
                        <button class="action-btn" id="btn-invalidate-summaries" onclick="openInvalidateSummariesModal()">
                            <span class="btn-icon">🗑️</span>
                            <span class="spinner"></span>
                            Invalidate Summaries
                        </button>
                        <button class="action-btn" id="btn-reindex-codebase" onclick="openReindexCodebaseModal()">
                            <span class="btn-icon">🔄</span>
                            <span class="spinner"></span>
                            Reindex Codebase
                        </button>
                    </div>
                </div>
            </div>
            <p class="refresh-note">Auto-refreshes every 5 seconds</p>
        </div>
        
        <!-- SEARCH TAB -->
        <div id="search-tab" class="tab-content">
            <h2>Search Memory</h2>
            <div class="card">
                <div class="search-form">
                    <input type="text" id="search-query" class="search-input" placeholder="Enter search query..." />
                    <button id="search-btn" class="search-btn">Search</button>
                </div>
                
                <!-- Memory Type Filters (Always Visible) -->
                <div class="search-type-filters">
                    <span class="type-filter-label">Types:</span>
                    <label class="type-checkbox type-code">
                        <input type="checkbox" id="filter-type-code" checked />
                        <span class="type-badge">Code</span>
                    </label>
                    <label class="type-checkbox type-decision">
                        <input type="checkbox" id="filter-type-decision" checked />
                        <span class="type-badge">Decisions</span>
                    </label>
                    <label class="type-checkbox type-lesson">
                        <input type="checkbox" id="filter-type-lesson" checked />
                        <span class="type-badge">Lessons</span>
                    </label>
                    <label class="type-checkbox type-conversation">
                        <input type="checkbox" id="filter-type-conversation" />
                        <span class="type-badge">Conversations</span>
                    </label>
                </div>
                
                <div class="search-options">
                    <label>Codebase:
                        <select id="search-codebase">
                            <option value="">All</option>
                        </select>
                    </label>
                    <label>Mode:
                        <select id="search-mode">
                            <option value="auto">Auto</option>
                            <option value="semantic">Semantic</option>
                            <option value="keyword">Keyword</option>
                            <option value="hybrid">Hybrid</option>
                        </select>
                    </label>
                    <label>Results:
                        <select id="search-limit">
                            <option value="5">5</option>
                            <option value="10" selected>10</option>
                            <option value="25">25</option>
                            <option value="50">50</option>
                        </select>
                    </label>
                    <span class="advanced-toggle" onclick="toggleAdvanced()">
                        <span id="advanced-arrow">&#9654;</span> More Filters
                    </span>
                </div>
                
                <div id="advanced-filters" class="advanced-filters">
                    <div class="filter-row">
                        <div class="filter-group">
                            <label>Languages (comma-sep)</label>
                            <input type="text" id="filter-languages" placeholder="python, java" />
                        </div>
                        <div class="filter-group">
                            <label>Include Tags</label>
                            <input type="text" id="filter-include-tags" placeholder="api, core" />
                        </div>
                        <div class="filter-group">
                            <label>Exclude Tags</label>
                            <input type="text" id="filter-exclude-tags" placeholder="test" />
                        </div>
                    </div>
                    <div class="filter-row">
                        <div class="filter-group">
                            <label>Class Names</label>
                            <input type="text" id="filter-classes" placeholder="UserService" />
                        </div>
                        <div class="filter-group">
                            <label>Function Names</label>
                            <input type="text" id="filter-functions" placeholder="authenticate" />
                        </div>
                    </div>
                    <div class="filter-row">
                        <div class="checkbox-group">
                            <input type="checkbox" id="filter-has-docstrings" />
                            <label for="filter-has-docstrings">Has Docstrings</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="filter-has-annotations" />
                            <label for="filter-has-annotations">Has Annotations</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="filter-include-summaries" />
                            <label for="filter-include-summaries">Include Summaries</label>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="search-results">
                <div class="empty-state">Enter a query and click Search</div>
            </div>
        </div>
        
        <!-- BROWSE TAB -->
        <div id="browse-tab" class="tab-content">
            <h2>Browse Data</h2>
            <div class="browse-controls">
                <label>View:
                    <select id="browse-view">
                        <option value="files">Indexed Files</option>
                        <option value="memories">Memories</option>
                        <option value="summaries">Summaries</option>
                    </select>
                </label>
                <label>Codebase:
                    <select id="browse-codebase"></select>
                </label>
                <input type="text" id="browse-filter" class="filter-input" placeholder="Filter by path..." />
            </div>
            
            <!-- Quick Filters (visible only for files view) -->
            <div class="browse-quick-filters" id="browse-quick-filters">
                <label>Summary:
                    <select id="filter-has-summary">
                        <option value="">All</option>
                        <option value="yes">Has Summary</option>
                        <option value="no">No Summary</option>
                    </select>
                </label>
                <label>Pattern:
                    <select id="filter-pattern">
                        <option value="">All Patterns</option>
                        <option value="service">service</option>
                        <option value="utility">utility</option>
                        <option value="model">model</option>
                        <option value="config">config</option>
                        <option value="controller">controller</option>
                        <option value="repository">repository</option>
                        <option value="test">test</option>
                    </select>
                </label>
                <label>Domain:
                    <select id="filter-domain">
                        <option value="">All Domains</option>
                        <option value="api">api</option>
                        <option value="database">database</option>
                        <option value="auth">auth</option>
                        <option value="search">search</option>
                        <option value="config">config</option>
                        <option value="core">core</option>
                        <option value="ui">ui</option>
                    </select>
                </label>
                <label>Language:
                    <select id="filter-language">
                        <option value="">All Languages</option>
                        <option value="python">Python</option>
                        <option value="kotlin">Kotlin</option>
                        <option value="java">Java</option>
                        <option value="typescript">TypeScript</option>
                        <option value="javascript">JavaScript</option>
                    </select>
                </label>
                <span class="filter-reset-btn" id="filter-reset-btn" onclick="resetQuickFilters()" style="display: none;">✕ Clear filters</span>
            </div>
            
            <!-- Master-Detail Split Layout -->
            <div class="browse-split-container">
                <!-- Left Panel: File List -->
                <div class="browse-list-panel">
                    <div class="browse-list-header">
                        <span id="browse-list-count">Loading...</span>
                        <div class="browse-view-toggle">
                            <button class="view-toggle-btn active" id="btn-tree-view" onclick="setBrowseViewMode('tree')" title="Tree view">🌲</button>
                            <button class="view-toggle-btn" id="btn-list-view" onclick="setBrowseViewMode('list')" title="List view">☰</button>
                        </div>
                    </div>
                    <div class="browse-list-content" id="browse-list-content" tabindex="0">
                        <div class="loading" style="padding: 40px; text-align: center;">Loading...</div>
                    </div>
                    <div class="browse-list-footer" id="browse-pagination"></div>
                </div>
                
                <!-- Right Panel: Details -->
                <div class="browse-detail-panel">
                    <div class="browse-detail-header">
                        <span class="browse-detail-title" id="detail-title">Select a file to view details</span>
                    </div>
                    <div class="browse-detail-content" id="detail-content">
                        <div class="browse-detail-empty">
                            <span>← Select an item from the list</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- SUMMARIES TAB -->
        <div id="summaries-tab" class="tab-content">
            <h2>Summary Validation</h2>
            
            <!-- Summary Statistics Section (collapsible) -->
            <div class="summary-stats-container">
                <div class="summary-stats-header" onclick="toggleSummaryStats()">
                    <div class="summary-stats-title">
                        <span class="summary-stats-arrow">▼</span>
                        Summary Statistics
                    </div>
                    <span class="summary-stats-total" id="summary-stats-total">Loading...</span>
                </div>
                <div class="summary-stats-body" id="summary-stats-body">
                    <div class="stats-loading" id="summary-stats-loading">Loading statistics...</div>
                </div>
            </div>
            
            <!-- Filter Bar -->
            <div class="validate-filter-bar">
                <label>Codebase:
                    <select id="validate-codebase"></select>
                </label>
                <label>Status:
                    <select id="validate-status">
                        <option value="unreviewed">Unreviewed</option>
                        <option value="all">All</option>
                        <option value="approved">Approved</option>
                        <option value="rejected">Rejected</option>
                    </select>
                </label>
                <label>Pattern:
                    <select id="validate-pattern">
                        <option value="">All Patterns</option>
                    </select>
                </label>
                <label>Domain:
                    <select id="validate-domain">
                        <option value="">All Domains</option>
                    </select>
                </label>
                <label>Sort:
                    <select id="validate-sort">
                        <option value="recent">Most Recent</option>
                        <option value="alpha">A-Z</option>
                        <option value="alpha-desc">Z-A</option>
                    </select>
                </label>
                <label>Per page:
                    <select id="validate-page-size">
                        <option value="20">20</option>
                        <option value="30">30</option>
                        <option value="40">40</option>
                        <option value="50">50</option>
                    </select>
                </label>
                <button class="validate-refresh-btn" id="validate-refresh-btn" onclick="loadValidateFiles()" title="Refresh list">↻</button>
                <span class="validate-filter-reset" id="validate-filter-reset" style="display: none;">✕ Clear filters</span>
                <span class="validate-progress" id="validate-progress">Loading...</span>
            </div>
            
            <!-- 3-Column Layout -->
            <div class="validate-split-container">
                <!-- Left: File List -->
                <div class="validate-file-list">
                    <div class="validate-file-list-header">
                        <span id="validate-list-count">Loading...</span>
                        <span>↑↓ nav</span>
                    </div>
                    <div class="validate-file-list-content" id="validate-list-content" tabindex="0">
                        <div class="loading" style="padding: 20px;">Loading...</div>
                    </div>
                    <div class="validate-file-list-footer" id="validate-pagination"></div>
                </div>
                
                <!-- Middle: Original File Content -->
                <div class="validate-panel validate-source">
                    <div class="validate-panel-header">
                        <span class="validate-panel-title">Original File</span>
                        <span class="validate-file-path" id="validate-file-path">Select a file...</span>
                    </div>
                    <div class="validate-panel-content" id="validate-source-content">
                        <div class="browse-detail-empty">
                            <span>Select a file from the list</span>
                        </div>
                    </div>
                </div>
                
                <!-- Right: LLM Summary -->
                <div class="validate-panel validate-summary">
                    <div class="validate-panel-header">
                        <span class="validate-panel-title">LLM Summary</span>
                        <span class="validate-model" id="validate-model"></span>
                    </div>
                    <div class="validate-panel-content" id="validate-summary-content">
                        <div class="browse-detail-empty">
                            <span>Summary will appear here</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Action Bar -->
            <div class="validate-actions">
                <button class="validate-btn validate-btn-prev" id="validate-prev" onclick="validatePrev()">← Prev <span class="keyboard-hint">←</span></button>
                <div class="validate-action-group">
                    <button class="validate-btn validate-btn-approve" onclick="validateApprove()">✓ Approve <span class="keyboard-hint">A</span></button>
                    <button class="validate-btn validate-btn-reject" onclick="validateReject()">✗ Reject <span class="keyboard-hint">R</span></button>
                    <button class="validate-btn validate-btn-regenerate" onclick="validateRegenerate()">↻ Regenerate</button>
                    <button class="validate-btn validate-btn-skip" onclick="validateSkip()">Skip <span class="keyboard-hint">S</span></button>
                </div>
                <button class="validate-btn validate-btn-next" id="validate-next" onclick="validateNext()">Next → <span class="keyboard-hint">→</span></button>
            </div>
        </div>
    </div>
    
    <!-- Toast container for notifications -->
    <div class="toast-container" id="toast-container"></div>
    
    <!-- Modal: Queue Summarization -->
    <div class="modal-overlay" id="modal-queue-summarization">
        <div class="modal">
            <div class="modal-title">Queue Summarization</div>
            <div class="modal-body">
                <p>Queue files from a codebase for LLM summarization.</p>
                <select id="queue-codebase-select"></select>
                <label>
                    <input type="checkbox" id="queue-only-missing" checked />
                    Only queue files without summaries
                </label>
            </div>
            <div class="modal-footer">
                <button class="modal-btn modal-btn-cancel" onclick="closeModal('modal-queue-summarization')">Cancel</button>
                <button class="modal-btn modal-btn-confirm" id="queue-confirm-btn" onclick="confirmQueueSummarization()">Queue</button>
            </div>
        </div>
    </div>
    
    <!-- Modal: Invalidate Summaries -->
    <div class="modal-overlay" id="modal-invalidate-summaries">
        <div class="modal">
            <div class="modal-title">Invalidate All Summaries</div>
            <div class="modal-body">
                <p>This will clear all existing summaries from all codebases. Files will need to be re-summarized.</p>
                <p style="color: #ff6b6b;"><strong>Warning:</strong> This action cannot be undone.</p>
            </div>
            <div class="modal-footer">
                <button class="modal-btn modal-btn-cancel" onclick="closeModal('modal-invalidate-summaries')">Cancel</button>
                <button class="modal-btn modal-btn-danger" id="invalidate-confirm-btn" onclick="confirmInvalidateSummaries()">Invalidate All</button>
            </div>
        </div>
    </div>
    
    <!-- Modal: Reindex Codebase -->
    <div class="modal-overlay" id="modal-reindex-codebase">
        <div class="modal">
            <div class="modal-title">Reindex Codebase</div>
            <div class="modal-body">
                <p>Force reindexing of a codebase to update metadata and heuristics.</p>
                <select id="reindex-codebase-select"></select>
            </div>
            <div class="modal-footer">
                <button class="modal-btn modal-btn-cancel" onclick="closeModal('modal-reindex-codebase')">Cancel</button>
                <button class="modal-btn modal-btn-confirm" id="reindex-confirm-btn" onclick="confirmReindexCodebase()">Reindex</button>
            </div>
        </div>
    </div>
    
    <script>
        // =========== State ===========
        let codebases = [];
        let currentBrowseOffset = 0;
        const BROWSE_LIMIT = 20;
        let statusInterval = null;
        
        // =========== Tab Management ===========
        const validTabs = ['status', 'search', 'browse', 'summaries'];
        
        function switchToTab(tabName) {
            // Validate tab name, default to 'status'
            if (!validTabs.includes(tabName)) {
                tabName = 'status';
            }
            
            // Update URL hash without triggering hashchange (if already correct)
            if (window.location.hash !== '#' + tabName) {
                history.replaceState(null, '', '#' + tabName);
            }
            
            // Update tab UI
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            const tabButton = document.querySelector(`.tab[data-tab="${tabName}"]`);
            if (tabButton) {
                tabButton.classList.add('active');
            }
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Start/stop status refresh based on active tab
            if (tabName === 'status') {
                startStatusRefresh();
            } else {
                stopStatusRefresh();
            }
            
            // Load data for browse tab
            if (tabName === 'browse') {
                loadBrowseData();
            }
            
            // Load data for summaries tab
            if (tabName === 'summaries') {
                loadSummaryStats();
                loadValidationQueue();
            }
        }
        
        // Tab click handlers
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                switchToTab(tab.dataset.tab);
            });
        });
        
        // Handle browser back/forward navigation
        window.addEventListener('hashchange', () => {
            const hash = window.location.hash.slice(1); // Remove '#'
            switchToTab(hash || 'status');
        });
        
        // Initialize tab from URL hash on page load
        function initTabFromHash() {
            const hash = window.location.hash.slice(1); // Remove '#'
            switchToTab(hash || 'status');
        }
        
        // =========== Status Tab ===========
        async function fetchStatus() {
            try {
                const [indexRes, summaryRes] = await Promise.all([
                    fetch('/api/status'),
                    fetch('/api/summarization')
                ]);
                const indexData = await indexRes.json();
                const summaryData = await summaryRes.json();
                
                document.getElementById('status-loading').style.display = 'none';
                document.getElementById('status-content').style.display = 'block';
                
                updateIndexingStats(indexData);
                updateSummaryStats(summaryData);
                updateCodebases(indexData, summaryData);
            } catch (err) {
                document.getElementById('status-loading').textContent = 'Error: ' + err.message;
            }
        }
        
        function updateIndexingStats(data) {
            const totalFiles = Object.values(data.codebases || {})
                .reduce((sum, cb) => sum + (cb.indexed_files_count || 0), 0);
            const numCodebases = Object.keys(data.codebases || {}).length;
            
            document.getElementById('indexing-stats').innerHTML = `
                <div class="stat"><div class="stat-value">${numCodebases}</div><div class="stat-label">Codebases</div></div>
                <div class="stat"><div class="stat-value">${totalFiles.toLocaleString()}</div><div class="stat-label">Indexed Files</div></div>
                <div class="stat"><div class="stat-value">${data.status || 'unknown'}</div><div class="stat-label">Status</div></div>
            `;
        }
        
        function updateSummaryStats(data) {
            const statusClass = data.is_running ? 'status-running' : 'status-idle';
            const statusText = data.is_running ? 'Running' : 'Idle';
            
            // Format timing information
            const avgTime = data.avg_time_per_file_seconds || 0;
            const avgTimeText = avgTime > 0 ? `${avgTime.toFixed(1)}s` : 'N/A';
            
            const estRemaining = data.estimated_time_remaining_minutes || 0;
            let estRemainingText = 'N/A';
            if (estRemaining > 0 && data.files_queued > 0) {
                if (estRemaining < 1) {
                    estRemainingText = `${Math.round(estRemaining * 60)}s`;
                } else if (estRemaining < 60) {
                    estRemainingText = `${estRemaining.toFixed(1)}min`;
                } else {
                    const hours = Math.floor(estRemaining / 60);
                    const mins = Math.round(estRemaining % 60);
                    estRemainingText = `${hours}h ${mins}m`;
                }
            }
            
            // Build stats - only show Failed if > 0
            const failedStat = (data.files_failed || 0) > 0 
                ? `<div class="stat"><div class="stat-value status-error">${data.files_failed}</div><div class="stat-label">Failed</div></div>`
                : '';
            
            // Only show timing stats if queue is active
            const timingStats = data.files_queued > 0 
                ? `<div class="stat"><div class="stat-value">${avgTimeText}</div><div class="stat-label">Avg Time/File</div></div>
                   <div class="stat"><div class="stat-value">${estRemainingText}</div><div class="stat-label">Est. Remaining</div></div>`
                : '';
            
            document.getElementById('summary-stats').innerHTML = `
                <div class="stat"><div class="stat-value ${statusClass}">${statusText}</div><div class="stat-label">Status</div></div>
                <div class="stat"><div class="stat-value">${data.total_summarized || 0}</div><div class="stat-label">Files Summarized</div></div>
                <div class="stat"><div class="stat-value">${data.files_queued || 0}</div><div class="stat-label">Files Pending</div></div>
                <div class="stat"><div class="stat-value">${data.simple_count || 0}</div><div class="stat-label">Simple Files</div></div>
                <div class="stat"><div class="stat-value">${data.llm_count || 0}</div><div class="stat-label">LLM Summarized</div></div>
                ${failedStat}
                ${timingStats}
            `;
            
            const total = data.files_total_queued || 0;
            const processed = data.files_processed || 0;
            const pct = data.progress_percentage || 0;
            const queued = data.files_queued || 0;
            
            // Determine what to show in progress section
            let progressHtml = '';
            if (data.is_running && queued > 0 && total > 0) {
                // Active processing: show progress bar
                progressHtml = `
                    <div style="display: flex; justify-content: space-between; font-size: 13px; color: #888; margin-top: 10px;">
                        <span>Progress (${processed}/${total})</span><span>${Math.round(pct)}% • ${estRemainingText} remaining</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" style="width: ${pct}%"></div></div>
                `;
            } else if (processed > 0 && queued === 0) {
                // Queue empty but work was done this session: show completion
                progressHtml = `
                    <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: #4ade80; margin-top: 10px;">
                        <span style="font-size: 16px;">✓</span>
                        <span>${processed} Items Summarized</span>
                    </div>
                `;
            }
            document.getElementById('progress-section').innerHTML = progressHtml;
            
            document.getElementById('current-file-section').innerHTML = data.current_file 
                ? `<div class="current-file">${data.current_file}</div>` : '';
            
            document.getElementById('error-section').innerHTML = data.last_error 
                ? `<div class="error-msg">${data.last_error}</div>` : '';
        }
        
        function updateCodebases(indexData, summaryData) {
            const byCodebase = summaryData.by_codebase || {};
            let html = '';
            
            for (const [name, info] of Object.entries(indexData.codebases || {})) {
                const summaryInfo = byCodebase[name] || {};
                const summarized = summaryInfo.total_summarized || 0;
                const indexed = info.indexed_files_count || 0;
                
                html += `
                    <div class="codebase">
                        <div>
                            <div class="codebase-name">${name}</div>
                            <div class="codebase-stats">${info.status}</div>
                        </div>
                        <div style="text-align: right;">
                            <div>${indexed.toLocaleString()} files indexed</div>
                            <div class="codebase-stats">${summarized} summarized</div>
                        </div>
                    </div>
                `;
            }
            document.getElementById('codebases').innerHTML = html || '<p style="color: #888;">No codebases configured</p>';
        }
        
        function startStatusRefresh() {
            if (!statusInterval) {
                fetchStatus();
                statusInterval = setInterval(fetchStatus, 5000);
            }
        }
        
        function stopStatusRefresh() {
            if (statusInterval) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
        }
        
        // =========== Codebases Loader ===========
        async function loadCodebases() {
            try {
                const res = await fetch('/api/codebases');
                const data = await res.json();
                codebases = data.codebases || [];
                
                // Populate dropdowns
                const searchSelect = document.getElementById('search-codebase');
                const browseSelect = document.getElementById('browse-codebase');
                const validateSelect = document.getElementById('validate-codebase');
                
                searchSelect.innerHTML = '<option value="">All</option>';
                browseSelect.innerHTML = '';
                validateSelect.innerHTML = '';
                
                codebases.forEach(cb => {
                    searchSelect.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
                    browseSelect.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
                    validateSelect.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
                });
                
                return codebases;
            } catch (err) {
                console.error('Failed to load codebases:', err);
                return [];
            }
        }
        
        // =========== Search Tab ===========
        function toggleAdvanced() {
            const filters = document.getElementById('advanced-filters');
            const arrow = document.getElementById('advanced-arrow');
            filters.classList.toggle('open');
            arrow.innerHTML = filters.classList.contains('open') ? '&#9660;' : '&#9654;';
        }
        
        async function performSearch() {
            const query = document.getElementById('search-query').value.trim();
            if (!query) return;
            
            const btn = document.getElementById('search-btn');
            btn.disabled = true;
            btn.textContent = 'Searching...';
            
            const resultsDiv = document.getElementById('search-results');
            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';
            
            // Gather filter values
            const parseList = (val) => val ? val.split(',').map(s => s.trim()).filter(s => s) : null;
            
            // Get type filters
            const includeCode = document.getElementById('filter-type-code').checked;
            const includeDecision = document.getElementById('filter-type-decision').checked;
            const includeLesson = document.getElementById('filter-type-lesson').checked;
            const includeConversation = document.getElementById('filter-type-conversation').checked;
            
            const body = {
                query,
                codebase: document.getElementById('search-codebase').value || null,
                max_results: parseInt(document.getElementById('search-limit').value) * 2,  // Request more to account for filtering
                search_mode: document.getElementById('search-mode').value,
                languages: parseList(document.getElementById('filter-languages').value),
                include_tags: parseList(document.getElementById('filter-include-tags').value),
                exclude_tags: parseList(document.getElementById('filter-exclude-tags').value),
                class_names: parseList(document.getElementById('filter-classes').value),
                function_names: parseList(document.getElementById('filter-functions').value),
                has_docstrings: document.getElementById('filter-has-docstrings').checked || null,
                has_annotations: document.getElementById('filter-has-annotations').checked || null,
                include_summaries: document.getElementById('filter-include-summaries').checked
            };
            
            // Remove null values
            Object.keys(body).forEach(k => body[k] === null && delete body[k]);
            
            try {
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                
                if (data.error) {
                    resultsDiv.innerHTML = `<div class="error-msg">${data.error}</div>`;
                } else {
                    // Apply type filtering client-side
                    const results = data.results || [];
                    const filteredResults = results.filter(r => {
                        const memoryType = r.memory_type || 'code';
                        if (memoryType === 'code' || memoryType === 'file_chunk') return includeCode;
                        if (memoryType === 'decision') return includeDecision;
                        if (memoryType === 'lesson') return includeLesson;
                        if (memoryType === 'conversation') return includeConversation;
                        return includeCode;  // Default to code for unknown types
                    });
                    
                    // Limit to requested amount
                    const limit = parseInt(document.getElementById('search-limit').value);
                    data.results = filteredResults.slice(0, limit);
                    data.total_found = filteredResults.length;
                    
                    renderSearchResults(data);
                }
            } catch (err) {
                resultsDiv.innerHTML = `<div class="error-msg">Error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Search';
            }
        }
        
        function renderSummaryPreview(summary) {
            if (!summary) return '';
            
            // Truncate purpose to ~100 chars
            let purpose = summary.purpose || '';
            if (purpose.length > 100) {
                purpose = purpose.substring(0, 100).trim() + '...';
            }
            
            // Build badges
            let badges = '';
            if (summary.pattern) {
                badges += `<span class="pattern-badge">${escapeHtml(summary.pattern)}</span>`;
            }
            if (summary.domain) {
                badges += `<span class="domain-badge">${escapeHtml(summary.domain)}</span>`;
            }
            
            // Build meta indicators
            let meta = [];
            if (summary.has_how_it_works) {
                meta.push('<span class="has-indicator">How It Works</span>');
            }
            if (summary.has_method_summaries) {
                const methodCount = Array.isArray(summary.method_summaries) ? summary.method_summaries.length : 0;
                meta.push(`<span class="has-indicator">${methodCount || 'Has'} Methods</span>`);
            }
            
            return `
                <div class="result-summary-preview">
                    <div class="result-summary-header">
                        <div class="result-summary-purpose">${escapeHtml(purpose)}</div>
                        ${badges ? `<div class="result-summary-badges">${badges}</div>` : ''}
                    </div>
                    ${meta.length > 0 ? `<div class="result-summary-meta">${meta.join('')}</div>` : ''}
                </div>
            `;
        }
        
        function renderSearchResults(data) {
            const resultsDiv = document.getElementById('search-results');
            const results = data.results || [];
            const includeSummaries = document.getElementById('filter-include-summaries').checked;
            
            if (results.length === 0) {
                resultsDiv.innerHTML = '<div class="empty-state">No results found. Try different search terms or adjust filters.</div>';
                return;
            }
            
            // Build active filters summary
            let activeFilters = [];
            if (!document.getElementById('filter-type-code').checked) activeFilters.push('-Code');
            if (!document.getElementById('filter-type-decision').checked) activeFilters.push('-Decisions');
            if (!document.getElementById('filter-type-lesson').checked) activeFilters.push('-Lessons');
            if (!document.getElementById('filter-type-conversation').checked) activeFilters.push('-Conversations');
            const filterNote = activeFilters.length > 0 ? ` | Filters: ${activeFilters.join(', ')}` : '';
            
            let html = `<div class="results-header">
                <span>Found ${data.total_found || results.length} results (${data.query_time_ms}ms) • Mode: ${data.search_mode_used}${filterNote}</span>
            </div>`;
            
            results.forEach((r, idx) => {
                const tags = (r.tags || []).slice(0, 5).map(t => `<span class="tag">${t}</span>`).join('');
                const content = escapeHtml(r.content || r.doc_text || '');
                const preview = content.length > 300 ? content.substring(0, 300) : content;
                const hasMore = content.length > 300;
                
                // Determine result type
                const memoryType = r.memory_type || 'code';
                const typeClass = memoryType === 'decision' ? 'decision' : 
                                 memoryType === 'lesson' ? 'lesson' :
                                 memoryType === 'conversation' ? 'conversation' : 'code';
                const typeLabel = memoryType === 'code' || memoryType === 'file_chunk' ? 'CODE' : memoryType.toUpperCase();
                const isPinned = r.pinned || memoryType === 'decision' || memoryType === 'lesson';
                
                // Calculate score segments (5 segments max)
                const score = r.relevance_score || 0;
                const filledSegments = Math.round(score * 5);
                const isHighScore = score >= 0.8;
                let scoreSegments = '';
                for (let i = 0; i < 5; i++) {
                    const filled = i < filledSegments;
                    const segClass = filled ? (isHighScore ? 'filled high' : 'filled') : '';
                    scoreSegments += `<div class="score-segment ${segClass}"></div>`;
                }
                
                // Render summary preview if summaries enabled and available
                const summaryPreview = (includeSummaries && r.has_summary && r.file_summary) 
                    ? renderSummaryPreview(r.file_summary) 
                    : '';
                
                html += `
                    <div class="result-item">
                        <div class="result-item-header">
                            <div class="result-type-badge">
                                <span class="result-type-icon ${typeClass}"></span>
                                <span class="result-type-label ${typeClass}">${typeLabel}</span>
                                ${isPinned ? '<span class="result-pinned">(pinned)</span>' : ''}
                            </div>
                            <div class="result-score-bar">
                                <div class="score-segments">${scoreSegments}</div>
                                <span class="score-value">${score.toFixed(2)}</span>
                            </div>
                        </div>
                        ${r.source ? `<div class="result-file">${r.source}</div>` : ''}
                        ${summaryPreview}
                        <div class="result-body">
                            ${tags ? `<div class="result-tags">${tags}</div>` : ''}
                            <div class="result-content" id="result-content-${idx}">${preview}${hasMore ? '...' : ''}</div>
                            <div class="result-actions">
                                ${hasMore ? `<span class="expand-btn" onclick="toggleResultExpand(${idx}, \\`${escapeJs(content)}\\`)">Show more</span>` : '<span></span>'}
                                <button class="copy-btn" onclick="copyResult(${idx}, \\`${escapeJs(content)}\\`)">Copy</button>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        function copyResult(idx, content) {
            navigator.clipboard.writeText(content).then(() => {
                const btn = document.querySelectorAll('.copy-btn')[idx];
                btn.textContent = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.classList.remove('copied');
                }, 2000);
            });
        }
        
        function toggleResultExpand(idx, fullContent) {
            const el = document.getElementById(`result-content-${idx}`);
            const btn = el.nextElementSibling;
            if (el.classList.contains('expanded')) {
                el.classList.remove('expanded');
                el.innerHTML = fullContent.substring(0, 300) + '...';
                btn.textContent = 'Show more';
            } else {
                el.classList.add('expanded');
                el.innerHTML = fullContent;
                btn.textContent = 'Show less';
            }
        }
        
        // Event listeners for search
        document.getElementById('search-btn').addEventListener('click', performSearch);
        document.getElementById('search-query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });
        
        // =========== Browse Tab ===========
        let browseData = { files: [], memories: [], summaries: [] };
        let selectedIndex = -1;
        let browseViewMode = 'tree';  // 'tree' or 'list'

        function setBrowseViewMode(mode) {
            browseViewMode = mode;
            document.getElementById('btn-tree-view').classList.toggle('active', mode === 'tree');
            document.getElementById('btn-list-view').classList.toggle('active', mode === 'list');
            // Re-render current data
            if (browseData.currentView === 'files' && browseData.files.length > 0) {
                renderFilesList({ files: browseData.files, total: browseData.total || browseData.files.length, offset: browseData.offset || 0 });
            }
        }

        function buildFileTree(files) {
            const root = { children: {}, files: [] };
            files.forEach((file, idx) => {
                const parts = file.path.split(/[\\/\\\\]/);
                let current = root;
                // Navigate to the folder, creating nodes as needed
                for (let i = 0; i < parts.length - 1; i++) {
                    const part = parts[i];
                    if (!current.children[part]) {
                        current.children[part] = { children: {}, files: [] };
                    }
                    current = current.children[part];
                }
                // Add file to the final folder
                current.files.push({ ...file, _idx: idx });
            });
            return root;
        }

        function renderTreeNode(node, name = '', depth = 0) {
            let html = '';
            const nodeId = 'tree-' + Math.random().toString(36).substr(2, 9);

            // Render folders first
            const folders = Object.entries(node.children).sort((a, b) => a[0].localeCompare(b[0]));
            folders.forEach(([folderName, childNode]) => {
                const fileCount = countFilesInNode(childNode);
                const isExpanded = depth < 1;  // Auto-expand first level
                html += `
                    <div class="tree-node">
                        <div class="tree-folder ${isExpanded ? 'expanded' : ''}" onclick="toggleTreeFolder(this)" style="padding-left: ${10 + depth * 12}px;">
                            <span class="tree-folder-icon">▶</span>
                            <span class="tree-folder-name">${escapeHtml(folderName)}</span>
                            <span class="tree-folder-count">${fileCount}</span>
                        </div>
                        <div class="tree-children ${isExpanded ? 'expanded' : ''}">
                            ${renderTreeNode(childNode, folderName, depth + 1)}
                        </div>
                    </div>`;
            });

            // Render files
            node.files.forEach(file => {
                const summaryBadge = getSummaryBadge(file.has_summary, file.simple_file, file.simple_file_reason);
                const fileName = file.path.split(/[\\/\\\\]/).pop();
                html += `
                    <div class="tree-file" data-index="${file._idx}" onclick="selectFileItem(${file._idx})" style="padding-left: ${16 + depth * 12}px;">
                        <span class="tree-file-icon">📄</span>
                        <span>${escapeHtml(fileName)}</span>
                        <div class="tree-file-meta">
                            ${summaryBadge}
                            <span style="font-size: 10px; color: #555;">${file.chunk_count}</span>
                        </div>
                    </div>`;
            });

            return html;
        }

        function countFilesInNode(node) {
            let count = node.files.length;
            Object.values(node.children).forEach(child => {
                count += countFilesInNode(child);
            });
            return count;
        }

        function toggleTreeFolder(folderEl) {
            folderEl.classList.toggle('expanded');
            const children = folderEl.nextElementSibling;
            if (children) {
                children.classList.toggle('expanded');
            }
        }
        
        function updateQuickFiltersVisibility() {
            const view = document.getElementById('browse-view').value;
            const quickFilters = document.getElementById('browse-quick-filters');
            quickFilters.style.display = view === 'files' ? 'flex' : 'none';
        }
        
        function getActiveFiltersCount() {
            let count = 0;
            if (document.getElementById('filter-has-summary').value) count++;
            if (document.getElementById('filter-pattern').value) count++;
            if (document.getElementById('filter-domain').value) count++;
            if (document.getElementById('filter-language').value) count++;
            return count;
        }
        
        function updateFilterIndicators() {
            const resetBtn = document.getElementById('filter-reset-btn');
            const activeCount = getActiveFiltersCount();
            resetBtn.style.display = activeCount > 0 ? 'inline' : 'none';
            
            // Highlight active filters
            ['filter-has-summary', 'filter-pattern', 'filter-domain', 'filter-language'].forEach(id => {
                const el = document.getElementById(id);
                if (el.value) {
                    el.classList.add('filter-active');
                } else {
                    el.classList.remove('filter-active');
                }
            });
        }
        
        function resetQuickFilters() {
            document.getElementById('filter-has-summary').value = '';
            document.getElementById('filter-pattern').value = '';
            document.getElementById('filter-domain').value = '';
            document.getElementById('filter-language').value = '';
            updateFilterIndicators();
            currentBrowseOffset = 0;
            loadBrowseData();
        }
        
        async function loadBrowseData() {
            const view = document.getElementById('browse-view').value;
            const codebase = document.getElementById('browse-codebase').value;
            const filter = document.getElementById('browse-filter').value;
            
            // Update quick filters visibility
            updateQuickFiltersVisibility();
            
            const listContent = document.getElementById('browse-list-content');
            const countSpan = document.getElementById('browse-list-count');
            listContent.innerHTML = '<div class="loading" style="padding: 40px; text-align: center;">Loading...</div>';
            countSpan.textContent = 'Loading...';
            selectedIndex = -1;
            clearDetailPanel();
            
            try {
                let url, data;
                
                if (view === 'files') {
                    url = `/api/files?codebase=${codebase}&limit=${BROWSE_LIMIT}&offset=${currentBrowseOffset}`;
                    if (filter) url += `&filter=${encodeURIComponent(filter)}`;
                    
                    // Add quick filter params
                    const hasSummary = document.getElementById('filter-has-summary').value;
                    const pattern = document.getElementById('filter-pattern').value;
                    const domain = document.getElementById('filter-domain').value;
                    const language = document.getElementById('filter-language').value;
                    
                    if (hasSummary) url += `&has_summary=${hasSummary}`;
                    if (pattern) url += `&pattern=${encodeURIComponent(pattern)}`;
                    if (domain) url += `&domain=${encodeURIComponent(domain)}`;
                    if (language) url += `&language=${encodeURIComponent(language)}`;
                    
                    const res = await fetch(url);
                    data = await res.json();
                    browseData.files = data.files || [];
                    browseData.currentView = 'files';
                    browseData.codebase = data.codebase;
                    renderFilesList(data);
                } else if (view === 'memories') {
                    url = `/api/memories?limit=${BROWSE_LIMIT}&offset=${currentBrowseOffset}`;
                    if (codebase) url += `&codebase=${codebase}`;
                    const res = await fetch(url);
                    data = await res.json();
                    browseData.memories = data.memories || [];
                    browseData.currentView = 'memories';
                    renderMemoriesList(data);
                } else if (view === 'summaries') {
                    url = `/api/summaries?codebase=${codebase}&limit=${BROWSE_LIMIT}&offset=${currentBrowseOffset}`;
                    const res = await fetch(url);
                    data = await res.json();
                    browseData.summaries = data.summaries || [];
                    browseData.currentView = 'summaries';
                    renderSummariesList(data);
                }
            } catch (err) {
                listContent.innerHTML = `<div class="error-msg" style="margin: 20px;">Error: ${err.message}</div>`;
            }
        }
        
        function clearDetailPanel() {
            document.getElementById('detail-title').textContent = 'Select a file to view details';
            document.getElementById('detail-content').innerHTML = `
                <div class="browse-detail-empty">
                    <span>← Select an item from the list</span>
                </div>
            `;
        }
        
        function renderFilesList(data) {
            const listContent = document.getElementById('browse-list-content');
            const countSpan = document.getElementById('browse-list-count');
            const files = data.files || [];

            // Store data for view mode switching
            browseData.total = data.total;
            browseData.offset = data.offset;

            // Show active filters count
            const activeFilters = getActiveFiltersCount();
            const filterNote = activeFilters > 0 ? ` (${activeFilters} filter${activeFilters > 1 ? 's' : ''})` : '';

            if (files.length === 0) {
                listContent.innerHTML = `<div class="empty-state" style="padding: 40px;">No files found${filterNote ? ' - try clearing filters' : ''}</div>`;
                countSpan.textContent = `0 files${filterNote}`;
                renderPagination(0, 0);
                return;
            }

            countSpan.textContent = `${data.offset + 1}-${data.offset + files.length} of ${data.total} files${filterNote}`;

            let html = '';

            if (browseViewMode === 'tree') {
                // Tree view
                const tree = buildFileTree(files);
                html = renderTreeNode(tree);
            } else {
                // List view
                files.forEach((f, idx) => {
                    const patternTag = f.pattern ? `<span class="tag" style="font-size: 10px;">${f.pattern}</span>` : '';
                    const summaryBadge = getSummaryBadge(f.has_summary, f.simple_file, f.simple_file_reason);
                    html += `
                        <div class="file-list-item" data-index="${idx}" onclick="selectFileItem(${idx})" tabindex="-1">
                            <span class="file-item-path" title="${escapeHtml(f.path)}">${escapeHtml(f.path)}</span>
                            <div class="file-item-meta">
                                ${summaryBadge}
                                ${patternTag}
                                <span class="file-item-chunks">${f.chunk_count}</span>
                            </div>
                        </div>
                    `;
                });
            }

            listContent.innerHTML = html;
            renderPagination(data.total, data.offset);
        }
        
        function selectFileItem(idx) {
            const view = browseData.currentView;

            // Update selection UI (handle both list and tree views)
            document.querySelectorAll('.file-list-item, .tree-file').forEach(el => {
                const elIdx = parseInt(el.getAttribute('data-index'));
                el.classList.toggle('selected', elIdx === idx);
            });
            selectedIndex = idx;
            
            // Load details based on view type
            if (view === 'files') {
                const file = browseData.files[idx];
                if (file) {
                    loadFileDetailsPanel(browseData.codebase, file.path);
                }
            } else if (view === 'memories') {
                const memory = browseData.memories[idx];
                if (memory) {
                    showMemoryDetailsPanel(memory);
                }
            } else if (view === 'summaries') {
                const summary = browseData.summaries[idx];
                if (summary) {
                    showSummaryDetailsPanel(summary);
                }
            }
        }
        
        async function loadFileDetailsPanel(codebase, filePath) {
            const titleEl = document.getElementById('detail-title');
            const contentEl = document.getElementById('detail-content');
            
            titleEl.textContent = filePath;
            contentEl.innerHTML = '<div class="loading" style="padding: 20px; text-align: center;">Loading...</div>';
            
            try {
                const res = await fetch(`/api/file-details?codebase=${codebase}&path=${encodeURIComponent(filePath)}`);
                const data = await res.json();
                
                if (data.error) {
                    contentEl.innerHTML = `<div class="error-msg">${data.error}</div>`;
                    return;
                }
                
                const summaryMethod = getSummarizationMethod(data.summary);
                
                let html = `<div class="detail-meta-grid">
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Codebase</div>
                        <div class="detail-meta-value">${data.codebase}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Chunks</div>
                        <div class="detail-meta-value">${data.chunk_count}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Indexed</div>
                        <div class="detail-meta-value">${data.indexed_at ? new Date(data.indexed_at).toLocaleString() : '-'}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Summarization</div>
                        <div class="detail-meta-value">${summaryMethod || 'Pending'}</div>
                    </div>
                </div>`;
                
                if (data.summary) {
                    html += renderSummarySection(data.summary);
                }
                
                html += '<h3 style="color: #888; font-size: 13px; margin: 15px 0 10px;">Chunks</h3><div class="chunk-list">';
                
                (data.chunks || []).forEach((chunk, idx) => {
                    const chunkContent = escapeHtml(chunk.content || '');
                    const preview = chunkContent.length > 200 ? chunkContent.substring(0, 200) + '...' : chunkContent;
                    
                    html += `
                        <div class="chunk-item">
                            <div class="chunk-header">
                                <span>Chunk ${idx + 1}</span>
                                <span>${chunk.memory_type || ''}</span>
                            </div>
                            <div class="chunk-content" id="detail-chunk-${idx}">${preview}</div>
                            ${chunkContent.length > 200 ? `<span class="expand-btn" onclick="toggleDetailChunk(${idx}, \\`${escapeJs(chunkContent)}\\`)">Show more</span>` : ''}
                        </div>
                    `;
                });
                
                html += '</div>';
                contentEl.innerHTML = html;
            } catch (err) {
                contentEl.innerHTML = `<div class="error-msg">Error: ${err.message}</div>`;
            }
        }
        
        function toggleDetailChunk(idx, fullContent) {
            const el = document.getElementById(`detail-chunk-${idx}`);
            const btn = el.nextElementSibling;
            if (el.classList.contains('expanded')) {
                el.classList.remove('expanded');
                el.innerHTML = fullContent.substring(0, 200) + '...';
                btn.textContent = 'Show more';
            } else {
                el.classList.add('expanded');
                el.innerHTML = fullContent;
                btn.textContent = 'Show less';
            }
        }
        
        // Legacy function for backward compatibility
        function renderFilesTable(data) {
            renderFilesList(data);
        }
        
        function renderMemoriesList(data) {
            const listContent = document.getElementById('browse-list-content');
            const countSpan = document.getElementById('browse-list-count');
            const memories = data.memories || [];
            
            if (memories.length === 0) {
                listContent.innerHTML = '<div class="empty-state" style="padding: 40px;">No memories found</div>';
                countSpan.textContent = '0 memories';
                renderPagination(0, 0);
                return;
            }
            
            countSpan.textContent = `${data.offset + 1}-${data.offset + memories.length} of ${data.total} memories`;
            
            let html = '';
            memories.forEach((m, idx) => {
                const badgeClass = m.type === 'decision' ? 'badge-decision' : 
                                   m.type === 'lesson' ? 'badge-lesson' : 'badge-conversation';
                html += `
                    <div class="file-list-item" data-index="${idx}" onclick="selectFileItem(${idx})" tabindex="-1">
                        <div style="display: flex; align-items: center; gap: 10px; overflow: hidden; flex: 1;">
                            <span class="badge ${badgeClass}">${m.type}</span>
                            <span class="file-item-path" title="${escapeHtml(m.content_preview)}">${escapeHtml(m.content_preview)}</span>
                        </div>
                        <div class="file-item-meta">
                            <span style="font-size: 11px; color: #555;">${m.codebase}</span>
                        </div>
                    </div>
                `;
            });
            
            listContent.innerHTML = html;
            renderPagination(data.total, data.offset);
        }
        
        function showMemoryDetailsPanel(memory) {
            const titleEl = document.getElementById('detail-title');
            const contentEl = document.getElementById('detail-content');
            
            titleEl.textContent = `Memory: ${memory.type}`;
            
            const tags = (memory.tags || []).map(t => `<span class="tag">${t}</span>`).join(' ');
            const created = memory.created_at ? new Date(memory.created_at).toLocaleString() : '-';
            
            contentEl.innerHTML = `
                <div class="detail-meta-grid">
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">ID</div>
                        <div class="detail-meta-value">${memory.id}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Type</div>
                        <div class="detail-meta-value">${memory.type}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Codebase</div>
                        <div class="detail-meta-value">${memory.codebase}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Source</div>
                        <div class="detail-meta-value">${memory.source}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Created</div>
                        <div class="detail-meta-value">${created}</div>
                    </div>
                </div>
                ${tags ? `<div style="margin-bottom: 15px;">${tags}</div>` : ''}
                <div style="background: #0f3460; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 13px; white-space: pre-wrap; max-height: 300px; overflow-y: auto;">${escapeHtml(memory.content)}</div>
            `;
        }
        
        function renderSummariesList(data) {
            const listContent = document.getElementById('browse-list-content');
            const countSpan = document.getElementById('browse-list-count');
            const summaries = data.summaries || [];
            
            if (summaries.length === 0) {
                listContent.innerHTML = '<div class="empty-state" style="padding: 40px;">No summaries found</div>';
                countSpan.textContent = '0 summaries';
                renderPagination(0, 0);
                return;
            }
            
            countSpan.textContent = `${data.offset + 1}-${data.offset + summaries.length} of ${data.total} summaries`;
            
            let html = '';
            summaries.forEach((s, idx) => {
                html += `
                    <div class="file-list-item" data-index="${idx}" onclick="selectFileItem(${idx})" tabindex="-1">
                        <span class="file-item-path" title="${escapeHtml(s.file_path)}">${escapeHtml(s.file_path)}</span>
                        <div class="file-item-meta">
                            <span class="tag">${s.pattern || '-'}</span>
                            <span class="tag">${s.domain || '-'}</span>
                        </div>
                    </div>
                `;
            });
            
            listContent.innerHTML = html;
            renderPagination(data.total, data.offset);
        }
        
        function showSummaryDetailsPanel(summary) {
            const titleEl = document.getElementById('detail-title');
            const contentEl = document.getElementById('detail-content');
            
            titleEl.textContent = summary.file_path;
            const summarizedAt = summary.summarized_at ? new Date(summary.summarized_at).toLocaleString() : '-';
            
            contentEl.innerHTML = `
                <div class="detail-meta-grid">
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Pattern</div>
                        <div class="detail-meta-value">${summary.pattern || '-'}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Domain</div>
                        <div class="detail-meta-value">${summary.domain || '-'}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Model</div>
                        <div class="detail-meta-value">${summary.model || '-'}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Summarized</div>
                        <div class="detail-meta-value">${summarizedAt}</div>
                    </div>
                </div>
                <div class="summary-section">
                    <div class="summary-content">${escapeHtml(summary.content)}</div>
                </div>
            `;
        }
        
        // Legacy functions for backward compatibility
        function renderMemoriesTable(data) {
            renderMemoriesList(data);
        }
        
        function renderSummariesTable(data) {
            renderSummariesList(data);
        }
        
        function renderPagination(total, offset) {
            const paginationDiv = document.getElementById('browse-pagination');
            if (total <= BROWSE_LIMIT) {
                paginationDiv.innerHTML = '';
                return;
            }
            
            const totalPages = Math.ceil(total / BROWSE_LIMIT);
            const currentPage = Math.floor(offset / BROWSE_LIMIT) + 1;
            
            let html = `<button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${currentPage - 1})">Prev</button>`;
            
            for (let i = 1; i <= Math.min(totalPages, 10); i++) {
                html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
            }
            
            if (totalPages > 10) {
                html += `<span style="color: #888;">... ${totalPages}</span>`;
            }
            
            html += `<button class="page-btn" ${currentPage === totalPages ? 'disabled' : ''} onclick="goToPage(${currentPage + 1})">Next</button>`;
            
            paginationDiv.innerHTML = html;
        }
        
        function goToPage(page) {
            currentBrowseOffset = (page - 1) * BROWSE_LIMIT;
            loadBrowseData();
        }
        
        // Keyboard navigation for browse list
        document.getElementById('browse-list-content').addEventListener('keydown', function(e) {
            const items = document.querySelectorAll('.file-list-item');
            if (items.length === 0) return;
            
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                const newIndex = selectedIndex < items.length - 1 ? selectedIndex + 1 : 0;
                selectFileItem(newIndex);
                items[newIndex].scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const newIndex = selectedIndex > 0 ? selectedIndex - 1 : items.length - 1;
                selectFileItem(newIndex);
                items[newIndex].scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'Enter' && selectedIndex >= 0) {
                e.preventDefault();
                // Already selected, just ensure it's loaded
                selectFileItem(selectedIndex);
            }
        });
        
        // Legacy compatibility - redirect old functions to new ones
        async function loadFileDetails(codebase, filePath) {
            await loadFileDetailsPanel(codebase, filePath);
        }
        
        function showMemoryDetails(memory) {
            showMemoryDetailsPanel(memory);
        }
        
        function showSummaryDetails(summary) {
            showSummaryDetailsPanel(summary);
        }
        
        function toggleChunkExpand(idx, fullContent) {
            toggleDetailChunk(idx, fullContent);
        }
        
        function closeDetails() {
            clearDetailPanel();
        }
        
        // Browse event listeners
        document.getElementById('browse-view').addEventListener('change', () => {
            currentBrowseOffset = 0;
            loadBrowseData();
        });
        document.getElementById('browse-codebase').addEventListener('change', () => {
            currentBrowseOffset = 0;
            loadBrowseData();
        });
        document.getElementById('browse-filter').addEventListener('input', debounce(() => {
            currentBrowseOffset = 0;
            loadBrowseData();
        }, 300));
        
        // Quick filter event listeners
        ['filter-has-summary', 'filter-pattern', 'filter-domain', 'filter-language'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                currentBrowseOffset = 0;
                updateFilterIndicators();
                loadBrowseData();
            });
        });
        
        // =========== Utilities ===========
        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                      .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
        }
        
        function escapeJs(str) {
            if (!str) return '';
            var bs = String.fromCharCode(92);
            return str.split(bs).join(bs+bs).split('`').join(bs+'`').split('$').join(bs+'$');
        }
        
        function debounce(fn, delay) {
            let timeout;
            return (...args) => {
                clearTimeout(timeout);
                timeout = setTimeout(() => fn(...args), delay);
            };
        }
        
        function getSummaryBadge(hasSummary, simpleFile, simpleFileReason) {
            if (!hasSummary) {
                return '<span class="summary-badge pending" title="No summary yet"><span class="summary-badge-icon">⏳</span>Pending</span>';
            }
            if (simpleFile) {
                const reasonMap = {
                    'barrel_reexport': 'Barrel/re-export file',
                    'empty_module': 'Empty module',
                    'generated_code': 'Generated code',
                    'constants_only': 'Constants only',
                    'type_definitions': 'Type definitions only'
                };
                const tooltip = reasonMap[simpleFileReason] || simpleFileReason || 'Simple file';
                return `<span class="summary-badge simple" title="${tooltip}"><span class="summary-badge-icon">🔹</span>Simple</span>`;
            }
            return '<span class="summary-badge llm" title="LLM summarized"><span class="summary-badge-icon">📝</span>LLM</span>';
        }
        
        function getSummarizationMethod(summary) {
            if (!summary) return null;
            if (summary.simple_file) {
                const reasonMap = {
                    'barrel_reexport': 'Barrel/re-export',
                    'empty_module': 'Empty module',
                    'generated_code': 'Generated code',
                    'constants_only': 'Constants only',
                    'type_definitions': 'Type definitions'
                };
                const reason = reasonMap[summary.simple_file_reason] || summary.simple_file_reason || 'template';
                return `Simple (${reason})`;
            }
            return `LLM (${summary.model || 'unknown'})`;
        }
        
        // Render summary section with Phase 2 fields (how_it_works, key_mechanisms, method_summaries)
        function renderSummarySection(summary) {
            if (!summary) return '';
            
            // Header with pattern and domain tags
            let html = `<div class="summary-section">
                <div class="summary-title">Summary</div>`;
            
            // Pattern and domain tags
            const tags = [];
            if (summary.pattern) tags.push(summary.pattern);
            if (summary.domain) tags.push(summary.domain);
            if (tags.length > 0) {
                html += `<div style="margin-bottom: 10px;">`;
                tags.forEach(tag => {
                    html += `<span class="tag">${escapeHtml(tag)}</span> `;
                });
                html += `</div>`;
            }
            
            // Purpose (from content or purpose field)
            const purpose = summary.purpose || summary.content || '';
            if (purpose) {
                html += `<div style="margin-bottom: 8px;"><strong style="color: #888;">Purpose:</strong> ${escapeHtml(purpose)}</div>`;
            }
            
            // How It Works (collapsible)
            if (summary.how_it_works) {
                const sectionId = 'how-it-works-' + Math.random().toString(36).substr(2, 9);
                html += `
                    <div class="collapsible-section">
                        <div class="collapsible-header" onclick="toggleCollapsible('${sectionId}', this)">
                            <span class="collapsible-arrow">▶</span>
                            <span>How It Works</span>
                        </div>
                        <div class="collapsible-body" id="${sectionId}">${escapeHtml(summary.how_it_works)}</div>
                    </div>`;
            }
            
            // Key Mechanisms (as tag badges)
            if (summary.key_mechanisms && summary.key_mechanisms.length > 0) {
                html += `
                    <div class="collapsible-section">
                        <div style="color: #888; font-size: 12px; margin-bottom: 6px;">Key Mechanisms:</div>
                        <div class="mechanism-tags">`;
                summary.key_mechanisms.forEach(mech => {
                    html += `<span class="mechanism-tag">${escapeHtml(mech)}</span>`;
                });
                html += `</div></div>`;
            }
            
            // Method Summaries (collapsible list)
            if (summary.method_summaries && Object.keys(summary.method_summaries).length > 0) {
                const methods = Object.entries(summary.method_summaries);
                const sectionId = 'method-summaries-' + Math.random().toString(36).substr(2, 9);
                html += `
                    <div class="collapsible-section">
                        <div class="collapsible-header" onclick="toggleCollapsible('${sectionId}', this)">
                            <span class="collapsible-arrow">▶</span>
                            <span>Method Summaries (${methods.length})</span>
                        </div>
                        <div class="collapsible-body" id="${sectionId}">
                            <div class="method-summaries-list">`;
                methods.forEach(([name, desc]) => {
                    html += `
                        <div class="method-item">
                            <span class="method-name">${escapeHtml(name)}()</span>
                            <span class="method-desc">${escapeHtml(desc)}</span>
                        </div>`;
                });
                html += `</div></div></div>`;
            }
            
            // Exports
            if (summary.exports && summary.exports.length > 0) {
                html += `
                    <div class="collapsible-section">
                        <div style="color: #888; font-size: 12px; margin-bottom: 6px;">Exports:</div>
                        <div style="color: #ccc; font-size: 13px;">${escapeHtml(summary.exports.join(', '))}</div>
                    </div>`;
            }
            
            html += `</div>`;
            return html;
        }
        
        // Toggle collapsible section
        function toggleCollapsible(sectionId, headerEl) {
            const body = document.getElementById(sectionId);
            if (body.classList.contains('show')) {
                body.classList.remove('show');
                headerEl.classList.remove('expanded');
            } else {
                body.classList.add('show');
                headerEl.classList.add('expanded');
            }
        }
        
        // Render summary panel for Summaries (Validate) tab with Phase 2 fields
        function renderValidateSummaryPanel(summary) {
            if (!summary) return '<div class="error-msg">No summary data</div>';
            
            let html = '';
            
            // Purpose section
            const purpose = summary.purpose || summary.content || '';
            if (purpose) {
                html += `
                    <div style="margin-bottom: 15px;">
                        <div style="color: #888; font-size: 11px; margin-bottom: 4px;">PURPOSE</div>
                        <div style="color: #eee; font-family: sans-serif;">${escapeHtml(purpose)}</div>
                    </div>`;
            }
            
            // Pattern and Domain tags
            if (summary.pattern || summary.domain) {
                html += `<div style="margin-bottom: 12px;">`;
                if (summary.pattern) {
                    html += `<span class="tag">${escapeHtml(summary.pattern)}</span> `;
                }
                if (summary.domain) {
                    html += `<span class="tag">${escapeHtml(summary.domain)}</span>`;
                }
                html += `</div>`;
            }
            
            // Key Mechanisms (as tag badges)
            if (summary.key_mechanisms && summary.key_mechanisms.length > 0) {
                html += `
                    <div style="margin-bottom: 12px;">
                        <div style="color: #888; font-size: 11px; margin-bottom: 6px;">KEY MECHANISMS</div>
                        <div class="mechanism-tags">`;
                summary.key_mechanisms.forEach(mech => {
                    html += `<span class="mechanism-tag">${escapeHtml(mech)}</span>`;
                });
                html += `</div></div>`;
            }
            
            // How It Works (expanded by default)
            if (summary.how_it_works) {
                const sectionId = 'validate-how-it-works-' + Math.random().toString(36).substr(2, 9);
                html += `
                    <div class="collapsible-section" style="margin-bottom: 8px;">
                        <div class="collapsible-header expanded" onclick="toggleCollapsible('${sectionId}', this)">
                            <span class="collapsible-arrow">▶</span>
                            <span>How It Works</span>
                        </div>
                        <div class="collapsible-body show" id="${sectionId}">${escapeHtml(summary.how_it_works)}</div>
                    </div>`;
            }

            // Method Summaries (expanded by default)
            if (summary.method_summaries && Object.keys(summary.method_summaries).length > 0) {
                const methods = Object.entries(summary.method_summaries);
                const sectionId = 'validate-methods-' + Math.random().toString(36).substr(2, 9);
                html += `
                    <div class="collapsible-section" style="margin-bottom: 8px;">
                        <div class="collapsible-header expanded" onclick="toggleCollapsible('${sectionId}', this)">
                            <span class="collapsible-arrow">▶</span>
                            <span>Methods (${methods.length})</span>
                        </div>
                        <div class="collapsible-body show" id="${sectionId}">
                            <div class="method-summaries-list">`;
                methods.forEach(([name, desc]) => {
                    html += `
                        <div class="method-item">
                            <span class="method-name">${escapeHtml(name)}()</span>
                            <span class="method-desc">${escapeHtml(desc)}</span>
                        </div>`;
                });
                html += `</div></div></div>`;
            }
            
            // Exports - handle both array and string formats
            const exports = summary.key_exports || summary.exports;
            if (exports && (Array.isArray(exports) ? exports.length > 0 : exports)) {
                const exportsText = Array.isArray(exports) ? exports.join(', ') : exports;
                html += `
                    <div style="margin-top: 12px;">
                        <div style="color: #888; font-size: 11px; margin-bottom: 4px;">EXPORTS</div>
                        <div style="color: #ccc; font-size: 13px;">${escapeHtml(exportsText)}</div>
                    </div>`;
            }
            
            return html || '<div style="color: #888;">No summary details available</div>';
        }
        
        // =========== Summary Statistics ===========
        let summaryStatsCollapsed = false;
        
        function toggleSummaryStats() {
            summaryStatsCollapsed = !summaryStatsCollapsed;
            const header = document.querySelector('.summary-stats-header');
            const body = document.getElementById('summary-stats-body');
            
            if (summaryStatsCollapsed) {
                header.classList.add('collapsed');
                body.classList.add('hidden');
            } else {
                header.classList.remove('collapsed');
                body.classList.remove('hidden');
            }
        }
        
        async function loadSummaryStats() {
            const codebase = document.getElementById('validate-codebase').value || '';
            const loadingEl = document.getElementById('summary-stats-loading');
            const bodyEl = document.getElementById('summary-stats-body');
            const totalEl = document.getElementById('summary-stats-total');
            
            try {
                const params = codebase ? `?codebase=${encodeURIComponent(codebase)}` : '';
                const res = await fetch(`/api/summary-stats${params}`);
                const data = await res.json();
                
                if (data.error) {
                    loadingEl.textContent = 'Error: ' + data.error;
                    return;
                }
                
                // Update total in header
                totalEl.textContent = `${data.total_summarized || 0} summaries`;
                
                // Render the three columns
                bodyEl.innerHTML = renderStatsColumns(data);
                
            } catch (err) {
                console.error('Error loading summary stats:', err);
                loadingEl.textContent = 'Failed to load statistics';
            }
        }
        
        function renderStatsColumns(data) {
            // Pattern column
            const patterns = Object.entries(data.by_pattern || {}).slice(0, 10);
            let patternHtml = '';
            patterns.forEach(([name, count]) => {
                patternHtml += `
                    <div class="stats-item">
                        <span class="stats-item-label" title="${escapeHtml(name)}">${escapeHtml(name || '(none)')}</span>
                        <span class="stats-item-count">${count}</span>
                    </div>`;
            });
            if (patterns.length === 0) {
                patternHtml = '<div style="color: #555; font-size: 12px; padding: 8px;">No patterns</div>';
            }
            
            // Domain column
            const domains = Object.entries(data.by_domain || {}).slice(0, 10);
            let domainHtml = '';
            domains.forEach(([name, count]) => {
                domainHtml += `
                    <div class="stats-item">
                        <span class="stats-item-label" title="${escapeHtml(name)}">${escapeHtml(name || '(none)')}</span>
                        <span class="stats-item-count">${count}</span>
                    </div>`;
            });
            if (domains.length === 0) {
                domainHtml = '<div style="color: #555; font-size: 12px; padding: 8px;">No domains</div>';
            }
            
            // Status column with icons
            const statuses = data.by_status || {};
            const statusOrder = ['approved', 'rejected', 'unreviewed'];
            const statusLabels = {
                'approved': '✓ Approved',
                'rejected': '✗ Rejected', 
                'unreviewed': '● Pending'
            };
            let statusHtml = '';
            statusOrder.forEach(status => {
                const count = statuses[status] || 0;
                if (count > 0 || status === 'unreviewed') {
                    statusHtml += `
                        <div class="stats-item status-${status}">
                            <span class="stats-item-label">${statusLabels[status] || status}</span>
                            <span class="stats-item-count">${count}</span>
                        </div>`;
                }
            });
            // Add any other statuses not in our order
            Object.entries(statuses).forEach(([status, count]) => {
                if (!statusOrder.includes(status) && count > 0) {
                    statusHtml += `
                        <div class="stats-item">
                            <span class="stats-item-label">${escapeHtml(status)}</span>
                            <span class="stats-item-count">${count}</span>
                        </div>`;
                }
            });
            if (!statusHtml) {
                statusHtml = '<div style="color: #555; font-size: 12px; padding: 8px;">No status data</div>';
            }
            
            return `
                <div class="stats-column">
                    <div class="stats-column-title">By Pattern</div>
                    <div class="stats-list">${patternHtml}</div>
                </div>
                <div class="stats-column">
                    <div class="stats-column-title">By Domain</div>
                    <div class="stats-list">${domainHtml}</div>
                </div>
                <div class="stats-column">
                    <div class="stats-column-title">By Status</div>
                    <div class="stats-list">${statusHtml}</div>
                </div>
            `;
        }
        
        // =========== Validate Tab ===========
        let validateFiles = [];         // Current page of files
        let validateTotal = 0;          // Total matching filters
        let validateOffset = 0;         // Current pagination offset
        let validateSelectedIndex = -1; // Selected index in current page
        
        function getValidatePageSize() {
            return parseInt(document.getElementById('validate-page-size').value) || 20;
        }
        
        // Alias for refresh button
        function loadValidateFiles() {
            loadValidationQueue();
        }

        async function loadValidationQueue() {
            const codebase = document.getElementById('validate-codebase').value;
            const status = document.getElementById('validate-status').value;
            const pattern = document.getElementById('validate-pattern').value;
            const domain = document.getElementById('validate-domain').value;
            const sort = document.getElementById('validate-sort').value;
            const limit = getValidatePageSize();

            if (!codebase) {
                document.getElementById('validate-progress').textContent = 'Select a codebase';
                return;
            }

            document.getElementById('validate-list-content').innerHTML = '<div class="loading" style="padding: 20px;">Loading...</div>';

            try {
                const params = new URLSearchParams({
                    codebase,
                    status,
                    offset: validateOffset,
                    limit,
                    sort
                });
                if (pattern) params.append('pattern', pattern);
                if (domain) params.append('domain', domain);
                
                const res = await fetch(`/api/validation-queue?${params}`);
                const data = await res.json();
                
                if (data.error) {
                    document.getElementById('validate-progress').textContent = 'Error: ' + data.error;
                    return;
                }
                
                validateFiles = data.files || [];
                validateTotal = data.total || 0;
                
                // Populate filter dropdowns with available options
                if (data.pattern_options) {
                    populateValidateFilterDropdown('validate-pattern', data.pattern_options, 'All Patterns');
                }
                if (data.domain_options) {
                    populateValidateFilterDropdown('validate-domain', data.domain_options, 'All Domains');
                }
                
                // Update progress
                const reviewed = data.reviewed_count || 0;
                document.getElementById('validate-progress').textContent = 
                    `${reviewed} reviewed / ${validateTotal} total`;
                
                // Render file list
                renderValidateFileList();
                
                // Auto-select first file if available and none selected
                if (validateFiles.length > 0 && validateSelectedIndex < 0) {
                    selectValidateFile(0);
                } else if (validateFiles.length === 0) {
                    clearValidatePanels();
                }
                
                // Update filter indicator
                updateValidateFilterIndicators();
                
            } catch (err) {
                document.getElementById('validate-progress').textContent = 'Error loading queue';
                console.error('Validation queue error:', err);
            }
        }
        
        function populateValidateFilterDropdown(selectId, options, defaultLabel) {
            const select = document.getElementById(selectId);
            const currentValue = select.value;
            select.innerHTML = `<option value="">${defaultLabel}</option>`;
            options.forEach(opt => {
                if (opt) {
                    select.innerHTML += `<option value="${opt}">${opt}</option>`;
                }
            });
            // Restore previous selection if still valid
            if (currentValue && options.includes(currentValue)) {
                select.value = currentValue;
            }
        }
        
        function renderValidateFileList() {
            const listContent = document.getElementById('validate-list-content');
            const countSpan = document.getElementById('validate-list-count');
            
            if (validateFiles.length === 0) {
                listContent.innerHTML = '<div class="empty-state" style="padding: 20px;">No files match filters</div>';
                countSpan.textContent = '0 files';
                renderValidatePagination();
                return;
            }
            
            const start = validateOffset + 1;
            const end = validateOffset + validateFiles.length;
            countSpan.textContent = `${start}-${end} of ${validateTotal}`;
            
            let html = '';
            validateFiles.forEach((file, idx) => {
                const statusIcon = file.status === 'approved' ? '✓' :
                                  file.status === 'rejected' ? '✗' : '●';
                const statusClass = file.status || 'unreviewed';
                const selected = idx === validateSelectedIndex ? 'selected' : '';
                const simpleClass = file.simple_file ? 'simple-file' : '';
                
                // Show simple file badge (auto-approved indicator)
                const simpleBadge = file.simple_file 
                    ? '<span class="validate-simple-badge" title="Simple file (auto-approved)">🔹</span>' 
                    : '';
                
                html += `
                    <div class="validate-file-item ${selected} ${simpleClass}" data-index="${idx}" onclick="selectValidateFile(${idx})" tabindex="-1">
                        ${simpleBadge}<span class="validate-file-item-path" title="${escapeHtml(file.path)}">${escapeHtml(file.path)}</span>
                        <span class="validate-status-icon ${statusClass}">${statusIcon}</span>
                    </div>
                `;
            });
            
            listContent.innerHTML = html;
            renderValidatePagination();
        }
        
        function renderValidatePagination() {
            const container = document.getElementById('validate-pagination');
            const limit = getValidatePageSize();
            const totalPages = Math.ceil(validateTotal / limit);
            const currentPage = Math.floor(validateOffset / limit) + 1;
            
            if (totalPages <= 1) {
                container.innerHTML = '';
                return;
            }
            
            let html = '';
            
            // Previous button
            if (currentPage > 1) {
                html += `<button class="page-btn" onclick="validateGoToPage(${currentPage - 1})">‹</button>`;
            }
            
            // Page numbers (show max 5 pages with ellipsis for compact display)
            const maxVisible = 5;
            let startPage = Math.max(1, currentPage - 2);
            let endPage = Math.min(totalPages, startPage + maxVisible - 1);
            startPage = Math.max(1, endPage - maxVisible + 1);
            
            if (startPage > 1) {
                html += `<button class="page-btn" onclick="validateGoToPage(1)">1</button>`;
                if (startPage > 2) html += '<span style="color:#555;padding:0 2px;">…</span>';
            }
            
            for (let p = startPage; p <= endPage; p++) {
                const active = p === currentPage ? 'active' : '';
                html += `<button class="page-btn ${active}" onclick="validateGoToPage(${p})">${p}</button>`;
            }
            
            if (endPage < totalPages) {
                if (endPage < totalPages - 1) html += '<span style="color:#555;padding:0 2px;">…</span>';
                html += `<button class="page-btn" onclick="validateGoToPage(${totalPages})">${totalPages}</button>`;
            }
            
            // Next button
            if (currentPage < totalPages) {
                html += `<button class="page-btn" onclick="validateGoToPage(${currentPage + 1})">›</button>`;
            }
            
            container.innerHTML = html;
        }
        
        function validateGoToPage(page) {
            const limit = getValidatePageSize();
            validateOffset = (page - 1) * limit;
            validateSelectedIndex = -1;
            loadValidationQueue();
        }
        
        function selectValidateFile(idx) {
            if (idx < 0 || idx >= validateFiles.length) return;
            
            validateSelectedIndex = idx;
            
            // Update selection UI
            document.querySelectorAll('.validate-file-item').forEach((el, i) => {
                el.classList.toggle('selected', i === idx);
            });
            
            // Load file content
            loadValidationFileContent(validateFiles[idx]);
        }
        
        async function loadValidationFileContent(file) {
            const codebase = document.getElementById('validate-codebase').value;
            
            document.getElementById('validate-file-path').textContent = file.path;
            document.getElementById('validate-source-content').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('validate-summary-content').innerHTML = '<div class="loading">Loading...</div>';
            
            try {
                const [fileRes, summaryRes] = await Promise.all([
                    fetch(`/api/file-content?codebase=${codebase}&path=${encodeURIComponent(file.path)}`),
                    fetch(`/api/file-summary?codebase=${codebase}&path=${encodeURIComponent(file.path)}`)
                ]);
                
                const fileData = await fileRes.json();
                const summaryData = await summaryRes.json();
                
                // Display file content
                if (fileData.content) {
                    document.getElementById('validate-source-content').textContent = fileData.content;
                } else {
                    document.getElementById('validate-source-content').innerHTML = 
                        `<div class="error-msg">${fileData.error || 'Could not load file'}</div>`;
                }
                
                // Display summary with Phase 2 fields
                if (summaryData.summary) {
                    const summary = summaryData.summary;
                    
                    // Build header: show simple file badge or model info
                    let headerHtml = '';
                    if (summary.simple_file) {
                        const reasonMap = {
                            'barrel_reexport': 'Barrel/re-export',
                            'empty_module': 'Empty module',
                            'generated_code': 'Generated code',
                            'constants_only': 'Constants only',
                            'type_definitions': 'Type definitions'
                        };
                        const reason = reasonMap[summary.simple_file_reason] || summary.simple_file_reason || 'template';
                        headerHtml = `<span class="validate-simple-header-badge">🔹 Simple (${reason})</span>`;
                        if (summary.pattern) {
                            headerHtml += ` <span style="color:#888">• ${summary.pattern}</span>`;
                        }
                    } else {
                        headerHtml = `${summary.model || 'unknown'} • ${summary.pattern || 'unknown'}`;
                    }
                    document.getElementById('validate-model').innerHTML = headerHtml;
                    document.getElementById('validate-summary-content').innerHTML = renderValidateSummaryPanel(summary);
                } else {
                    document.getElementById('validate-model').textContent = '';
                    document.getElementById('validate-summary-content').innerHTML = 
                        '<div class="error-msg">No summary available</div>';
                }
                
                updateValidateNavigation();
            } catch (err) {
                document.getElementById('validate-source-content').innerHTML = 
                    `<div class="error-msg">Error: ${err.message}</div>`;
            }
        }
        
        function clearValidatePanels() {
            document.getElementById('validate-file-path').textContent = 'No files match filters';
            document.getElementById('validate-source-content').innerHTML = 
                '<div class="browse-detail-empty"><span>No files to display</span></div>';
            document.getElementById('validate-summary-content').innerHTML = 
                '<div class="browse-detail-empty"><span>No summaries to review</span></div>';
            document.getElementById('validate-model').textContent = '';
            updateValidateNavigation();
        }
        
        function updateValidateNavigation() {
            // Can go prev if not on first item of first page
            const canPrev = validateOffset > 0 || validateSelectedIndex > 0;
            // Can go next if not on last item of last page
            const canNext = validateFiles.length > 0 && 
                           (validateOffset + validateSelectedIndex) < (validateTotal - 1);
            
            document.getElementById('validate-prev').disabled = !canPrev;
            document.getElementById('validate-next').disabled = !canNext;
        }
        
        function validatePrev() {
            if (validateSelectedIndex > 0) {
                // Move to previous in current page
                selectValidateFile(validateSelectedIndex - 1);
            } else if (validateOffset > 0) {
                // Go to previous page, select last item
                const limit = getValidatePageSize();
                validateOffset = Math.max(0, validateOffset - limit);
                loadValidationQueue().then(() => {
                    // Select last item after load
                    if (validateFiles.length > 0) {
                        selectValidateFile(validateFiles.length - 1);
                    }
                });
            }
        }
        
        function validateNext() {
            if (validateSelectedIndex < validateFiles.length - 1) {
                // Move to next in current page
                selectValidateFile(validateSelectedIndex + 1);
            } else if (validateOffset + validateFiles.length < validateTotal) {
                // Go to next page, select first item
                const limit = getValidatePageSize();
                validateOffset += limit;
                validateSelectedIndex = -1;
                loadValidationQueue().then(() => {
                    if (validateFiles.length > 0) {
                        selectValidateFile(0);
                    }
                });
            }
        }
        
        function validateSkip() {
            validateNext();
        }
        
        async function validateApprove() {
            await submitValidation('approved');
        }
        
        async function validateReject() {
            await submitValidation('rejected');
        }
        
        async function submitValidation(newStatus) {
            if (validateFiles.length === 0 || validateSelectedIndex < 0) return;
            
            const file = validateFiles[validateSelectedIndex];
            const codebase = document.getElementById('validate-codebase').value;
            const currentStatusFilter = document.getElementById('validate-status').value;
            
            try {
                const res = await fetch('/api/summary/validate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        codebase: codebase,
                        file_path: file.path,
                        status: newStatus
                    })
                });
                
                const data = await res.json();
                if (data.success) {
                    // If viewing "unreviewed" and we just approved/rejected, remove from list
                    if (currentStatusFilter === 'unreviewed') {
                        // Remove the file from the local array
                        validateFiles.splice(validateSelectedIndex, 1);
                        validateTotal--;
                        
                        // If there are more files, select the next one (or previous if at end)
                        if (validateFiles.length > 0) {
                            if (validateSelectedIndex >= validateFiles.length) {
                                validateSelectedIndex = validateFiles.length - 1;
                            }
                            renderValidateFileList();
                            selectValidateFile(validateSelectedIndex);
                        } else if (validateTotal > 0) {
                            // Current page is empty but there are more pages
                            // Go to previous page if possible
                            if (validateOffset > 0) {
                                validateOffset = Math.max(0, validateOffset - getValidatePageSize());
                            }
                            validateSelectedIndex = -1;
                            loadValidationQueue();
                        } else {
                            // No more files
                            renderValidateFileList();
                            clearValidatePanels();
                        }
                    } else {
                        // Just update the status icon and move to next
                        file.status = newStatus;
                        renderValidateFileList();
                        validateNext();
                    }
                }
            } catch (err) {
                console.error('Validation error:', err);
            }
        }
        
        async function validateRegenerate() {
            if (validateFiles.length === 0 || validateSelectedIndex < 0) return;
            
            const file = validateFiles[validateSelectedIndex];
            const codebase = document.getElementById('validate-codebase').value;
            
            document.getElementById('validate-summary-content').innerHTML = '<div class="loading">Regenerating...</div>';
            
            try {
                const res = await fetch('/api/summary/regenerate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        codebase: codebase,
                        file_path: file.path
                    })
                });
                
                const data = await res.json();
                if (data.success) {
                    loadValidationFileContent(file);
                } else {
                    document.getElementById('validate-summary-content').innerHTML = 
                        `<div class="error-msg">${data.error || 'Failed to regenerate'}</div>`;
                }
            } catch (err) {
                document.getElementById('validate-summary-content').innerHTML = 
                    `<div class="error-msg">Error: ${err.message}</div>`;
            }
        }
        
        function updateValidateFilterIndicators() {
            const pattern = document.getElementById('validate-pattern').value;
            const domain = document.getElementById('validate-domain').value;
            const hasFilters = pattern || domain;
            
            document.getElementById('validate-filter-reset').style.display = hasFilters ? 'inline' : 'none';
        }
        
        function resetValidateFilters() {
            document.getElementById('validate-pattern').value = '';
            document.getElementById('validate-domain').value = '';
            validateOffset = 0;
            validateSelectedIndex = -1;
            updateValidateFilterIndicators();
            loadValidationQueue();
        }
        
        // Keyboard navigation for validate file list
        document.getElementById('validate-list-content').addEventListener('keydown', function(e) {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (validateSelectedIndex < validateFiles.length - 1) {
                    selectValidateFile(validateSelectedIndex + 1);
                }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (validateSelectedIndex > 0) {
                    selectValidateFile(validateSelectedIndex - 1);
                }
            }
        });
        
        // Validate tab filter event listeners
        ['validate-codebase', 'validate-status', 'validate-pattern', 'validate-domain', 'validate-sort', 'validate-page-size'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                validateOffset = 0;
                validateSelectedIndex = -1;
                loadValidationQueue();
                // Reload stats when codebase changes
                if (id === 'validate-codebase') {
                    loadSummaryStats();
                }
            });
        });
        
        // Clear filters button
        document.getElementById('validate-filter-reset').addEventListener('click', resetValidateFilters);
        
        // Populate validate codebase dropdown when codebases are loaded
        function populateValidateCodebase() {
            const select = document.getElementById('validate-codebase');
            select.innerHTML = '';
            codebases.forEach(cb => {
                select.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
            });
        }
        
        // =========== Global Keyboard Shortcuts ===========
        document.addEventListener('keydown', function(e) {
            // Escape key closes modals (works even in inputs)
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal-overlay.open').forEach(m => m.classList.remove('open'));
                return;
            }
            
            // Don't trigger other shortcuts if typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                return;
            }
            
            // Get active tab
            const activeTab = document.querySelector('.tab.active');
            const tabName = activeTab ? activeTab.dataset.tab : '';
            
            // Global shortcuts
            if (e.key === '/' || (e.ctrlKey && e.key === 'k')) {
                e.preventDefault();
                // Switch to search tab and focus input
                document.querySelector('[data-tab="search"]').click();
                setTimeout(() => document.getElementById('search-query').focus(), 100);
                return;
            }
            
            // Tab-specific shortcuts
            if (tabName === 'search') {
                // No additional shortcuts needed, Enter already works
            } else if (tabName === 'browse') {
                // Already has arrow key navigation via the list content
            } else if (tabName === 'validate') {
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    validatePrev();
                } else if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    validateNext();
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    if (validateSelectedIndex > 0) {
                        selectValidateFile(validateSelectedIndex - 1);
                    }
                } else if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    if (validateSelectedIndex < validateFiles.length - 1) {
                        selectValidateFile(validateSelectedIndex + 1);
                    }
                } else if (e.key === 'a' || e.key === 'A') {
                    e.preventDefault();
                    validateApprove();
                } else if (e.key === 'r' || e.key === 'R') {
                    e.preventDefault();
                    validateReject();
                } else if (e.key === 's' || e.key === 'S') {
                    e.preventDefault();
                    validateSkip();
                }
            }
        });
        
        // =========== Action Modals & Toasts ===========
        function showToast(message, type = 'success') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.innerHTML = `<span>${type === 'success' ? '✓' : '✗'}</span><span>${message}</span>`;
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideIn 0.3s ease reverse';
                setTimeout(() => toast.remove(), 300);
            }, 4000);
        }
        
        function closeModal(modalId) {
            document.getElementById(modalId).classList.remove('open');
        }
        
        function openModal(modalId) {
            document.getElementById(modalId).classList.add('open');
        }
        
        // Close modal on overlay click (Escape key handled in global keyboard shortcuts)
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', function(e) {
                if (e.target === this) {
                    this.classList.remove('open');
                }
            });
        });
        
        function populateActionCodebaseDropdowns() {
            const queueSelect = document.getElementById('queue-codebase-select');
            const reindexSelect = document.getElementById('reindex-codebase-select');
            
            queueSelect.innerHTML = '';
            reindexSelect.innerHTML = '';
            
            codebases.forEach(cb => {
                queueSelect.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
                reindexSelect.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
            });
        }
        
        function openQueueSummarizationModal() {
            populateActionCodebaseDropdowns();
            openModal('modal-queue-summarization');
        }
        
        function openInvalidateSummariesModal() {
            openModal('modal-invalidate-summaries');
        }
        
        function openReindexCodebaseModal() {
            populateActionCodebaseDropdowns();
            openModal('modal-reindex-codebase');
        }
        
        async function confirmQueueSummarization() {
            const codebase = document.getElementById('queue-codebase-select').value;
            const onlyMissing = document.getElementById('queue-only-missing').checked;
            const btn = document.getElementById('btn-queue-summarization');
            const confirmBtn = document.getElementById('queue-confirm-btn');
            
            if (!codebase) {
                showToast('Please select a codebase', 'error');
                return;
            }
            
            closeModal('modal-queue-summarization');
            btn.classList.add('loading');
            confirmBtn.disabled = true;
            
            try {
                const res = await fetch('/api/queue-summarization', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ codebase, only_missing: onlyMissing })
                });
                
                const data = await res.json();
                
                if (data.success) {
                    showToast(`Queued ${data.files_queued} files for summarization`, 'success');
                    fetchStatus(); // Refresh status
                } else {
                    showToast(data.error || 'Failed to queue summarization', 'error');
                }
            } catch (err) {
                showToast('Error: ' + err.message, 'error');
            } finally {
                btn.classList.remove('loading');
                confirmBtn.disabled = false;
            }
        }
        
        async function confirmInvalidateSummaries() {
            const btn = document.getElementById('btn-invalidate-summaries');
            const confirmBtn = document.getElementById('invalidate-confirm-btn');
            
            closeModal('modal-invalidate-summaries');
            btn.classList.add('loading');
            confirmBtn.disabled = true;
            
            try {
                const res = await fetch('/api/invalidate-summaries', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: '{}'
                });
                
                const data = await res.json();
                
                if (data.success) {
                    const count = data.total_summaries_cleared || 0;
                    showToast(`Invalidated ${count} summaries`, 'success');
                    fetchStatus(); // Refresh status
                } else {
                    showToast(data.error || 'Failed to invalidate summaries', 'error');
                }
            } catch (err) {
                showToast('Error: ' + err.message, 'error');
            } finally {
                btn.classList.remove('loading');
                confirmBtn.disabled = false;
            }
        }
        
        async function confirmReindexCodebase() {
            const codebase = document.getElementById('reindex-codebase-select').value;
            const btn = document.getElementById('btn-reindex-codebase');
            const confirmBtn = document.getElementById('reindex-confirm-btn');
            
            if (!codebase) {
                showToast('Please select a codebase', 'error');
                return;
            }
            
            closeModal('modal-reindex-codebase');
            btn.classList.add('loading');
            confirmBtn.disabled = true;
            
            try {
                const res = await fetch('/api/reindex', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ codebase })
                });
                
                const data = await res.json();
                
                if (data.success || data.files_indexed !== undefined) {
                    const count = data.files_indexed || data.files_reindexed || 0;
                    showToast(`Reindexed ${count} files in ${codebase}`, 'success');
                    fetchStatus(); // Refresh status
                } else {
                    showToast(data.error || 'Failed to reindex codebase', 'error');
                }
            } catch (err) {
                showToast('Error: ' + err.message, 'error');
            } finally {
                btn.classList.remove('loading');
                confirmBtn.disabled = false;
            }
        }
        
        // =========== Init ===========
        loadCodebases().then(() => {
            populateValidateCodebase();
            populateActionCodebaseDropdowns();
        });
        initTabFromHash();
    </script>
</body>
</html>"""
    
    return HTMLResponse(content=html)


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
    min_function_count: int | None = None,
    # Phase 1: Implementation Signal Filtering
    calls: list[str] | None = None,
    accesses: list[str] | None = None,
    subscripts: list[str] | None = None,
    # Phase 5: Summary Integration
    include_summaries: bool = False,
    boost_summarized: bool = True
) -> dict[str, Any]:
    """
    Search for relevant memories using semantic similarity, keyword matching, or both.
    
    Args:
        query: Search query for semantic similarity
        max_results: Maximum number of results to return (default 10)
        project_id: Optional filter by project ID
        codebase: Optional codebase name to search (None = search all codebases)
        min_relevance: Minimum relevance score 0-1 (default 0.1)
        search_mode: Search mode - "auto" (default), "semantic", "keyword", "hybrid", or "verify"
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
        calls: Filter by method calls (matches calls:* tags, e.g., ['iloc', 'fit'])
        accesses: Filter by attribute access (matches reads:* tags, e.g., ['bar_index'])
        subscripts: Filter by subscript patterns (matches subscript:* tags, e.g., ['iloc'])
        include_summaries: Include file summary data in results (Phase 5)
        boost_summarized: Apply relevance boost to files with summaries (Phase 5)
    
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
        min_function_count=min_function_count,
        calls=calls,
        accesses=accesses,
        subscripts=subscripts,
        include_summaries=include_summaries,
        boost_summarized=boost_summarized
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


@mcp.tool()
async def memory_summarization_status() -> dict[str, Any]:
    """
    Get the status of background LLM summarization.
    
    Returns:
        Dictionary with summarization status including:
        - enabled: Whether summarization is enabled
        - is_running: Whether summarization is currently active
        - files_queued: Files waiting to be summarized
        - files_completed: Files summarized this session
        - total_summarized: Total files with summaries (persistent)
        - current_file: File currently being processed
        - avg_time_per_file_seconds: Average processing time per file
        - estimated_time_remaining_seconds: Estimated time to complete queue
        - estimated_time_remaining_minutes: Estimated time in minutes
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    return memory_service.get_summarization_status()


@mcp.tool()
async def memory_reindex_codebase(codebase: str) -> dict[str, Any]:
    """
    Force reindexing of a specific codebase to update metadata and heuristics.
    
    Args:
        codebase: Name of the codebase to reindex
    
    Returns:
        Dictionary with reindexing results
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    try:
        result = await memory_service.reindex_codebase_async(codebase)
        return result
    except Exception as e:
        return {"error": f"Failed to reindex codebase '{codebase}': {str(e)}"}


@mcp.tool()
async def memory_queue_codebase_summarization(
    codebase: str, 
    only_missing: bool = True
) -> dict[str, Any]:
    """
    Queue all files from a specific codebase for LLM summarization.
    
    Use this when:
    - Adding a new codebase that hasn't been summarized yet
    - Re-summarizing an existing codebase after making changes
    - Resuming summarization after it was interrupted
    
    Args:
        codebase: Name of the codebase to queue for summarization
        only_missing: If True (default), only queue files without summaries.
                     If False, queue all files for re-summarization.
    
    Returns:
        Dictionary with queue results:
        - success: Whether the operation succeeded
        - message: Human-readable status message
        - files_queued: Number of files added to the queue
        - files_skipped: Number of files skipped (already summarized or excluded)
        - total_queue_size: Current total queue size
    
    Example:
        # Queue new codebase for first-time summarization
        memory_queue_codebase_summarization("my-new-project")
        
        # Re-summarize entire codebase
        memory_queue_codebase_summarization("my-project", only_missing=False)
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    try:
        result = await memory_service.queue_codebase_for_summarization(codebase, only_missing)
        return result
    except Exception as e:
        return {"error": f"Failed to queue codebase '{codebase}' for summarization: {str(e)}"}


@mcp.tool()
async def memory_method_relationships(
    method: str,
    codebase: str | None = None,
    relationship: str = "all"
) -> dict[str, Any]:
    """
    Query method call relationships (callers and callees) for a specific method.
    
    This enables "what calls X?" and "what does X call?" queries using the
    method call graph built during indexing.
    
    Args:
        method: The method name to query. Can be:
            - Simple name: "process_data" - matches any method with this name
            - Qualified name: "MyClass.process_data" - matches exact qualified name
        codebase: Codebase to search in (None = uses first/default codebase)
        relationship: Type of relationships to return:
            - "callers": Only methods that call this method
            - "callees": Only methods called by this method
            - "all" (default): Both callers and callees
    
    Returns:
        Dictionary with:
        - method: The queried method name
        - codebase: The codebase searched
        - callers: List of calling methods (if relationship is "callers" or "all")
        - callees: List of called methods (if relationship is "callees" or "all")
        - stats: Summary counts (caller_count, callee_count)
        
        Each caller/callee entry contains:
        - name: The qualified method name
        - file: Path to the source file
        - line: Line number where method is defined
        - class_name: Containing class (if any)
    
    Example:
        # Find what calls _generate_features
        memory_method_relationships(
            method="_generate_features",
            codebase="options-ml-trader",
            relationship="callers"
        )
        
        # Find what process_data calls
        memory_method_relationships(
            method="MyClass.process_data",
            relationship="callees"
        )
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    # Validate relationship parameter
    valid_relationships = {"callers", "callees", "all"}
    if relationship not in valid_relationships:
        return {
            "error": f"Invalid relationship '{relationship}'. Must be one of: {', '.join(valid_relationships)}"
        }
    
    # Determine codebase to use
    target_codebase = codebase
    if not target_codebase:
        # Use first available codebase
        codebases = memory_service.list_codebases()
        if not codebases:
            return {"error": "No codebases configured"}
        target_codebase = codebases[0]["name"]
    
    # Check if codebase exists and has a call graph
    call_graph = memory_service.get_call_graph(target_codebase)
    if not call_graph:
        return {
            "error": f"No call graph available for codebase '{target_codebase}'. "
                     "The codebase may not be indexed yet or has no method details."
        }
    
    # Build the response
    result: dict[str, Any] = {
        "method": method,
        "codebase": target_codebase,
        "stats": {}
    }
    
    # Get callers if requested
    if relationship in ("callers", "all"):
        callers_raw = memory_service.get_method_callers(method, target_codebase)
        callers = [
            {
                "name": c["qualified_name"],
                "file": c["file_path"],
                "line": c["line_number"],
                "class_name": c.get("class_name")
            }
            for c in callers_raw
        ]
        result["callers"] = callers
        result["stats"]["caller_count"] = len(callers)
    
    # Get callees if requested
    if relationship in ("callees", "all"):
        callees_raw = memory_service.get_method_callees(method, target_codebase)
        callees = [
            {
                "name": c["qualified_name"],
                "file": c["file_path"],
                "line": c["line_number"],
                "class_name": c.get("class_name")
            }
            for c in callees_raw
        ]
        result["callees"] = callees
        result["stats"]["callee_count"] = len(callees)
    
    # Check if method was found at all
    callers_count = result["stats"].get("caller_count", 0)
    callees_count = result["stats"].get("callee_count", 0)
    
    if callers_count == 0 and callees_count == 0:
        # Method might not exist or might be isolated
        # Check if it exists in the graph
        method_exists = False
        if '.' in method:
            method_exists = call_graph.get_node(method) is not None
        else:
            # Check for any method with this name
            for node in call_graph.nodes.values():
                if node.method_name == method:
                    method_exists = True
                    break
        
        if not method_exists:
            result["warning"] = f"Method '{method}' not found in call graph for '{target_codebase}'"
        else:
            result["info"] = f"Method '{method}' exists but has no recorded call relationships"
    
    return result


@mcp.tool()
async def memory_invalidate_summaries(
    codebase: str | None = None
) -> dict[str, Any]:
    """
    Invalidate all existing summaries to force re-summarization.
    
    This is useful when:
    - The summary schema has changed (new fields, different format)
    - Summaries need to be regenerated with a different model
    - Summaries are corrupted or inconsistent
    
    The operation clears:
    1. All entries from the summary index (file -> summary mapping)
    2. All summary chunks from the main collection (searchable summary text)
    3. Updates the schema version to current
    
    After invalidation:
    - Use memory_queue_codebase_summarization to re-summarize files
    - Or wait for the background summarizer to pick up the files
    
    Args:
        codebase: Optional codebase name to invalidate (None = all codebases)
    
    Returns:
        Dictionary with:
        - success: Whether the operation succeeded
        - total_summaries_cleared: Total summary entries removed
        - total_chunks_removed: Total summary chunks removed from main collection
        - codebases: Per-codebase breakdown
    
    Example:
        # Invalidate all summaries in all codebases
        memory_invalidate_summaries()
        
        # Invalidate summaries for a specific codebase
        memory_invalidate_summaries(codebase="my-project")
    """
    if not memory_service:
        return {"error": "Memory service not initialized"}
    
    try:
        result = await memory_service.invalidate_summaries_async(codebase)
        return result
    except Exception as e:
        return {"error": f"Failed to invalidate summaries: {str(e)}"}


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
    
    # Suppress noisy httpx logging (ChromaDB HTTP client)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

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
    
    # Start ChromaDB server if in HTTP mode
    global chroma_manager
    chroma_manager = None
    if config.chroma_mode == "http":
        chroma_server_path = config.chroma_server_path or os.path.join(
            config.persist_directory or os.path.expanduser("~/.conductor-memory"),
            "chroma-server"
        )
        chroma_manager = ChromaServerManager(
            host=config.chroma_host,
            port=config.chroma_port,
            data_path=chroma_server_path
        )
        
        # Start ChromaDB synchronously before creating MemoryService
        # Use start_sync() to avoid async task issues with asyncio.run()
        if not chroma_manager.start_sync():
            logger.error("Failed to start ChromaDB server, falling back to embedded mode")
            config.chroma_mode = "embedded"
            chroma_manager = None
        else:
            logger.info(f"ChromaDB server running at http://{config.chroma_host}:{config.chroma_port}")
    
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
    logger.info(f"Dashboard available at: http://{args.host}:{args.port}/")
    logger.info("Configure OpenCode with:")
    logger.info(f'  "type": "remote", "url": "http://{args.host}:{args.port}/sse"')
    
    # Start the server with background tasks
    import uvicorn
    import asyncio
    
    async def run_server():
        global chroma_manager
        
        # Start file watchers and background summarizer in the async context
        if config.enable_file_watcher:
            await memory_service.start_file_watchers_async()
        
        # Start background summarizer (with proper callback system)
        # Use basic status to avoid expensive ChromaDB queries at startup
        summarization_status = memory_service.get_summarization_status_basic()
        if summarization_status.get("enabled") and summarization_status.get("llm_enabled"):
            logger.info("[Summarization] Starting background summarizer...")
            await memory_service.start_background_summarizer_async()
        
        # Start ChromaDB monitoring now that we're in the async context
        if chroma_manager:
            chroma_manager.start_monitoring()
            logger.info("ChromaDB health monitoring started")
        
        # Get the ASGI app from FastMCP
        app = mcp.sse_app()
        
        # Run uvicorn
        config_uvicorn = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config_uvicorn)
        
        try:
            await server.serve()
        finally:
            # Cleanup on shutdown
            if chroma_manager:
                logger.info("Shutting down ChromaDB server...")
                await chroma_manager.stop()
    
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
