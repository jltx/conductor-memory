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
    """Get summarization status as JSON."""
    if memory_service:
        return JSONResponse(memory_service.get_summarization_status())
    return JSONResponse({"error": "Service not initialized"}, status_code=503)


# Indexing status API endpoint
@mcp.custom_route("/api/status", methods=["GET"])
async def indexing_status_api(request):
    """Get full service status as JSON."""
    if memory_service:
        return JSONResponse(memory_service.get_status())
    return JSONResponse({"error": "Service not initialized"}, status_code=503)


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
            include_summaries=body.get("include_summaries", False),
            boost_summarized=body.get("boost_summarized", True)
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# List indexed files API endpoint
@mcp.custom_route("/api/files", methods=["GET"])
async def api_list_files(request):
    """List indexed files with pagination."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        params = request.query_params
        result = await memory_service.list_indexed_files_async(
            codebase=params.get("codebase"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0)),
            search_filter=params.get("filter")
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
    """List all configured codebases."""
    if not memory_service:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    try:
        codebases = memory_service.list_codebases()
        return JSONResponse({"codebases": codebases})
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
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #eee; 
            min-height: 100vh;
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
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
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
        
        /* Search results */
        .results-header { 
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 10px; color: #888; font-size: 13px;
        }
        .result-item {
            background: #0f3460; border-radius: 6px; padding: 12px; margin-bottom: 10px;
            border-left: 3px solid #00d4ff;
        }
        .result-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .result-file { font-family: monospace; color: #00d4ff; font-size: 13px; }
        .result-score { 
            background: #16213e; padding: 2px 8px; border-radius: 10px;
            font-size: 12px; color: #00ff88;
        }
        .result-tags { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }
        .tag { 
            background: #16213e; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; color: #888;
        }
        .result-content {
            font-family: monospace; font-size: 12px; color: #ccc;
            white-space: pre-wrap; overflow: hidden;
            max-height: 80px; position: relative;
        }
        .result-content.expanded { max-height: none; }
        .expand-btn {
            color: #00d4ff; cursor: pointer; font-size: 12px;
            margin-top: 5px; display: inline-block;
        }
        
        /* Browse tab styles */
        .browse-controls { 
            display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; align-items: center;
        }
        .browse-controls select {
            padding: 8px 12px; background: #0f3460; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #eee; font-size: 14px;
        }
        .filter-input {
            padding: 8px 12px; background: #0f3460; border: 1px solid #1a3a6e;
            border-radius: 4px; color: #eee; font-size: 14px; width: 200px;
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
        
        /* Details panel */
        .details-panel {
            background: #16213e; border-radius: 8px; padding: 20px;
            margin-top: 15px; border: 1px solid #0f3460; display: none;
        }
        .details-panel.open { display: block; }
        .details-header { 
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #0f3460;
        }
        .details-title { font-family: monospace; color: #00d4ff; }
        .close-btn { 
            background: none; border: none; color: #888; cursor: pointer;
            font-size: 20px; line-height: 1;
        }
        .close-btn:hover { color: #ff6b6b; }
        .details-meta { display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap; }
        .meta-item { font-size: 13px; }
        .meta-label { color: #888; }
        .meta-value { color: #eee; font-family: monospace; }
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
        
        /* Loading & empty states */
        .loading { text-align: center; padding: 40px; color: #888; }
        .empty-state { text-align: center; padding: 40px; color: #888; }
        
        /* Refresh note */
        .refresh-note { text-align: center; color: #555; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Conductor Memory</h1>
        <div class="tabs">
            <button class="tab active" data-tab="status">Status</button>
            <button class="tab" data-tab="search">Search</button>
            <button class="tab" data-tab="browse">Browse</button>
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
                        <span id="advanced-arrow">&#9654;</span> Advanced Filters
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
                            <input type="text" id="filter-include-tags" placeholder="decision, lesson" />
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
            
            <div class="card">
                <div id="browse-content">
                    <div class="loading">Loading...</div>
                </div>
                <div id="browse-pagination" class="pagination"></div>
            </div>
            
            <div id="details-panel" class="details-panel">
                <div class="details-header">
                    <span class="details-title" id="details-title">File Details</span>
                    <button class="close-btn" onclick="closeDetails()">&times;</button>
                </div>
                <div id="details-content"></div>
            </div>
        </div>
    </div>
    
    <script>
        // =========== State ===========
        let codebases = [];
        let currentBrowseOffset = 0;
        const BROWSE_LIMIT = 50;
        let statusInterval = null;
        
        // =========== Tab Management ===========
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab + '-tab').classList.add('active');
                
                // Start/stop status refresh based on active tab
                if (tab.dataset.tab === 'status') {
                    startStatusRefresh();
                } else {
                    stopStatusRefresh();
                }
                
                // Load data for browse tab
                if (tab.dataset.tab === 'browse') {
                    loadBrowseData();
                }
            });
        });
        
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
            
            document.getElementById('summary-stats').innerHTML = `
                <div class="stat"><div class="stat-value ${statusClass}">${statusText}</div><div class="stat-label">Status</div></div>
                <div class="stat"><div class="stat-value">${data.total_summarized || 0}</div><div class="stat-label">Summarized</div></div>
                <div class="stat"><div class="stat-value">${data.files_queued || 0}</div><div class="stat-label">In Queue</div></div>
                <div class="stat"><div class="stat-value">${data.files_completed || 0}</div><div class="stat-label">This Session</div></div>
                <div class="stat"><div class="stat-value">${data.files_skipped || 0}</div><div class="stat-label">Skipped</div></div>
                <div class="stat"><div class="stat-value">${data.files_failed || 0}</div><div class="stat-label">Failed</div></div>
                <div class="stat"><div class="stat-value">${avgTimeText}</div><div class="stat-label">Avg Time/File</div></div>
                <div class="stat"><div class="stat-value">${estRemainingText}</div><div class="stat-label">Est. Remaining</div></div>
            `;
            
            const total = (data.files_completed || 0) + (data.files_queued || 0);
            const pct = total > 0 ? Math.round((data.files_completed / total) * 100) : 0;
            
            document.getElementById('progress-section').innerHTML = data.is_running && total > 0 ? `
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: #888; margin-top: 10px;">
                    <span>Progress (${data.files_completed}/${total})</span><span>${pct}% • ${estRemainingText} remaining</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width: ${pct}%"></div></div>
            ` : '';
            
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
                
                searchSelect.innerHTML = '<option value="">All</option>';
                browseSelect.innerHTML = '';
                
                codebases.forEach(cb => {
                    searchSelect.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
                    browseSelect.innerHTML += `<option value="${cb.name}">${cb.name}</option>`;
                });
            } catch (err) {
                console.error('Failed to load codebases:', err);
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
            
            const body = {
                query,
                codebase: document.getElementById('search-codebase').value || null,
                max_results: parseInt(document.getElementById('search-limit').value),
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
                    renderSearchResults(data);
                }
            } catch (err) {
                resultsDiv.innerHTML = `<div class="error-msg">Error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Search';
            }
        }
        
        function renderSearchResults(data) {
            const resultsDiv = document.getElementById('search-results');
            const results = data.results || [];
            
            if (results.length === 0) {
                resultsDiv.innerHTML = '<div class="empty-state">No results found</div>';
                return;
            }
            
            let html = `<div class="results-header">
                <span>Found ${data.total_found || results.length} results (${data.query_time_ms}ms) - Mode: ${data.search_mode_used}</span>
            </div>`;
            
            results.forEach((r, idx) => {
                const tags = (r.tags || []).slice(0, 5).map(t => `<span class="tag">${t}</span>`).join('');
                const content = escapeHtml(r.content || r.doc_text || '');
                const preview = content.length > 300 ? content.substring(0, 300) : content;
                const hasMore = content.length > 300;
                
                html += `
                    <div class="result-item">
                        <div class="result-header">
                            <span class="result-file">${r.source || 'unknown'}</span>
                            <span class="result-score">${(r.relevance_score || 0).toFixed(3)}</span>
                        </div>
                        ${tags ? `<div class="result-tags">${tags}</div>` : ''}
                        <div class="result-content" id="result-content-${idx}">${preview}${hasMore ? '...' : ''}</div>
                        ${hasMore ? `<span class="expand-btn" onclick="toggleResultExpand(${idx}, \`${escapeJs(content)}\`)">Show more</span>` : ''}
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
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
        async function loadBrowseData() {
            const view = document.getElementById('browse-view').value;
            const codebase = document.getElementById('browse-codebase').value;
            const filter = document.getElementById('browse-filter').value;
            
            const contentDiv = document.getElementById('browse-content');
            contentDiv.innerHTML = '<div class="loading">Loading...</div>';
            closeDetails();
            
            try {
                let url, data;
                
                if (view === 'files') {
                    url = `/api/files?codebase=${codebase}&limit=${BROWSE_LIMIT}&offset=${currentBrowseOffset}`;
                    if (filter) url += `&filter=${encodeURIComponent(filter)}`;
                    const res = await fetch(url);
                    data = await res.json();
                    renderFilesTable(data);
                } else if (view === 'memories') {
                    url = `/api/memories?limit=${BROWSE_LIMIT}&offset=${currentBrowseOffset}`;
                    if (codebase) url += `&codebase=${codebase}`;
                    const res = await fetch(url);
                    data = await res.json();
                    renderMemoriesTable(data);
                } else if (view === 'summaries') {
                    url = `/api/summaries?codebase=${codebase}&limit=${BROWSE_LIMIT}&offset=${currentBrowseOffset}`;
                    const res = await fetch(url);
                    data = await res.json();
                    renderSummariesTable(data);
                }
            } catch (err) {
                contentDiv.innerHTML = `<div class="error-msg">Error: ${err.message}</div>`;
            }
        }
        
        function renderFilesTable(data) {
            const contentDiv = document.getElementById('browse-content');
            const files = data.files || [];
            
            if (files.length === 0) {
                contentDiv.innerHTML = '<div class="empty-state">No files found</div>';
                renderPagination(0, 0);
                return;
            }
            
            let html = `<table class="data-table">
                <thead><tr>
                    <th>Path</th>
                    <th>Chunks</th>
                    <th>Summary</th>
                    <th>Indexed At</th>
                </tr></thead>
                <tbody>`;
            
            files.forEach(f => {
                const indexedAt = f.indexed_at ? new Date(f.indexed_at).toLocaleString() : '-';
                html += `
                    <tr class="clickable" onclick="loadFileDetails('${data.codebase}', '${escapeJs(f.path)}')">
                        <td style="font-family: monospace; color: #00d4ff;">${f.path}</td>
                        <td>${f.chunk_count}</td>
                        <td><span class="badge ${f.has_summary ? 'badge-yes' : 'badge-no'}">${f.has_summary ? 'Yes' : 'No'}</span></td>
                        <td style="color: #888;">${indexedAt}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            contentDiv.innerHTML = html;
            renderPagination(data.total, data.offset);
        }
        
        function renderMemoriesTable(data) {
            const contentDiv = document.getElementById('browse-content');
            const memories = data.memories || [];
            
            if (memories.length === 0) {
                contentDiv.innerHTML = '<div class="empty-state">No memories found</div>';
                renderPagination(0, 0);
                return;
            }
            
            let html = `<table class="data-table">
                <thead><tr>
                    <th>Type</th>
                    <th>Content</th>
                    <th>Tags</th>
                    <th>Codebase</th>
                    <th>Created</th>
                </tr></thead>
                <tbody>`;
            
            memories.forEach(m => {
                const badgeClass = m.type === 'decision' ? 'badge-decision' : 
                                   m.type === 'lesson' ? 'badge-lesson' : 'badge-conversation';
                const tags = (m.tags || []).slice(0, 3).map(t => `<span class="tag">${t}</span>`).join('');
                const created = m.created_at ? new Date(m.created_at).toLocaleString() : '-';
                
                html += `
                    <tr class="clickable" onclick="showMemoryDetails(${JSON.stringify(m).replace(/"/g, '&quot;')})">
                        <td><span class="badge ${badgeClass}">${m.type}</span></td>
                        <td>${m.content_preview}</td>
                        <td>${tags}</td>
                        <td>${m.codebase}</td>
                        <td style="color: #888;">${created}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            contentDiv.innerHTML = html;
            renderPagination(data.total, data.offset);
        }
        
        function renderSummariesTable(data) {
            const contentDiv = document.getElementById('browse-content');
            const summaries = data.summaries || [];
            
            if (summaries.length === 0) {
                contentDiv.innerHTML = '<div class="empty-state">No summaries found</div>';
                renderPagination(0, 0);
                return;
            }
            
            let html = `<table class="data-table">
                <thead><tr>
                    <th>File</th>
                    <th>Pattern</th>
                    <th>Domain</th>
                    <th>Model</th>
                    <th>Summarized At</th>
                </tr></thead>
                <tbody>`;
            
            summaries.forEach(s => {
                const summarizedAt = s.summarized_at ? new Date(s.summarized_at).toLocaleString() : '-';
                html += `
                    <tr class="clickable" onclick="showSummaryDetails(${JSON.stringify(s).replace(/"/g, '&quot;')})">
                        <td style="font-family: monospace; color: #00d4ff;">${s.file_path}</td>
                        <td><span class="tag">${s.pattern || '-'}</span></td>
                        <td><span class="tag">${s.domain || '-'}</span></td>
                        <td style="color: #888;">${s.model || '-'}</td>
                        <td style="color: #888;">${summarizedAt}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            contentDiv.innerHTML = html;
            renderPagination(data.total, data.offset);
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
        
        // Details panels
        async function loadFileDetails(codebase, filePath) {
            const panel = document.getElementById('details-panel');
            const content = document.getElementById('details-content');
            
            panel.classList.add('open');
            document.getElementById('details-title').textContent = filePath;
            content.innerHTML = '<div class="loading">Loading...</div>';
            
            try {
                const res = await fetch(`/api/file-details?codebase=${codebase}&path=${encodeURIComponent(filePath)}`);
                const data = await res.json();
                
                if (data.error) {
                    content.innerHTML = `<div class="error-msg">${data.error}</div>`;
                    return;
                }
                
                let html = `<div class="details-meta">
                    <div class="meta-item"><span class="meta-label">Codebase:</span> <span class="meta-value">${data.codebase}</span></div>
                    <div class="meta-item"><span class="meta-label">Chunks:</span> <span class="meta-value">${data.chunk_count}</span></div>
                    <div class="meta-item"><span class="meta-label">Indexed:</span> <span class="meta-value">${data.indexed_at ? new Date(data.indexed_at).toLocaleString() : '-'}</span></div>
                    <div class="meta-item"><span class="meta-label">Hash:</span> <span class="meta-value">${(data.content_hash || '').substring(0, 16)}...</span></div>
                </div>`;
                
                if (data.summary) {
                    html += `<div class="summary-section">
                        <div class="summary-title">LLM Summary (${data.summary.model} - ${data.summary.pattern})</div>
                        <div class="summary-content">${escapeHtml(data.summary.content)}</div>
                    </div>`;
                }
                
                html += '<h3 style="color: #888; font-size: 13px; margin-bottom: 10px;">Chunks</h3><div class="chunk-list">';
                
                (data.chunks || []).forEach((chunk, idx) => {
                    const chunkContent = escapeHtml(chunk.content || '');
                    const preview = chunkContent.length > 200 ? chunkContent.substring(0, 200) + '...' : chunkContent;
                    
                    html += `
                        <div class="chunk-item">
                            <div class="chunk-header">
                                <span>Chunk ${idx + 1}</span>
                                <span>${chunk.memory_type}</span>
                            </div>
                            <div class="chunk-content" id="chunk-${idx}">${preview}</div>
                            ${chunkContent.length > 200 ? `<span class="expand-btn" onclick="toggleChunkExpand(${idx}, \`${escapeJs(chunkContent)}\`)">Show more</span>` : ''}
                        </div>
                    `;
                });
                
                html += '</div>';
                content.innerHTML = html;
            } catch (err) {
                content.innerHTML = `<div class="error-msg">Error: ${err.message}</div>`;
            }
        }
        
        function showMemoryDetails(memory) {
            const panel = document.getElementById('details-panel');
            const content = document.getElementById('details-content');
            
            panel.classList.add('open');
            document.getElementById('details-title').textContent = `Memory: ${memory.type}`;
            
            const tags = (memory.tags || []).map(t => `<span class="tag">${t}</span>`).join(' ');
            
            content.innerHTML = `
                <div class="details-meta">
                    <div class="meta-item"><span class="meta-label">ID:</span> <span class="meta-value">${memory.id}</span></div>
                    <div class="meta-item"><span class="meta-label">Type:</span> <span class="meta-value">${memory.type}</span></div>
                    <div class="meta-item"><span class="meta-label">Codebase:</span> <span class="meta-value">${memory.codebase}</span></div>
                    <div class="meta-item"><span class="meta-label">Source:</span> <span class="meta-value">${memory.source}</span></div>
                    <div class="meta-item"><span class="meta-label">Created:</span> <span class="meta-value">${memory.created_at ? new Date(memory.created_at).toLocaleString() : '-'}</span></div>
                </div>
                ${tags ? `<div style="margin-bottom: 15px;">${tags}</div>` : ''}
                <div style="background: #0f3460; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 13px; white-space: pre-wrap;">${escapeHtml(memory.content)}</div>
            `;
        }
        
        function showSummaryDetails(summary) {
            const panel = document.getElementById('details-panel');
            const content = document.getElementById('details-content');
            
            panel.classList.add('open');
            document.getElementById('details-title').textContent = summary.file_path;
            
            content.innerHTML = `
                <div class="details-meta">
                    <div class="meta-item"><span class="meta-label">Pattern:</span> <span class="meta-value">${summary.pattern || '-'}</span></div>
                    <div class="meta-item"><span class="meta-label">Domain:</span> <span class="meta-value">${summary.domain || '-'}</span></div>
                    <div class="meta-item"><span class="meta-label">Model:</span> <span class="meta-value">${summary.model || '-'}</span></div>
                    <div class="meta-item"><span class="meta-label">Summarized:</span> <span class="meta-value">${summary.summarized_at ? new Date(summary.summarized_at).toLocaleString() : '-'}</span></div>
                </div>
                <div class="summary-section">
                    <div class="summary-content">${escapeHtml(summary.content)}</div>
                </div>
            `;
        }
        
        function toggleChunkExpand(idx, fullContent) {
            const el = document.getElementById(`chunk-${idx}`);
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
        
        function closeDetails() {
            document.getElementById('details-panel').classList.remove('open');
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
        
        // =========== Init ===========
        loadCodebases();
        startStatusRefresh();
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
    logger.info(f"Dashboard available at: http://{args.host}:{args.port}/")
    logger.info("Configure OpenCode with:")
    logger.info(f'  "type": "remote", "url": "http://{args.host}:{args.port}/sse"')
    
    # Start the server with background tasks
    import uvicorn
    import asyncio
    
    async def run_server():
        # Start file watchers and background summarizer in the async context
        if config.enable_file_watcher:
            await memory_service.start_file_watchers_async()
        
        # Start background summarizer if enabled
        summarization_status = memory_service.get_summarization_status()
        if summarization_status.get("enabled") and summarization_status.get("llm_enabled"):
            logger.info("[Summarization] Starting background summarizer...")
            await memory_service.start_background_summarizer_async()
        
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
        await server.serve()
    
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
