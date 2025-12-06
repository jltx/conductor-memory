#!/usr/bin/env python3
"""
Unified Memory Server - HTTP + TCP MCP server for multi-client access

This server provides:
1. HTTP API (port 9800) - REST endpoints for curl/debugging
2. TCP MCP (port 9801) - JSON-RPC for MCP bridge connections

Clients connect via mcp_bridge.py which proxies stdio to TCP.

Usage:
    # Start the daemon (HTTP + TCP)
    python src/memory_server.py

    # With explicit config
    python src/memory_server.py --config memory_server_config.json
"""

# Suppress TensorFlow warnings before any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import asyncio
import json
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Optional

# Lightweight imports only - no TensorFlow/ChromaDB
from ..config.server import ServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

memory_service: Optional[Any] = None
_config: Optional[ServerConfig] = None
_initialization_done = False
_initialization_lock = threading.Lock()


def initialize_memory_service(config: ServerConfig):
    """Initialize the memory service (must be called before event loop starts)"""
    global memory_service, _initialization_done, _config
    
    _config = config
    
    if _initialization_done:
        return
    
    with _initialization_lock:
        if _initialization_done:
            return
        
        if config.get_enabled_codebases():
            logger.info("Initializing memory service...")
            from ..service.memory_service import MemoryService
            memory_service = MemoryService(config)
            memory_service.initialize()
            logger.info("Memory service initialized")
        
        _initialization_done = True


def ensure_initialized():
    """Check that memory service is initialized (no-op, initialization done at startup)"""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tool Functions
# ─────────────────────────────────────────────────────────────────────────────

def tool_memory_search(
    query: str,
    max_results: int = 10,
    project_id: str | None = None,
    codebase: str | None = None,
    min_relevance: float = 0.1,
    domain_boosts: dict[str, float] | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None
) -> dict[str, Any]:
    """Search for relevant memories using semantic similarity with optional boosting and filtering."""
    ensure_initialized()
    if not memory_service:
        return {"error": "Memory service not initialized", "results": []}
    return memory_service.search(
        query=query,
        codebase=codebase,
        max_results=max_results,
        project_id=project_id,
        domain_boosts=domain_boosts,
        include_tags=include_tags,
        exclude_tags=exclude_tags
    )


def tool_memory_store(
    content: str,
    project_id: str = "default",
    codebase: str | None = None,
    role: str = "user",
    tags: list[str] | None = None,
    pin: bool = False,
    source: str = "opencode"
) -> dict[str, Any]:
    """Store a new memory chunk for later retrieval."""
    ensure_initialized()
    if not memory_service:
        return {"error": "Memory service not initialized", "success": False}
    return memory_service.store(
        content=content,
        project_id=project_id,
        codebase=codebase,
        role=role,
        tags=tags,
        pin=pin,
        source=source
    )


def tool_memory_status() -> dict[str, Any]:
    """Get the current status of the memory system."""
    ensure_initialized()
    if not memory_service:
        return {"error": "Memory service not initialized"}
    return memory_service.get_status()


def tool_memory_prune(
    project_id: str | None = None,
    max_age_days: int = 30
) -> dict[str, Any]:
    """Prune obsolete memories based on age and relevance."""
    return {"message": "Pruning not yet implemented", "pruned": 0}


# Tool registry for JSON-RPC dispatch
TOOLS = {
    "memory_search": {
        "function": tool_memory_search,
        "description": "Search for relevant memories using semantic similarity with optional boosting and filtering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for semantic similarity"},
                "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results"},
                "project_id": {"type": ["string", "null"], "description": "Optional filter by project ID"},
                "codebase": {"type": ["string", "null"], "description": "Optional codebase name to search"},
                "min_relevance": {"type": "number", "default": 0.1, "description": "Minimum relevance score 0-1"},
                "domain_boosts": {
                    "type": ["object", "null"], 
                    "description": "Per-query domain boost overrides (e.g., {'class': 1.5, 'test': 0.5})",
                    "additionalProperties": {"type": "number"}
                },
                "include_tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Include only results with these tags (supports prefix:* patterns, e.g., 'domain:*', 'ext:.py')"
                },
                "exclude_tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Exclude results with these tags (supports prefix:* patterns, e.g., 'domain:test')"
                }
            },
            "required": ["query"]
        }
    },
    "memory_store": {
        "function": tool_memory_store,
        "description": "Store a new memory chunk for later retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The text content to store"},
                "project_id": {"type": "string", "default": "default", "description": "Project identifier"},
                "codebase": {"type": ["string", "null"], "description": "Codebase to store in"},
                "role": {"type": "string", "default": "user", "description": "Role of the memory"},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Optional tags"},
                "pin": {"type": "boolean", "default": False, "description": "Pin to prevent pruning"},
                "source": {"type": "string", "default": "opencode", "description": "Source of the memory"}
            },
            "required": ["content"]
        }
    },
    "memory_status": {
        "function": tool_memory_status,
        "description": "Get the current status of the memory system.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    "memory_prune": {
        "function": tool_memory_prune,
        "description": "Prune obsolete memories based on age and relevance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": ["string", "null"], "description": "Optional project ID filter"},
                "max_age_days": {"type": "integer", "default": 30, "description": "Maximum age in days"}
            }
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# MCP JSON-RPC Protocol Handler
# ─────────────────────────────────────────────────────────────────────────────

class MCPProtocolHandler:
    """Handles MCP JSON-RPC protocol messages"""
    
    def __init__(self):
        self.initialized = False
    
    async def handle_message(self, message: dict) -> dict | None:
        """Process a JSON-RPC message and return response (or None for notifications)"""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})
        
        # Notifications (no id) don't get responses
        is_notification = msg_id is None
        
        try:
            result = await self._dispatch(method, params)
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as e:
            logger.exception(f"Error handling {method}")
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(e)}
            }
    
    async def _dispatch(self, method: str, params: dict) -> Any:
        """Dispatch method to appropriate handler"""
        
        # Initialize handshake
        if method == "initialize":
            self.initialized = True
            return {
                "_meta": None,
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "experimental": None,
                    "logging": None,
                    "prompts": None,
                    "resources": None,
                    "tools": {"listChanged": None},
                    "completions": None,
                    "tasks": None
                },
                "serverInfo": {
                    "name": "conductor_memory",
                    "title": None,
                    "version": "2.0.0",
                    "websiteUrl": None,
                    "icons": None
                },
                "instructions": None
            }
        
        # Initialized notification (no response needed)
        if method == "notifications/initialized":
            return None
        
        # List available tools
        if method == "tools/list":
            tools = []
            for name, spec in TOOLS.items():
                tools.append({
                    "name": name,
                    "title": None,
                    "description": spec["description"],
                    "inputSchema": spec["inputSchema"],
                    "outputSchema": None,
                    "icons": None,
                    "annotations": None,
                    "_meta": None,
                    "execution": None
                })
            return {"_meta": None, "nextCursor": None, "tools": tools}
        
        # Call a tool - run in thread pool since memory_service uses asyncio.run()
        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            
            if tool_name not in TOOLS:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            tool_func = TOOLS[tool_name]["function"]
            # Run sync function in thread pool to avoid event loop conflicts
            result = await asyncio.to_thread(tool_func, **tool_args)
            
            return {
                "_meta": None,
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2), "annotations": None, "_meta": None}
                ],
                "structuredContent": None,
                "isError": False
            }
        
        # Ping
        if method == "ping":
            return {"_meta": None}
        
        # Unknown method
        raise ValueError(f"Unknown method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# TCP Server for MCP
# ─────────────────────────────────────────────────────────────────────────────

class TCPMCPServer:
    """TCP server that accepts MCP JSON-RPC connections"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9801):
        self.host = host
        self.port = port
        self.server = None
        self.clients: set = set()
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single client connection"""
        addr = writer.get_extra_info('peername')
        logger.info(f"TCP MCP client connected: {addr}")
        self.clients.add(writer)
        
        handler = MCPProtocolHandler()
        buffer = ""
        
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                
                buffer += data.decode('utf-8')
                
                # Process complete lines (JSON-RPC messages are newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        message = json.loads(line)
                        response = await handler.handle_message(message)
                        
                        if response:
                            response_str = json.dumps(response) + '\n'
                            writer.write(response_str.encode('utf-8'))
                            await writer.drain()
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from client: {e}")
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"}
                        }
                        writer.write((json.dumps(error_response) + '\n').encode('utf-8'))
                        await writer.drain()
        
        except asyncio.CancelledError:
            pass
        except ConnectionResetError:
            # Client disconnected - this is normal
            pass
        except OSError as e:
            # WinError 64 and similar - client disconnected
            if e.winerror in (64, 10054):  # Network name no longer available, Connection reset
                pass
            else:
                logger.exception(f"Error handling client {addr}: {e}")
        except Exception as e:
            logger.exception(f"Error handling client {addr}: {e}")
        finally:
            self.clients.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except:
                pass
            logger.info(f"TCP MCP client disconnected: {addr}")
    
    async def start(self):
        """Start the TCP server"""
        self.server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"TCP MCP server listening on {self.host}:{self.port}")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def stop(self):
        """Stop the TCP server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all client connections
        for writer in list(self.clients):
            writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Server (FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

def create_http_app():
    """Create the FastAPI application"""
    from datetime import datetime
    from typing import List
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    
    class SearchRequest(BaseModel):
        query: str = Field(..., description="Search query")
        project_id: Optional[str] = Field(None)
        codebase: Optional[str] = Field(None)
        max_results: int = Field(10)
        min_relevance: float = Field(0.1)
        domain_boosts: Optional[dict] = Field(None, description="Per-query domain boost overrides")
        include_tags: Optional[List[str]] = Field(None, description="Include only results with these tags")
        exclude_tags: Optional[List[str]] = Field(None, description="Exclude results with these tags")

    class StoreRequest(BaseModel):
        content: str = Field(...)
        project_id: str = Field("default")
        codebase: Optional[str] = Field(None)
        role: str = Field("user")
        tags: List[str] = Field(default_factory=list)
        pin: bool = Field(False)
        source: str = Field("api")

    app = FastAPI(
        title="MCP Memory Server",
        description="HTTP API for semantic memory search and codebase indexing",
        version="2.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/status")
    async def get_status():
        ensure_initialized()
        if not memory_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return memory_service.get_status()

    @app.get("/codebases")
    async def list_codebases():
        ensure_initialized()
        if not memory_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return {"codebases": memory_service.list_codebases()}

    @app.get("/codebases/{codebase_name}/status")
    async def get_codebase_status(codebase_name: str):
        ensure_initialized()
        if not memory_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        status = memory_service.get_codebase_status(codebase_name)
        if not status:
            raise HTTPException(status_code=404, detail=f"Codebase not found: {codebase_name}")
        return status

    @app.post("/codebases/{codebase_name}/reindex")
    async def reindex_codebase(codebase_name: str, background_tasks: BackgroundTasks):
        ensure_initialized()
        if not memory_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        status = memory_service.get_codebase_status(codebase_name)
        if not status:
            raise HTTPException(status_code=404, detail=f"Codebase not found: {codebase_name}")
        background_tasks.add_task(memory_service.reindex_codebase_async, codebase_name)
        return {"message": f"Re-indexing started for codebase: {codebase_name}"}

    @app.post("/search")
    async def search_memories(request: SearchRequest):
        result = tool_memory_search(
            query=request.query,
            codebase=request.codebase,
            max_results=request.max_results,
            project_id=request.project_id,
            domain_boosts=request.domain_boosts,
            include_tags=request.include_tags,
            exclude_tags=request.exclude_tags
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result

    @app.post("/store")
    async def store_memory(request: StoreRequest):
        return tool_memory_store(
            content=request.content,
            project_id=request.project_id,
            codebase=request.codebase,
            role=request.role,
            tags=request.tags,
            pin=request.pin,
            source=request.source
        )

    @app.post("/prune")
    async def prune_memories(project_id: Optional[str] = None, max_age_days: int = 30):
        return tool_memory_prune(project_id=project_id, max_age_days=max_age_days)

    return app


async def run_http_server(host: str, port: int, log_level: str):
    """Run the HTTP server"""
    import uvicorn
    app = create_http_app()
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=False
    )
    server = uvicorn.Server(config)
    await server.serve()


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(config: ServerConfig, http_port: int, tcp_port: int, log_level: str):
    """Main async entry point - runs both HTTP and TCP servers"""
    # Start file watchers in the main event loop
    # (they couldn't be started during initialize() because that runs in a temp loop)
    if memory_service and config.enable_file_watcher:
        await memory_service.start_file_watchers_async()
    
    # Create servers
    tcp_server = TCPMCPServer(host=config.host, port=tcp_port)
    
    # Run both servers concurrently
    logger.info(f"Starting HTTP server on {config.host}:{http_port}")
    logger.info(f"Starting TCP MCP server on {config.host}:{tcp_port}")
    
    try:
        await asyncio.gather(
            run_http_server(config.host, http_port, log_level),
            tcp_server.start()
        )
    except asyncio.CancelledError:
        logger.info("Shutting down servers...")
        await tcp_server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Memory Server - HTTP + TCP MCP daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with auto-detected config
  python src/memory_server.py

  # Start with explicit config
  python src/memory_server.py --config memory_server_config.json

  # Custom ports
  python src/memory_server.py --http-port 9800 --tcp-port 9801
        """
    )
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--codebase-path", type=str, help="Single codebase path")
    parser.add_argument("--codebase-name", type=str, default="default", help="Codebase name")
    parser.add_argument("--persist-dir", type=str, default="./data/chroma", help="Storage directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--http-port", type=int, default=9800, help="HTTP server port")
    parser.add_argument("--tcp-port", type=int, default=9801, help="TCP MCP server port")
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
    if not config_path and not args.codebase_path:
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
    
    if config_path:
        try:
            config = ServerConfig.from_file(config_path)
            config.host = args.host
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
        config.host = args.host
    else:
        logger.warning(f"No config file found. Create {DEFAULT_HOME_CONFIG} or use --codebase-path")
        config = ServerConfig()
        config.persist_directory = args.persist_dir
        config.host = args.host
    
    # Initialize memory service BEFORE starting the event loop
    # This avoids asyncio.run() conflicts since MemoryService uses it internally
    initialize_memory_service(config)
    
    # Handle graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def shutdown_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    try:
        loop.run_until_complete(main_async(config, args.http_port, args.tcp_port, args.log_level))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
