#!/usr/bin/env python3
"""
Startup script for MCP Memory Server

This script provides an easy way to start the memory server with common configurations.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_default_config_path() -> Path:
    """Get the default config file path.

    Priority order:
    1. CONDUCTOR_MEMORY_CONFIG environment variable
    2. ~/.conductor-memory/config.json (documented default)
    3. ./memory_server_config.json (legacy/backwards compat)
    """
    # Check environment variable first
    env_config = os.environ.get("CONDUCTOR_MEMORY_CONFIG")
    if env_config:
        env_path = Path(env_config)
        if env_path.exists():
            return env_path

    # Then check standard locations
    locations = [
        Path.home() / ".conductor-memory" / "config.json",
        Path.cwd() / "memory_server_config.json",
    ]
    for loc in locations:
        if loc.exists():
            return loc
    return Path.home() / ".conductor-memory" / "config.json"


def get_default_persist_dir() -> Path:
    """Get the default persist directory"""
    return Path.home() / ".conductor-memory" / "data"


def main():
    parser = argparse.ArgumentParser(description="Start MCP Memory Server")
    parser.add_argument("--http-port", type=int, default=9800, help="HTTP server port (default: 9800)")
    parser.add_argument("--tcp-port", type=int, default=9801, help="TCP MCP server port (default: 9801)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--codebase", type=str, help="Path to codebase for indexing (single codebase mode)")
    parser.add_argument("--codebase-name", type=str, default="default", help="Name for the codebase")
    parser.add_argument("--config", type=str, help="Path to config file (JSON/YAML)")
    parser.add_argument("--generate-config", type=str, help="Generate an example config file and exit")
    parser.add_argument("--persist-dir", type=str, help=f"Directory for persistent storage (default: {get_default_persist_dir()})")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--public", action="store_true", help="Allow public access (bind to 0.0.0.0)")
    
    args = parser.parse_args()
    
    # Generate example config if requested
    if args.generate_config:
        from conductor_memory.config.server import generate_example_config
        generate_example_config(args.generate_config)
        print(f"Example configuration saved to: {args.generate_config}")
        return
    
    # Determine host
    host = "0.0.0.0" if args.public else args.host
    
    # Determine persist directory
    persist_dir = args.persist_dir or str(get_default_persist_dir())
    
    # Ensure persist directory exists
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "-m", "conductor_memory.server.unified",
        "--http-port", str(args.http_port),
        "--tcp-port", str(args.tcp_port),
        "--host", host,
        "--persist-dir", persist_dir,
        "--log-level", args.log_level
    ]
    
    if args.config:
        # Config file mode
        cmd.extend(["--config", args.config])
        print(f"Starting MCP Memory Server...")
        print(f"  Config: {args.config}")
    elif args.codebase:
        # Single codebase mode
        cmd.extend(["--codebase-path", args.codebase])
        cmd.extend(["--codebase-name", args.codebase_name])
        print(f"Starting MCP Memory Server (single codebase mode)...")
        print(f"  Codebase: {args.codebase}")
        print(f"  Codebase Name: {args.codebase_name}")
    else:
        # Check for default config
        default_config = get_default_config_path()
        if default_config.exists():
            cmd.extend(["--config", str(default_config)])
            print(f"Starting MCP Memory Server...")
            print(f"  Config: {default_config}")
        else:
            print(f"Starting MCP Memory Server (no codebases configured)...")
            print(f"  Create config at: {default_config}")
    
    print(f"  Host: {host}")
    print(f"  HTTP Port: {args.http_port}")
    print(f"  TCP Port: {args.tcp_port}")
    print(f"  Persist Dir: {persist_dir}")
    print(f"  Log Level: {args.log_level}")
    print()
    print(f"HTTP API: http://{host}:{args.http_port}")
    print(f"API Docs: http://{host}:{args.http_port}/docs")
    print(f"TCP MCP:  {host}:{args.tcp_port}")
    print()
    
    # Set environment variables
    env = os.environ.copy()
    env["MCP_MEMORY_SERVER_URL"] = f"http://{host}:{args.http_port}"
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nShutting down MCP Memory Server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
