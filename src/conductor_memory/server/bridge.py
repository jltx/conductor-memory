#!/usr/bin/env python3
"""
MCP Bridge - Lightweight stdio-to-TCP proxy for MCP connections

This script is spawned by OpenCode and proxies MCP JSON-RPC messages
between stdio (OpenCode) and TCP (memory_server.py).

Features:
- Instant startup (~50ms) - no heavy imports
- Auto-spawns memory_server.py if not running
- Bidirectional proxy: stdin→TCP, TCP→stdout

Usage (in opencode.json.backup):
{
  "mcp": {
    "conductor_memory": {
      "command": ["python", "C:/path/to/conductor/src/mcp_bridge.py"],
      "enabled": true
    }
  }
}
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# Configuration
# conductor-memory project root (3 levels up from this file)
PROJECT_DIR = Path(__file__).parent.parent.parent.parent
SERVER_SCRIPT = PROJECT_DIR / "src" / "conductor_memory" / "server" / "unified.py"
CONFIG_FILE = PROJECT_DIR / "memory_server_config.json"

# Fallback to looking for config in standard locations
def find_config_file() -> Path:
    """Find the config file in standard locations"""
    locations = [
        CONFIG_FILE,
        Path.home() / ".conductor-memory" / "config.json",
        Path.home() / ".conductor-memory" / "memory_server_config.json",
    ]
    for loc in locations:
        if loc.exists():
            return loc
    return CONFIG_FILE  # Return default even if not exists
TCP_HOST = "127.0.0.1"
TCP_PORT = 9801
CONNECT_TIMEOUT = 30  # seconds to wait for server to start
CONNECT_RETRY_INTERVAL = 0.5  # seconds between connection attempts

# Use a specific Python with conductor-memory's dependencies installed
def get_conductor_memory_python() -> str:
    """Get the Python interpreter that has conductor-memory's dependencies"""
    # First check for project's own venv
    venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    
    # Check for Linux/Mac venv
    venv_python_unix = PROJECT_DIR / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        return str(venv_python_unix)
    
    # Use current interpreter (should work if installed via pip)
    return sys.executable

CONDUCTOR_MEMORY_PYTHON = get_conductor_memory_python()


def log(msg: str):
    """Log to stderr (stdout is reserved for MCP protocol)"""
    print(f"[mcp_bridge] {msg}", file=sys.stderr, flush=True)


def is_server_running() -> bool:
    """Check if the memory server is accepting connections"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((TCP_HOST, TCP_PORT))
        sock.close()
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def start_server() -> subprocess.Popen:
    """Start the memory server as a background process"""
    log(f"Starting memory server: {SERVER_SCRIPT}")
    
    # Use conductor-memory's Python to ensure dependencies are available
    log(f"Using Python: {CONDUCTOR_MEMORY_PYTHON}")
    
    # Build command - use module execution
    cmd = [CONDUCTOR_MEMORY_PYTHON, "-m", "conductor_memory.server.unified"]
    
    # Find config file
    config_file = find_config_file()
    if config_file.exists():
        cmd.extend(["--config", str(config_file)])
    
    # Log file for server output (helps debug startup issues)
    log_dir = Path.home() / ".conductor-memory"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "memory_server.log"
    log(f"Server logs will be written to: {log_file}")
    
    # Start detached process with log file
    log_handle = open(log_file, "w")
    kwargs = {
        "stdout": log_handle,
        "stderr": log_handle,
        "stdin": subprocess.DEVNULL,
        "cwd": str(PROJECT_DIR),
    }
    
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    
    proc = subprocess.Popen(cmd, **kwargs)
    log(f"Server started with PID {proc.pid}")
    return proc


def wait_for_server(timeout: float = CONNECT_TIMEOUT) -> bool:
    """Wait for server to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_server_running():
            return True
        time.sleep(CONNECT_RETRY_INTERVAL)
    return False


def connect_to_server() -> socket.socket:
    """Connect to the TCP server, starting it if necessary"""
    if is_server_running():
        log("Server already running")
    else:
        start_server()
        log("Waiting for server to initialize...")
        if not wait_for_server():
            log("ERROR: Server failed to start within timeout")
            sys.exit(1)
        log("Server is ready")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_HOST, TCP_PORT))
    return sock


def stdin_to_socket(sock: socket.socket, shutdown_event: threading.Event):
    """Thread: Read from stdin and write to socket"""
    try:
        while not shutdown_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    # EOF - wait briefly for any pending responses
                    time.sleep(0.5)
                    shutdown_event.set()
                    break
                log(f">>> TO SERVER: {line.strip()[:200]}")
                sock.sendall(line.encode('utf-8'))
            except Exception as e:
                log(f"stdin read error: {e}")
                shutdown_event.set()
                break
    except Exception as e:
        log(f"stdin thread error: {e}")
        shutdown_event.set()


def socket_to_stdout(sock: socket.socket, shutdown_event: threading.Event):
    """Thread: Read from socket and write to stdout"""
    try:
        buffer = b""
        while not shutdown_event.is_set():
            try:
                sock.settimeout(0.5)
                data = sock.recv(4096)
                if not data:
                    log("Socket closed by server")
                    shutdown_event.set()
                    break
                
                buffer += data
                
                # Process complete lines
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    decoded = line.decode('utf-8')
                    log(f"<<< FROM SERVER: {decoded[:200]}")
                    sys.stdout.write(decoded + '\n')
                    sys.stdout.flush()
            except socket.timeout:
                continue
            except OSError:
                # Socket closed - normal during shutdown
                break
            except Exception as e:
                if not shutdown_event.is_set():
                    log(f"socket read error: {e}")
                break
    except Exception as e:
        if not shutdown_event.is_set():
            log(f"socket thread error: {e}")


def run_bridge():
    """Main bridge loop - proxy between stdin/stdout and TCP using threads"""
    log("Bridge starting...")
    sock = connect_to_server()
    log(f"Connected to server at {TCP_HOST}:{TCP_PORT}")
    log("Bridge ready - waiting for MCP messages on stdin")
    
    shutdown_event = threading.Event()
    
    # Start reader/writer threads
    stdin_thread = threading.Thread(
        target=stdin_to_socket, 
        args=(sock, shutdown_event),
        daemon=True
    )
    socket_thread = threading.Thread(
        target=socket_to_stdout,
        args=(sock, shutdown_event),
        daemon=True
    )
    
    stdin_thread.start()
    socket_thread.start()
    
    try:
        # Wait for either thread to signal shutdown
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        log("Interrupted")
        shutdown_event.set()
    finally:
        sock.close()
        log("Bridge closed")


def main():
    """Entry point"""
    # Change to project directory so relative paths work
    os.chdir(PROJECT_DIR)
    
    try:
        run_bridge()
    except Exception as e:
        log(f"Fatal error: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
