#!/usr/bin/env bash
#
# conductor-memory service management script for macOS/Linux
#
# Usage:
#   ./conductor-memory.sh [start|stop|restart|status]
#
# Environment variables:
#   CONDUCTOR_MEMORY_PORT     - Port to run on (default: 9820)
#   CONDUCTOR_MEMORY_FORCE_CPU - Set to 1 to force CPU mode (for MPS issues)
#   CONDA_ENV                 - Conda environment name (default: ml-shared)
#

set -e

SERVICE_NAME="conductor-memory"
PORT="${CONDUCTOR_MEMORY_PORT:-9820}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/.$SERVICE_NAME.pid"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/$SERVICE_NAME.log"
CONDA_ENV="${CONDA_ENV:-ml-shared}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       echo "unknown" ;;
    esac
}

# Find Python executable
find_python() {
    # Try conda environment first
    if command -v conda &> /dev/null; then
        local conda_prefix
        conda_prefix=$(conda info --base 2>/dev/null || echo "")
        if [[ -n "$conda_prefix" ]]; then
            local conda_python="$conda_prefix/envs/$CONDA_ENV/bin/python"
            if [[ -x "$conda_python" ]]; then
                echo "$conda_python"
                return
            fi
        fi
    fi
    
    # Try pyenv
    if command -v pyenv &> /dev/null; then
        local pyenv_python
        pyenv_python=$(pyenv which python 2>/dev/null || echo "")
        if [[ -x "$pyenv_python" ]]; then
            echo "$pyenv_python"
            return
        fi
    fi
    
    # Try system python3
    if command -v python3 &> /dev/null; then
        echo "python3"
        return
    fi
    
    # Fallback to python
    if command -v python &> /dev/null; then
        echo "python"
        return
    fi
    
    echo ""
}

# Get PID of running service
get_pid() {
    # Check PID file first
    if [[ -f "$PID_FILE" ]]; then
        local stored_pid
        stored_pid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ -n "$stored_pid" ]] && kill -0 "$stored_pid" 2>/dev/null; then
            echo "$stored_pid"
            return
        fi
    fi
    
    # Check by port
    local pid
    if [[ "$(detect_os)" == "macos" ]]; then
        pid=$(lsof -ti:"$PORT" 2>/dev/null | head -1)
    else
        pid=$(ss -tlnp "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K\d+' | head -1)
        if [[ -z "$pid" ]]; then
            pid=$(netstat -tlnp 2>/dev/null | grep ":$PORT " | grep -oP '\d+(?=/)' | head -1)
        fi
    fi
    
    echo "$pid"
}

# Stop the service
do_stop() {
    local pid
    pid=$(get_pid)
    
    if [[ -n "$pid" ]]; then
        echo -e "${YELLOW}Stopping $SERVICE_NAME (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null || true
        
        # Wait for process to stop
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [[ $waited -lt 10 ]]; do
            sleep 1
            ((waited++))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Force killing...${NC}"
            kill -9 "$pid" 2>/dev/null || true
        fi
        
        rm -f "$PID_FILE"
        echo -e "${GREEN}$SERVICE_NAME stopped.${NC}"
        return 0
    else
        echo -e "${GRAY}$SERVICE_NAME is not running.${NC}"
        return 1
    fi
}

# Start the service
do_start() {
    local existing_pid
    existing_pid=$(get_pid)
    
    if [[ -n "$existing_pid" ]]; then
        echo -e "${YELLOW}$SERVICE_NAME is already running (PID: $existing_pid)${NC}"
        echo -e "${GRAY}Use 'restart' to restart, or 'stop' to stop.${NC}"
        return 1
    fi
    
    local python_exe
    python_exe=$(find_python)
    
    if [[ -z "$python_exe" ]]; then
        echo -e "${RED}Error: Could not find Python executable.${NC}"
        echo -e "${GRAY}Please install Python or activate a conda environment.${NC}"
        return 1
    fi
    
    echo -e "${CYAN}Starting $SERVICE_NAME...${NC}"
    echo -e "${GRAY}Python: $python_exe${NC}"
    
    # Export environment variables
    export CONDUCTOR_MEMORY_PORT="$PORT"
    
    # Start the server in background
    cd "$PROJECT_ROOT"
    nohup "$python_exe" -u -m conductor_memory.server.sse --port "$PORT" \
        >> "$LOG_FILE" 2>&1 &
    
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    
    # Wait for server to be ready
    echo -e "${GRAY}Waiting for server to be ready...${NC}"
    local waited=0
    local max_wait=30
    
    while [[ $waited -lt $max_wait ]]; do
        sleep 1
        ((waited++))
        
        if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            echo ""
            echo -e "${GREEN}$SERVICE_NAME is ready!${NC}"
            echo -e "  ${CYAN}URL: http://127.0.0.1:$PORT/sse${NC}"
            echo -e "  ${CYAN}Dashboard: http://127.0.0.1:$PORT/${NC}"
            echo -e "  ${GRAY}PID: $new_pid${NC}"
            echo -e "  ${GRAY}Logs: $LOG_FILE${NC}"
            return 0
        fi
        
        # Check if process is still running
        if ! kill -0 "$new_pid" 2>/dev/null; then
            echo ""
            echo -e "${RED}$SERVICE_NAME failed to start.${NC}"
            echo -e "${GRAY}Check logs: $LOG_FILE${NC}"
            rm -f "$PID_FILE"
            return 1
        fi
        
        echo -n "."
    done
    
    echo ""
    echo -e "${YELLOW}Server may still be starting (indexing can take a while).${NC}"
    echo -e "${GRAY}Check logs: $LOG_FILE${NC}"
}

# Show service status
do_status() {
    local pid
    pid=$(get_pid)
    
    if [[ -n "$pid" ]]; then
        echo -e "${GREEN}$SERVICE_NAME is running${NC}"
        echo -e "  ${GRAY}PID: $pid${NC}"
        echo -e "  ${CYAN}URL: http://127.0.0.1:$PORT/sse${NC}"
        echo -e "  ${CYAN}Dashboard: http://127.0.0.1:$PORT/${NC}"
        
        # Show memory usage
        if [[ "$(detect_os)" == "macos" ]]; then
            local mem
            mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f", $1/1024}')
            if [[ -n "$mem" ]]; then
                echo -e "  ${GRAY}Memory: ${mem} MB${NC}"
            fi
        else
            local mem
            mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f", $1/1024}')
            if [[ -n "$mem" ]]; then
                echo -e "  ${GRAY}Memory: ${mem} MB${NC}"
            fi
        fi
        
        # Show uptime
        if [[ "$(detect_os)" == "macos" ]]; then
            local start_time
            start_time=$(ps -o lstart= -p "$pid" 2>/dev/null)
            if [[ -n "$start_time" ]]; then
                echo -e "  ${GRAY}Started: $start_time${NC}"
            fi
        fi
        
        return 0
    else
        echo -e "${YELLOW}$SERVICE_NAME is not running${NC}"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 {start|stop|restart|status}"
    echo ""
    echo "Commands:"
    echo "  start   - Start the conductor-memory server"
    echo "  stop    - Stop the conductor-memory server"
    echo "  restart - Restart the conductor-memory server"
    echo "  status  - Show server status"
    echo ""
    echo "Environment variables:"
    echo "  CONDUCTOR_MEMORY_PORT      - Port to run on (default: 9820)"
    echo "  CONDUCTOR_MEMORY_FORCE_CPU - Set to 1 to force CPU mode"
    echo "  CONDA_ENV                  - Conda environment name (default: ml-shared)"
}

# Main
case "${1:-}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop || true
        sleep 1
        do_start
        ;;
    status)
        do_status
        ;;
    -h|--help|help)
        show_usage
        ;;
    *)
        if [[ -n "${1:-}" ]]; then
            echo -e "${RED}Unknown command: $1${NC}"
            echo ""
        fi
        show_usage
        exit 1
        ;;
esac
