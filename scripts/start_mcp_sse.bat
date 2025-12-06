@echo off
REM Start the MCP Memory Server (SSE) for OpenCode remote connections
REM Run this once before starting OpenCode, or add to Windows startup

cd /d "%~dp0\.."
echo Starting MCP Memory Server (SSE) on http://127.0.0.1:9820/sse
echo.
echo Configure OpenCode with:
echo   "type": "remote", "url": "http://127.0.0.1:9820/sse"
echo.
python -m conductor_memory.server.sse %*
