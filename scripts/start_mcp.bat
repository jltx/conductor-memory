@echo off
REM Start the MCP Memory Server (stdio mode)
REM This is spawned by OpenCode for MCP communication

cd /d "%~dp0\.."
python -m conductor_memory.server.stdio %*
