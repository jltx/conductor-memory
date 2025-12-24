# Installation Guide

This guide covers multiple ways to install and run conductor-memory on Windows, macOS, and Linux.

## Quick Install (Recommended)

### Prerequisites

- **Python 3.10 or higher** (3.11+ recommended)
- **Git** (for development installation)

### Option 1: Install from PyPI (Easiest)

```bash
pip install conductor-memory
```

### Option 2: Install from GitHub (Latest)

```bash
pip install git+https://github.com/jltx/conductor-memory.git
```

### Option 3: Development Installation

```bash
git clone https://github.com/jltx/conductor-memory.git
cd conductor-memory
pip install -e .
```

## Platform-Specific Instructions

### Windows

#### Using pip (Recommended)
```cmd
# Install Python 3.11+ from python.org if not already installed
pip install conductor-memory

# Or use the convenience script
curl -O https://raw.githubusercontent.com/jltx/conductor-memory/main/scripts/install-windows.ps1
powershell -ExecutionPolicy Bypass -File install-windows.ps1
```

#### Using conda/mamba
```cmd
conda create -n conductor-memory python=3.11
conda activate conductor-memory
pip install conductor-memory
```

### macOS

#### Using pip
```bash
# Install Python 3.11+ via Homebrew if needed
brew install python@3.11

pip3 install conductor-memory
```

#### Using Homebrew (Future)
```bash
# Coming soon - homebrew formula
brew install conductor-memory
```

### Linux (Ubuntu/Debian)

#### Using pip
```bash
# Install Python 3.11+ and pip
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Create virtual environment (recommended)
python3.11 -m venv conductor-memory-env
source conductor-memory-env/bin/activate

pip install conductor-memory
```

#### Using the install script
```bash
curl -sSL https://raw.githubusercontent.com/jltx/conductor-memory/main/scripts/install-linux.sh | bash
```

### Linux (CentOS/RHEL/Fedora)

```bash
# Install Python 3.11+
sudo dnf install python3.11 python3.11-pip

# Create virtual environment
python3.11 -m venv conductor-memory-env
source conductor-memory-env/bin/activate

pip install conductor-memory
```

## Verification

After installation, verify it works:

```bash
# Check installation
conductor-memory --version

# Test basic functionality
conductor-memory --help
```

## Configuration

### 1. Create Configuration Directory

The configuration will be automatically created at:
- **Windows**: `%USERPROFILE%\.conductor-memory\`
- **macOS/Linux**: `~/.conductor-memory/`

### 2. Basic Configuration

Create `~/.conductor-memory/config.json`:

```json
{
  "host": "127.0.0.1",
  "port": 9820,
  "persist_directory": "~/.conductor-memory",
  "codebases": [
    {
      "name": "my-project",
      "path": "/path/to/your/project",
      "extensions": [".py", ".js", ".ts", ".md"],
      "ignore_patterns": ["__pycache__", ".git", "node_modules", "venv"]
    }
  ],
  "embedding_model": "all-MiniLM-L12-v2",
  "enable_file_watcher": true,
  "summarization": {
    "enabled": false,
    "llm_enabled": false
  }
}
```

### 3. Start the Server

```bash
# Start the server
conductor-memory

# Or specify a custom config
conductor-memory --config /path/to/config.json

# Or use the SSE server directly
conductor-memory-sse --port 9820
```

The server will be available at:
- **Dashboard**: http://localhost:9820/
- **MCP endpoint**: http://localhost:9820/sse
- **Health check**: http://localhost:9820/health

## Windows Service Installation

For production use on Windows, you can install conductor-memory as a Windows service that starts automatically on boot.

### Option 1: NSSM (Recommended)

NSSM (Non-Sucking Service Manager) is the most reliable way to run Python applications as Windows services.

```powershell
# Run PowerShell as Administrator
cd path\to\conductor-memory

# Install and start the service (auto-downloads NSSM if needed)
.\scripts\install-service-nssm.ps1

# Check status
.\scripts\install-service-nssm.ps1 -Status

# Stop/Start/Uninstall
.\scripts\install-service-nssm.ps1 -Stop
.\scripts\install-service-nssm.ps1 -Start
.\scripts\install-service-nssm.ps1 -Uninstall
```

The script will automatically:
- Download NSSM if not installed
- Configure automatic restart on failure
- Set up log rotation
- Start the service

### Option 2: pywin32 (Native)

Uses Python's native Windows service support via pywin32:

```cmd
# Install pywin32
pip install conductor-memory[windows-service]

# Run post-install (required once)
python -m pywin32_postinstall -install
```

Then install the service:

```powershell
# Run PowerShell as Administrator
.\scripts\install-service.ps1
```

Or manually:

```cmd
# Run Command Prompt as Administrator
conductor-memory-service install
conductor-memory-service start
```

**Note**: pywin32 can have issues with conda environments. If the service fails to start, use NSSM instead.

### Service Configuration

The Windows service uses the same configuration file as the regular server:
- `%USERPROFILE%\.conductor-memory\config.json`
- Or set via `CONDUCTOR_MEMORY_CONFIG` environment variable

### Service Details

| Property | Value |
|----------|-------|
| Service Name | `ConductorMemory` |
| Display Name | `Conductor Memory` |
| Startup Type | Automatic |
| Dashboard | `http://127.0.0.1:9820/` |
| SSE MCP | `http://127.0.0.1:9820/sse` |
| Log File | `%USERPROFILE%\.conductor-memory\logs\service*.log` |

### Debug Mode

To troubleshoot issues, run the service in foreground debug mode:

```cmd
conductor-memory-service debug
```

This runs the service interactively so you can see all output. Press Ctrl+C to stop.

### Log Files

The service logs to:
- **NSSM**: `%USERPROFILE%\.conductor-memory\logs\service-stdout.log` and `service-stderr.log`
- **pywin32**: `%USERPROFILE%\.conductor-memory\logs\service.log`

View in Event Viewer:
```cmd
eventvwr.msc
# Navigate to: Windows Logs > Application > Filter by Source: ConductorMemory
```

### Troubleshooting Windows Service

#### "Access denied" when installing
- Run PowerShell or Command Prompt as Administrator

#### Service fails to start
1. Check the log file: `%USERPROFILE%\.conductor-memory\logs\service.log`
2. Run in debug mode: `conductor-memory-service debug`
3. Verify config.json is valid JSON

#### "pywin32" not found
```cmd
pip install pywin32
python -m pywin32_postinstall -install
```

#### Service starts but HTTP API not responding
- Wait 30-60 seconds for initial codebase indexing
- Check if ports 9800/9801 are available: `netstat -an | findstr "9800"`

## Optional: LLM Integration

To enable file summarization with local LLMs:

### Install Ollama

#### Windows
```cmd
# Download from https://ollama.ai/download/windows
# Or use winget
winget install Ollama.Ollama
```

#### macOS
```bash
# Download from https://ollama.ai/download/macos
# Or use Homebrew
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Configure LLM

1. Start Ollama:
```bash
ollama serve
```

2. Pull a coding model:
```bash
ollama pull qwen2.5-coder:1.5b
```

3. Update your config.json:
```json
{
  "summarization": {
    "enabled": true,
    "llm_enabled": true,
    "ollama_url": "http://localhost:11434",
    "model": "qwen2.5-coder:1.5b"
  }
}
```

## Troubleshooting

### Common Issues

#### "conductor-memory command not found"
- Ensure Python's Scripts directory is in your PATH
- Try `python -m conductor_memory.server.sse` instead

#### "Permission denied" on Linux/macOS
- Use `pip install --user conductor-memory`
- Or install in a virtual environment

#### "Module not found" errors
- Ensure you're using Python 3.10+
- Try reinstalling: `pip uninstall conductor-memory && pip install conductor-memory`

#### ChromaDB issues
- Delete `~/.conductor-memory/chroma_db/` and restart
- Ensure sufficient disk space

#### Port already in use
- Change the port in config.json
- Or kill the existing process: `lsof -ti:9820 | xargs kill`

#### Segmentation fault on Apple Silicon (M1/M2/M3/M4)
If you see `zsh: segmentation fault` when starting the server, this is caused by known PyTorch MPS (Metal Performance Shaders) issues:
- Race conditions in the MPS backend ([pytorch#167541](https://github.com/pytorch/pytorch/pull/167541))
- OpenMP conflicts on M4 chips ([pytorch#161865](https://github.com/pytorch/pytorch/issues/161865))

**Solutions (in order of preference):**

1. **Update PyTorch to 2.5+** (best fix - includes many MPS stability improvements):
   ```bash
   pip install --upgrade torch
   ```

2. **Force CPU mode** (reliable workaround):
   ```bash
   export CONDUCTOR_MEMORY_FORCE_CPU=1
   conductor-memory-sse --port 9820
   ```
   
   To make this permanent, add to your `~/.zshrc` or `~/.bashrc`:
   ```bash
   export CONDUCTOR_MEMORY_FORCE_CPU=1
   ```

3. **Set device to CPU in config.json**:
   ```json
   {
     "device": "cpu"
   }
   ```

4. **Use smaller batch sizes** (may help with intermittent crashes):
   ```json
   {
     "embedding_batch_size": 32
   }
   ```

**Note:** The CPU fallback is only ~20-30% slower for embedding operations since the MiniLM model is small. For most codebases, indexing will complete in a similar timeframe.

### Getting Help

- **Issues**: https://github.com/jltx/conductor-memory/issues
- **Discussions**: https://github.com/jltx/conductor-memory/discussions
- **Documentation**: https://github.com/jltx/conductor-memory#readme

## Uninstallation

```bash
# Remove the package
pip uninstall conductor-memory

# Remove data directory (optional)
rm -rf ~/.conductor-memory
```

## Development Setup

For contributors and advanced users:

```bash
# Clone the repository
git clone https://github.com/jltx/conductor-memory.git
cd conductor-memory

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with hot reload
python -m conductor_memory.server.sse --reload
```