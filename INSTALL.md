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