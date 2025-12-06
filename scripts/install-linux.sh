#!/bin/bash
# Conductor Memory - Linux Installation Script
# This script installs conductor-memory and sets up a basic configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Configuration
INSTALL_PATH="$HOME/.conductor-memory"
WITH_OLLAMA=false
CREATE_SYSTEMD_SERVICE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-ollama)
            WITH_OLLAMA=true
            shift
            ;;
        --systemd)
            CREATE_SYSTEMD_SERVICE=true
            shift
            ;;
        --install-path)
            INSTALL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --with-ollama          Install Ollama for LLM integration"
            echo "  --systemd              Create systemd service"
            echo "  --install-path PATH    Custom installation path (default: ~/.conductor-memory)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}🚀 Installing Conductor Memory for Linux...${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    echo -e "${YELLOW}⚠️ Running as root. Consider using a regular user account.${NC}"
fi

# Detect distribution
if command -v apt-get &> /dev/null; then
    DISTRO="debian"
    PKG_MANAGER="apt-get"
elif command -v dnf &> /dev/null; then
    DISTRO="fedora"
    PKG_MANAGER="dnf"
elif command -v yum &> /dev/null; then
    DISTRO="rhel"
    PKG_MANAGER="yum"
elif command -v pacman &> /dev/null; then
    DISTRO="arch"
    PKG_MANAGER="pacman"
else
    echo -e "${YELLOW}⚠️ Unknown distribution. Proceeding with generic installation...${NC}"
    DISTRO="unknown"
fi

# Check Python version
echo -e "${BLUE}🐍 Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
        echo -e "${RED}❌ Python 3.10+ required. Found: Python $PYTHON_VERSION${NC}"
        
        # Try to install newer Python
        case $DISTRO in
            debian)
                echo -e "${BLUE}📦 Installing Python 3.11...${NC}"
                sudo $PKG_MANAGER update
                sudo $PKG_MANAGER install -y python3.11 python3.11-pip python3.11-venv
                PYTHON_CMD="python3.11"
                ;;
            fedora|rhel)
                echo -e "${BLUE}📦 Installing Python 3.11...${NC}"
                sudo $PKG_MANAGER install -y python3.11 python3.11-pip
                PYTHON_CMD="python3.11"
                ;;
            arch)
                echo -e "${BLUE}📦 Installing Python...${NC}"
                sudo pacman -S --noconfirm python python-pip
                PYTHON_CMD="python3"
                ;;
            *)
                echo -e "${RED}❌ Please install Python 3.10+ manually${NC}"
                exit 1
                ;;
        esac
    else
        echo -e "${GREEN}✅ Python version OK: Python $PYTHON_VERSION${NC}"
        PYTHON_CMD="python3"
    fi
else
    echo -e "${RED}❌ Python not found. Installing...${NC}"
    
    case $DISTRO in
        debian)
            sudo $PKG_MANAGER update
            sudo $PKG_MANAGER install -y python3 python3-pip python3-venv
            ;;
        fedora|rhel)
            sudo $PKG_MANAGER install -y python3 python3-pip
            ;;
        arch)
            sudo pacman -S --noconfirm python python-pip
            ;;
        *)
            echo -e "${RED}❌ Please install Python 3.10+ manually${NC}"
            exit 1
            ;;
    esac
    PYTHON_CMD="python3"
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${BLUE}📦 Installing pip...${NC}"
    case $DISTRO in
        debian)
            sudo $PKG_MANAGER install -y python3-pip
            ;;
        fedora|rhel)
            sudo $PKG_MANAGER install -y python3-pip
            ;;
        arch)
            sudo pacman -S --noconfirm python-pip
            ;;
    esac
fi

# Create virtual environment (recommended)
echo -e "${BLUE}🏠 Setting up virtual environment...${NC}"
if [[ ! -d "$INSTALL_PATH" ]]; then
    mkdir -p "$INSTALL_PATH"
fi

if [[ ! -d "$INSTALL_PATH/venv" ]]; then
    $PYTHON_CMD -m venv "$INSTALL_PATH/venv"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
source "$INSTALL_PATH/venv/bin/activate"

# Install conductor-memory
echo -e "${BLUE}📦 Installing conductor-memory package...${NC}"
pip install --upgrade pip
pip install conductor-memory

echo -e "${GREEN}✅ Package installed successfully${NC}"

# Create basic configuration
CONFIG_PATH="$INSTALL_PATH/config.json"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${BLUE}⚙️ Creating basic configuration...${NC}"
    
    cat > "$CONFIG_PATH" << EOF
{
  "host": "127.0.0.1",
  "port": 9820,
  "persist_directory": "$INSTALL_PATH",
  "codebases": [],
  "embedding_model": "all-MiniLM-L12-v2",
  "enable_file_watcher": true,
  "summarization": {
    "enabled": false,
    "llm_enabled": false,
    "ollama_url": "http://localhost:11434",
    "model": "qwen2.5-coder:1.5b"
  }
}
EOF
    
    echo -e "${GREEN}✅ Configuration created: $CONFIG_PATH${NC}"
fi

# Install Ollama if requested
if [[ "$WITH_OLLAMA" == true ]]; then
    echo -e "${BLUE}🤖 Installing Ollama...${NC}"
    if ! command -v ollama &> /dev/null; then
        curl -fsSL https://ollama.ai/install.sh | sh
        echo -e "${GREEN}✅ Ollama installed${NC}"
        
        # Start Ollama service
        if command -v systemctl &> /dev/null; then
            sudo systemctl enable ollama
            sudo systemctl start ollama
            echo -e "${GREEN}✅ Ollama service started${NC}"
        fi
        
        # Pull coding model
        echo -e "${BLUE}📥 Downloading coding model...${NC}"
        ollama pull qwen2.5-coder:1.5b
        echo -e "${GREEN}✅ Model downloaded${NC}"
    else
        echo -e "${GREEN}✅ Ollama already installed${NC}"
    fi
fi

# Create systemd service if requested
if [[ "$CREATE_SYSTEMD_SERVICE" == true ]]; then
    echo -e "${BLUE}🔧 Creating systemd service...${NC}"
    
    SERVICE_FILE="/etc/systemd/system/conductor-memory.service"
    sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Conductor Memory Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_PATH
Environment=PATH=$INSTALL_PATH/venv/bin
ExecStart=$INSTALL_PATH/venv/bin/conductor-memory --config $CONFIG_PATH
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable conductor-memory
    echo -e "${GREEN}✅ Systemd service created and enabled${NC}"
    echo -e "${CYAN}To start: sudo systemctl start conductor-memory${NC}"
    echo -e "${CYAN}To check status: sudo systemctl status conductor-memory${NC}"
fi

# Create shell script wrapper
WRAPPER_SCRIPT="$INSTALL_PATH/start.sh"
cat > "$WRAPPER_SCRIPT" << EOF
#!/bin/bash
# Conductor Memory startup script
cd "$INSTALL_PATH"
source venv/bin/activate
conductor-memory --config "$CONFIG_PATH" "\$@"
EOF
chmod +x "$WRAPPER_SCRIPT"

# Add to PATH if not already there
SHELL_RC=""
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [[ -n "$SHELL_RC" ]] && ! grep -q "conductor-memory" "$SHELL_RC"; then
    echo "# Conductor Memory" >> "$SHELL_RC"
    echo "alias conductor-memory='$WRAPPER_SCRIPT'" >> "$SHELL_RC"
    echo -e "${GREEN}✅ Added conductor-memory alias to $SHELL_RC${NC}"
fi

# Test installation
echo -e "${BLUE}🧪 Testing installation...${NC}"
if "$INSTALL_PATH/venv/bin/conductor-memory" --version &> /dev/null; then
    VERSION=$("$INSTALL_PATH/venv/bin/conductor-memory" --version)
    echo -e "${GREEN}✅ Installation test passed: $VERSION${NC}"
else
    echo -e "${YELLOW}⚠️ Installation test failed. Try running the wrapper script directly.${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Installation complete!${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo -e "${NC}1. Edit your configuration: $CONFIG_PATH${NC}"
echo -e "${NC}2. Add your codebase paths to the 'codebases' array${NC}"
echo -e "${NC}3. Start the server: $WRAPPER_SCRIPT${NC}"
echo -e "${NC}4. Open dashboard: http://localhost:9820${NC}"
echo ""
echo -e "${GRAY}For help: https://github.com/jltx/conductor-memory#readme${NC}"

# Deactivate virtual environment
deactivate