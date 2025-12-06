# Conductor Memory - Windows Installation Script
# This script installs conductor-memory and sets up a basic configuration

param(
    [string]$InstallPath = "$env:USERPROFILE\.conductor-memory",
    [switch]$WithOllama = $false,
    [switch]$CreateDesktopShortcut = $false
)

Write-Host "🚀 Installing Conductor Memory for Windows..." -ForegroundColor Green

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "❌ Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
            Write-Host "Please install Python 3.11+ from https://python.org" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "✅ Python version OK: $pythonVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Install conductor-memory
Write-Host "📦 Installing conductor-memory package..." -ForegroundColor Blue
try {
    pip install conductor-memory
    Write-Host "✅ Package installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install package: $_" -ForegroundColor Red
    exit 1
}

# Create configuration directory
Write-Host "📁 Creating configuration directory..." -ForegroundColor Blue
if (!(Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
    Write-Host "✅ Created directory: $InstallPath" -ForegroundColor Green
}

# Create basic configuration
$configPath = Join-Path $InstallPath "config.json"
if (!(Test-Path $configPath)) {
    Write-Host "⚙️ Creating basic configuration..." -ForegroundColor Blue
    
    $config = @{
        host = "127.0.0.1"
        port = 9820
        persist_directory = $InstallPath.Replace('\', '/')
        codebases = @()
        embedding_model = "all-MiniLM-L12-v2"
        enable_file_watcher = $true
        summarization = @{
            enabled = $false
            llm_enabled = $false
            ollama_url = "http://localhost:11434"
            model = "qwen2.5-coder:1.5b"
        }
    } | ConvertTo-Json -Depth 10
    
    $config | Out-File -FilePath $configPath -Encoding UTF8
    Write-Host "✅ Configuration created: $configPath" -ForegroundColor Green
}

# Install Ollama if requested
if ($WithOllama) {
    Write-Host "🤖 Installing Ollama..." -ForegroundColor Blue
    try {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install Ollama.Ollama
            Write-Host "✅ Ollama installed via winget" -ForegroundColor Green
        } else {
            Write-Host "⚠️ winget not available. Please download Ollama from https://ollama.ai/download/windows" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️ Failed to install Ollama automatically. Please download from https://ollama.ai/download/windows" -ForegroundColor Yellow
    }
}

# Create desktop shortcut if requested
if ($CreateDesktopShortcut) {
    Write-Host "🔗 Creating desktop shortcut..." -ForegroundColor Blue
    try {
        $WshShell = New-Object -comObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Conductor Memory.lnk")
        $Shortcut.TargetPath = "cmd.exe"
        $Shortcut.Arguments = "/k conductor-memory"
        $Shortcut.WorkingDirectory = $InstallPath
        $Shortcut.IconLocation = "shell32.dll,21"
        $Shortcut.Description = "Start Conductor Memory Server"
        $Shortcut.Save()
        Write-Host "✅ Desktop shortcut created" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Failed to create desktop shortcut: $_" -ForegroundColor Yellow
    }
}

# Test installation
Write-Host "🧪 Testing installation..." -ForegroundColor Blue
try {
    $version = conductor-memory --version 2>&1
    Write-Host "✅ Installation test passed: $version" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Installation test failed. Try running 'python -m conductor_memory.server.sse --help'" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit your configuration: $configPath" -ForegroundColor White
Write-Host "2. Add your codebase paths to the 'codebases' array" -ForegroundColor White
Write-Host "3. Start the server: conductor-memory" -ForegroundColor White
Write-Host "4. Open dashboard: http://localhost:9820" -ForegroundColor White
Write-Host ""
Write-Host "For help: https://github.com/jltx/conductor-memory#readme" -ForegroundColor Gray