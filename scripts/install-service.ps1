# Conductor Memory - Windows Service Installation Script
# Run this script as Administrator to install conductor-memory as a Windows service
#
# Usage:
#   .\install-service.ps1              # Install and start service
#   .\install-service.ps1 -Uninstall   # Remove the service
#   .\install-service.ps1 -Reinstall   # Reinstall (remove + install)
#   .\install-service.ps1 -Status      # Show service status

param(
    [switch]$Uninstall,
    [switch]$Reinstall,
    [switch]$Status,
    [switch]$Start,
    [switch]$Stop,
    [string]$PythonPath = ""
)

$ServiceName = "ConductorMemory"
$DisplayName = "Conductor Memory"
$Description = "Semantic memory service with codebase indexing for AI agents"

# Check for administrator privileges
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Host "ERROR: This script requires Administrator privileges" -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again" -ForegroundColor Yellow
    exit 1
}

# Find Python executable
function Find-Python {
    param([string]$PreferredPath)
    
    if ($PreferredPath -and (Test-Path $PreferredPath)) {
        return $PreferredPath
    }
    
    # Check conda environments
    $condaPaths = @(
        "$env:USERPROFILE\miniconda3\envs\ml-shared\python.exe",
        "$env:USERPROFILE\anaconda3\envs\ml-shared\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe"
    )
    
    foreach ($path in $condaPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    # Check system Python
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython) {
        return $systemPython.Source
    }
    
    return $null
}

function Get-ServiceStatus {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        Write-Host "Service: $DisplayName" -ForegroundColor Cyan
        Write-Host "  Name:   $ServiceName" -ForegroundColor Gray
        Write-Host "  Status: $($service.Status)" -ForegroundColor $(if ($service.Status -eq 'Running') { 'Green' } else { 'Yellow' })
        
        if ($service.Status -eq 'Running') {
            # Get process info
            $wmiService = Get-WmiObject Win32_Service -Filter "Name='$ServiceName'" -ErrorAction SilentlyContinue
            if ($wmiService -and $wmiService.ProcessId) {
                $process = Get-Process -Id $wmiService.ProcessId -ErrorAction SilentlyContinue
                if ($process) {
                    Write-Host "  PID:    $($process.Id)" -ForegroundColor Gray
                    Write-Host "  Memory: $([math]::Round($process.WorkingSet64 / 1MB, 1)) MB" -ForegroundColor Gray
                    Write-Host "  Started: $($process.StartTime)" -ForegroundColor Gray
                }
            }
        }
        
        # Check if service is healthy
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:9800/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "  Health: OK" -ForegroundColor Green
                Write-Host "  HTTP API: http://127.0.0.1:9800" -ForegroundColor Cyan
                Write-Host "  TCP MCP:  127.0.0.1:9801" -ForegroundColor Cyan
            }
        } catch {
            if ($service.Status -eq 'Running') {
                Write-Host "  Health: Starting..." -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "Service '$DisplayName' is not installed" -ForegroundColor Yellow
    }
}

function Install-MemoryService {
    Write-Host "Installing Conductor Memory as Windows Service..." -ForegroundColor Cyan
    
    # Find Python
    $python = Find-Python -PreferredPath $PythonPath
    if (-not $python) {
        Write-Host "ERROR: Python not found" -ForegroundColor Red
        Write-Host "Please specify the Python path with -PythonPath" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "Using Python: $python" -ForegroundColor Gray
    
    # Check if pywin32 is installed
    Write-Host "Checking pywin32 installation..." -ForegroundColor Gray
    $pywin32Check = & $python -c "import win32serviceutil" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing pywin32..." -ForegroundColor Yellow
        & $python -m pip install pywin32
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install pywin32" -ForegroundColor Red
            exit 1
        }
        # Run post-install script
        Write-Host "Running pywin32 post-install..." -ForegroundColor Gray
        & $python -m pywin32_postinstall -install 2>&1 | Out-Null
    }
    
    # Check if conductor-memory is installed
    Write-Host "Checking conductor-memory installation..." -ForegroundColor Gray
    $cmCheck = & $python -c "from conductor_memory.server.windows_service import main" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: conductor-memory not installed or windows_service module not found" -ForegroundColor Red
        Write-Host "Install with: pip install conductor-memory[windows-service]" -ForegroundColor Yellow
        exit 1
    }
    
    # Install the service
    Write-Host "Installing service..." -ForegroundColor Gray
    & $python -m conductor_memory.server.windows_service install
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Service installed successfully!" -ForegroundColor Green
        
        # Configure service to auto-start
        Write-Host "Configuring automatic startup..." -ForegroundColor Gray
        Set-Service -Name $ServiceName -StartupType Automatic
        
        # Start the service
        Write-Host "Starting service..." -ForegroundColor Gray
        Start-Service -Name $ServiceName
        
        # Wait for startup
        Start-Sleep -Seconds 3
        Get-ServiceStatus
    } else {
        Write-Host "ERROR: Failed to install service" -ForegroundColor Red
        exit 1
    }
}

function Uninstall-MemoryService {
    Write-Host "Uninstalling Conductor Memory service..." -ForegroundColor Yellow
    
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Host "Service is not installed" -ForegroundColor Gray
        return
    }
    
    # Stop the service first
    if ($service.Status -eq 'Running') {
        Write-Host "Stopping service..." -ForegroundColor Gray
        Stop-Service -Name $ServiceName -Force
        Start-Sleep -Seconds 2
    }
    
    # Find Python and use it to remove the service
    $python = Find-Python -PreferredPath $PythonPath
    if ($python) {
        & $python -m conductor_memory.server.windows_service remove
    } else {
        # Fallback to sc.exe
        sc.exe delete $ServiceName
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Service uninstalled successfully" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to uninstall service" -ForegroundColor Red
        exit 1
    }
}

function Start-MemoryService {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Host "ERROR: Service is not installed" -ForegroundColor Red
        exit 1
    }
    
    if ($service.Status -eq 'Running') {
        Write-Host "Service is already running" -ForegroundColor Yellow
    } else {
        Write-Host "Starting service..." -ForegroundColor Cyan
        Start-Service -Name $ServiceName
        Start-Sleep -Seconds 3
        Get-ServiceStatus
    }
}

function Stop-MemoryService {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Host "ERROR: Service is not installed" -ForegroundColor Red
        exit 1
    }
    
    if ($service.Status -eq 'Stopped') {
        Write-Host "Service is already stopped" -ForegroundColor Yellow
    } else {
        Write-Host "Stopping service..." -ForegroundColor Yellow
        Stop-Service -Name $ServiceName -Force
        Write-Host "Service stopped" -ForegroundColor Green
    }
}

# Main logic
if ($Status) {
    Get-ServiceStatus
} elseif ($Start) {
    Start-MemoryService
} elseif ($Stop) {
    Stop-MemoryService
} elseif ($Uninstall) {
    Uninstall-MemoryService
} elseif ($Reinstall) {
    Uninstall-MemoryService
    Start-Sleep -Seconds 2
    Install-MemoryService
} else {
    Install-MemoryService
}
