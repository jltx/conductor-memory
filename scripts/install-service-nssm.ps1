# Conductor Memory - Windows Service Installation using NSSM
# NSSM (Non-Sucking Service Manager) provides more reliable service management
#
# Usage:
#   .\install-service-nssm.ps1              # Install and start service
#   .\install-service-nssm.ps1 -Uninstall   # Remove the service
#   .\install-service-nssm.ps1 -Reinstall   # Reinstall (remove + install)
#   .\install-service-nssm.ps1 -Status      # Show service status

param(
    [switch]$Uninstall,
    [switch]$Reinstall,
    [switch]$Status,
    [switch]$Start,
    [switch]$Stop,
    [string]$PythonPath = "",
    [switch]$DownloadNssm
)

$ServiceName = "ConductorMemory"
$DisplayName = "Conductor Memory"
$Description = "Semantic memory service with codebase indexing for AI agents"
$NssmPath = "$env:ProgramFiles\nssm\nssm.exe"
$NssmAltPath = "$PSScriptRoot\nssm.exe"

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

# Find NSSM
function Find-Nssm {
    if (Test-Path $NssmPath) { return $NssmPath }
    if (Test-Path $NssmAltPath) { return $NssmAltPath }
    
    # Check if installed via winget/scoop/choco
    $nssmCmd = Get-Command nssm -ErrorAction SilentlyContinue
    if ($nssmCmd) { return $nssmCmd.Source }
    
    return $null
}

function Install-Nssm {
    Write-Host "NSSM not found. Attempting to install..." -ForegroundColor Yellow
    
    # Try winget first
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing NSSM via winget..." -ForegroundColor Cyan
        winget install NSSM.NSSM --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            return Find-Nssm
        }
    }
    
    # Try chocolatey
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "Installing NSSM via Chocolatey..." -ForegroundColor Cyan
        choco install nssm -y
        if ($LASTEXITCODE -eq 0) {
            return Find-Nssm
        }
    }
    
    # Download directly
    Write-Host "Downloading NSSM directly..." -ForegroundColor Cyan
    $nssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
    $tempZip = "$env:TEMP\nssm.zip"
    $tempDir = "$env:TEMP\nssm"
    
    try {
        Invoke-WebRequest -Uri $nssmUrl -OutFile $tempZip
        Expand-Archive -Path $tempZip -DestinationPath $tempDir -Force
        
        # Find the 64-bit executable
        $nssmExe = Get-ChildItem -Path $tempDir -Recurse -Filter "nssm.exe" | 
                   Where-Object { $_.DirectoryName -like "*win64*" } | 
                   Select-Object -First 1
        
        if ($nssmExe) {
            # Copy to scripts directory
            Copy-Item $nssmExe.FullName -Destination $NssmAltPath -Force
            Write-Host "NSSM installed to: $NssmAltPath" -ForegroundColor Green
            return $NssmAltPath
        }
    } catch {
        Write-Host "Failed to download NSSM: $_" -ForegroundColor Red
    } finally {
        Remove-Item $tempZip -ErrorAction SilentlyContinue
        Remove-Item $tempDir -Recurse -ErrorAction SilentlyContinue
    }
    
    return $null
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
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:9820/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "  Health: OK" -ForegroundColor Green
                Write-Host "  Dashboard: http://127.0.0.1:9820/" -ForegroundColor Cyan
                Write-Host "  SSE MCP:   http://127.0.0.1:9820/sse" -ForegroundColor Cyan
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
    Write-Host "Installing Conductor Memory as Windows Service (using NSSM)..." -ForegroundColor Cyan
    
    # Find or install NSSM
    $nssm = Find-Nssm
    if (-not $nssm) {
        $nssm = Install-Nssm
    }
    if (-not $nssm) {
        Write-Host "ERROR: NSSM not found and could not be installed" -ForegroundColor Red
        Write-Host "Please install NSSM manually: https://nssm.cc/download" -ForegroundColor Yellow
        Write-Host "Or run: winget install NSSM.NSSM" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Using NSSM: $nssm" -ForegroundColor Gray
    
    # Find Python
    $python = Find-Python -PreferredPath $PythonPath
    if (-not $python) {
        Write-Host "ERROR: Python not found" -ForegroundColor Red
        Write-Host "Please specify the Python path with -PythonPath" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Using Python: $python" -ForegroundColor Gray
    
    # Verify conductor-memory is installed
    $testResult = & $python -c "import conductor_memory; print('OK')" 2>&1
    if ($testResult -ne "OK") {
        Write-Host "ERROR: conductor-memory not installed for this Python" -ForegroundColor Red
        Write-Host "Install with: $python -m pip install conductor-memory" -ForegroundColor Yellow
        exit 1
    }
    
    # Remove existing service if present
    $existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($existingService) {
        Write-Host "Removing existing service..." -ForegroundColor Yellow
        if ($existingService.Status -eq 'Running') {
            & $nssm stop $ServiceName
            Start-Sleep -Seconds 2
        }
        & $nssm remove $ServiceName confirm
        Start-Sleep -Seconds 1
    }
    
    # Install service
    Write-Host "Installing service..." -ForegroundColor Gray
    & $nssm install $ServiceName $python
    
    # Config and log paths - use explicit paths since service runs as SYSTEM user
    $configDir = "$env:USERPROFILE\.conductor-memory"
    $configFile = "$configDir\config.json"
    $logDir = "$configDir\logs"
    
    # Build app parameters - include config path explicitly
    $appParams = "-m conductor_memory.server.sse"
    if (Test-Path $configFile) {
        $appParams = "$appParams --config `"$configFile`""
        Write-Host "Using config: $configFile" -ForegroundColor Gray
    } else {
        Write-Host "WARNING: No config file found at $configFile" -ForegroundColor Yellow
    }
    
    & $nssm set $ServiceName AppParameters $appParams
    & $nssm set $ServiceName DisplayName $DisplayName
    & $nssm set $ServiceName Description $Description
    & $nssm set $ServiceName Start SERVICE_AUTO_START
    
    # Set working directory
    $projectRoot = Resolve-Path "$PSScriptRoot\.."
    & $nssm set $ServiceName AppDirectory $projectRoot
    
    # Configure logging
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    & $nssm set $ServiceName AppStdout "$logDir\service-stdout.log"
    & $nssm set $ServiceName AppStderr "$logDir\service-stderr.log"
    & $nssm set $ServiceName AppRotateFiles 1
    & $nssm set $ServiceName AppRotateBytes 10485760  # 10MB
    
    # Set environment variables - include USERPROFILE for the service
    & $nssm set $ServiceName AppEnvironmentExtra "TF_CPP_MIN_LOG_LEVEL=3" "TF_ENABLE_ONEDNN_OPTS=0" "USERPROFILE=$env:USERPROFILE"
    
    # Configure restart on failure
    & $nssm set $ServiceName AppExit Default Restart
    & $nssm set $ServiceName AppRestartDelay 5000  # 5 seconds
    
    Write-Host "Service installed successfully!" -ForegroundColor Green
    
    # Start the service
    Write-Host "Starting service..." -ForegroundColor Gray
    & $nssm start $ServiceName
    
    # Wait for startup
    Start-Sleep -Seconds 5
    Get-ServiceStatus
}

function Uninstall-MemoryService {
    Write-Host "Uninstalling Conductor Memory service..." -ForegroundColor Yellow
    
    $nssm = Find-Nssm
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    
    if (-not $service) {
        Write-Host "Service is not installed" -ForegroundColor Gray
        return
    }
    
    # Stop the service first
    if ($service.Status -eq 'Running') {
        Write-Host "Stopping service..." -ForegroundColor Gray
        if ($nssm) {
            & $nssm stop $ServiceName
        } else {
            Stop-Service -Name $ServiceName -Force
        }
        Start-Sleep -Seconds 2
    }
    
    # Remove the service
    if ($nssm) {
        & $nssm remove $ServiceName confirm
    } else {
        sc.exe delete $ServiceName
    }
    
    Write-Host "Service uninstalled successfully" -ForegroundColor Green
}

function Start-MemoryService {
    $nssm = Find-Nssm
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    
    if (-not $service) {
        Write-Host "ERROR: Service is not installed" -ForegroundColor Red
        exit 1
    }
    
    if ($service.Status -eq 'Running') {
        Write-Host "Service is already running" -ForegroundColor Yellow
    } else {
        Write-Host "Starting service..." -ForegroundColor Cyan
        if ($nssm) {
            & $nssm start $ServiceName
        } else {
            Start-Service -Name $ServiceName
        }
        Start-Sleep -Seconds 3
        Get-ServiceStatus
    }
}

function Stop-MemoryService {
    $nssm = Find-Nssm
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    
    if (-not $service) {
        Write-Host "ERROR: Service is not installed" -ForegroundColor Red
        exit 1
    }
    
    if ($service.Status -eq 'Stopped') {
        Write-Host "Service is already stopped" -ForegroundColor Yellow
    } else {
        Write-Host "Stopping service..." -ForegroundColor Yellow
        if ($nssm) {
            & $nssm stop $ServiceName
        } else {
            Stop-Service -Name $ServiceName -Force
        }
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
} elseif ($DownloadNssm) {
    $result = Install-Nssm
    if ($result) {
        Write-Host "NSSM available at: $result" -ForegroundColor Green
    }
} else {
    Install-MemoryService
}
