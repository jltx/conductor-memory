# Start conductor-memory service using shared ml-shared conda environment
param(
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status
)

$ServiceName = "conductor-memory"
$Port = 9820
$PidFile = "$PSScriptRoot\..\.$ServiceName.pid"
$LogFile = "$PSScriptRoot\..\logs\$ServiceName.log"
$CondaExe = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
$CondaEnv = "ml-shared"

# Ensure logs directory exists
$LogDir = Split-Path $LogFile -Parent
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Get-ServicePid {
    if (Test-Path $PidFile) {
        $storedPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($storedPid -and (Get-Process -Id $storedPid -ErrorAction SilentlyContinue)) {
            return [int]$storedPid
        }
    }
    # Also check by port
    $netstat = netstat -ano | Select-String ":$Port.*LISTENING"
    if ($netstat) {
        $parts = $netstat -split '\s+'
        return [int]$parts[-1]
    }
    return $null
}

function Stop-Service {
    $servicePid = Get-ServicePid
    if ($servicePid) {
        Write-Host "Stopping $ServiceName (PID: $servicePid)..." -ForegroundColor Yellow
        Stop-Process -Id $servicePid -Force -ErrorAction SilentlyContinue
        if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
        Start-Sleep -Seconds 1
        Write-Host "$ServiceName stopped." -ForegroundColor Green
        return $true
    } else {
        Write-Host "$ServiceName is not running." -ForegroundColor Gray
        return $false
    }
}

function Start-Service {
    $existingPid = Get-ServicePid
    if ($existingPid) {
        Write-Host "$ServiceName is already running (PID: $existingPid)" -ForegroundColor Yellow
        Write-Host "Use -Restart to restart, or -Stop to stop." -ForegroundColor Gray
        return
    }
    
    Write-Host "Starting $ServiceName..." -ForegroundColor Cyan
    
    $projectRoot = Resolve-Path "$PSScriptRoot\.."
    $pythonExe = "$env:USERPROFILE\miniconda3\envs\$CondaEnv\python.exe"
    $stderrLog = "$LogFile.stderr"
    
    # Start Python directly to capture the correct PID
    # stdout goes to main log, stderr to separate file (check both when debugging)
    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList "-u", "-m", "conductor_memory.server.sse" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError $stderrLog `
        -PassThru
    
    $process.Id | Out-File $PidFile -Force
    
    # Wait for server to be ready
    Write-Host "Waiting for server to be ready..." -ForegroundColor Gray
    $maxWait = 30
    $waited = 0
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 1
        $waited++
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host ""
                Write-Host "$ServiceName is ready!" -ForegroundColor Green
                Write-Host "  URL: http://127.0.0.1:$Port/sse" -ForegroundColor Cyan
                Write-Host "  Dashboard: http://127.0.0.1:$Port/" -ForegroundColor Cyan
                Write-Host "  PID: $($process.Id)" -ForegroundColor Gray
                Write-Host "  Logs: $LogFile (stdout), $LogFile.stderr (errors)" -ForegroundColor Gray
                return
            }
        } catch {
            Write-Host "." -NoNewline -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "Server may still be starting (indexing can take a while)." -ForegroundColor Yellow
    Write-Host "Check logs: $LogFile (stdout), $LogFile.stderr (errors)" -ForegroundColor Gray
}

function Show-Status {
    $servicePid = Get-ServicePid
    if ($servicePid) {
        $process = Get-Process -Id $servicePid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "$ServiceName is running" -ForegroundColor Green
            Write-Host "  PID: $servicePid" -ForegroundColor Gray
            Write-Host "  URL: http://127.0.0.1:$Port/sse" -ForegroundColor Cyan
            Write-Host "  Memory: $([math]::Round($process.WorkingSet64 / 1MB, 1)) MB" -ForegroundColor Gray
            Write-Host "  Started: $($process.StartTime)" -ForegroundColor Gray
        }
    } else {
        Write-Host "$ServiceName is not running" -ForegroundColor Yellow
    }
}

# Main logic
if ($Stop) {
    Stop-Service
} elseif ($Restart) {
    Stop-Service
    Start-Sleep -Seconds 1
    Start-Service
} elseif ($Status) {
    Show-Status
} else {
    Start-Service
}
