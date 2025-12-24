#!/usr/bin/env python3
"""
Windows Service wrapper for Conductor Memory

This module allows conductor-memory to run as a proper Windows service
that starts automatically on boot and runs in the background.

NOTE: The recommended approach is to use NSSM (install-service-nssm.ps1)
which is more reliable. This pywin32-based service is provided as an
alternative for environments where NSSM cannot be used.

Installation:
    # Install the service (run as Administrator)
    conductor-memory-service install
    
    # Start the service
    conductor-memory-service start
    
    # Stop the service
    conductor-memory-service stop
    
    # Remove the service
    conductor-memory-service remove

Or via sc.exe:
    sc start ConductorMemory
    sc stop ConductorMemory
    
The service reads configuration from:
    1. CONDUCTOR_MEMORY_CONFIG environment variable
    2. ~/.conductor-memory/config.json
    3. ./memory_server_config.json (legacy)
"""

import os
import sys

# Ensure the conductor_memory package is importable when running as a service
# This handles both installed packages and development installs
_this_file = os.path.abspath(__file__)
_server_dir = os.path.dirname(_this_file)
_conductor_memory_dir = os.path.dirname(_server_dir)
_src_dir = os.path.dirname(_conductor_memory_dir)

# Add src directory to path if needed (for development installs)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check if pywin32 is available
try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False

# Configure logging to Windows Event Log and file
LOG_DIR = Path.home() / ".conductor-memory" / "logs"
LOG_FILE = LOG_DIR / "service.log"


def setup_logging():
    """Setup logging for the service"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
    except Exception as e:
        # Fallback to stderr if file logging fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stderr
        )
        logging.error(f"Failed to setup file logging: {e}")
    
    return logging.getLogger(__name__)


logger = setup_logging()

# Log startup info for debugging
logger.info(f"Windows service module loaded")
logger.info(f"Python: {sys.executable}")
logger.info(f"Python path: {sys.path[:3]}...")  # First 3 entries


def find_config_file() -> Optional[Path]:
    """Find the configuration file in standard locations"""
    # Check environment variable first
    env_config = os.environ.get("CONDUCTOR_MEMORY_CONFIG")
    if env_config:
        path = Path(env_config)
        if path.exists():
            return path
    
    # Check default home config
    home_config = Path.home() / ".conductor-memory" / "config.json"
    if home_config.exists():
        return home_config
    
    # Check legacy location
    legacy_config = Path("memory_server_config.json")
    if legacy_config.exists():
        return legacy_config
    
    return None


class ConductorMemoryService:
    """
    Windows Service for Conductor Memory
    
    This class can be used standalone for testing or wrapped by 
    win32serviceutil.ServiceFramework for actual Windows service operation.
    
    Runs the SSE server which provides:
    - HTTP API on port 9820
    - SSE MCP endpoint at /sse
    - Web dashboard at /
    """
    
    def __init__(self):
        self.running = False
        self.stop_event = threading.Event()
        self.server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
    def run_server(self):
        """Run the SSE server in an asyncio event loop"""
        # Import here to avoid loading heavy dependencies at module level
        from .sse import main as sse_main
        
        logger.info("Starting SSE server...")
        
        # The SSE server's main() handles everything including config loading
        # We need to simulate command line args
        original_argv = sys.argv
        
        config_file = find_config_file()
        if config_file:
            sys.argv = ['conductor-memory', '--config', str(config_file)]
            logger.info(f"Using config: {config_file}")
        else:
            sys.argv = ['conductor-memory']
            logger.warning("No config file found, using defaults")
        
        try:
            sse_main()
        except Exception as e:
            logger.exception(f"SSE server error: {e}")
        finally:
            sys.argv = original_argv
    
    def start(self):
        """Start the service"""
        logger.info("Starting Conductor Memory service...")
        self.running = True
        self.stop_event.clear()
        
        try:
            # Start server in a separate thread
            self.server_thread = threading.Thread(
                target=self.run_server,
                daemon=True
            )
            self.server_thread.start()
            
            logger.info("Conductor Memory service started successfully")
            logger.info("Dashboard: http://127.0.0.1:9820/")
            logger.info("SSE MCP:   http://127.0.0.1:9820/sse")
            
            # Wait for stop signal
            while self.running and not self.stop_event.is_set():
                self.stop_event.wait(timeout=1.0)
                
        except Exception as e:
            logger.exception(f"Failed to start service: {e}")
            raise
    
    def stop(self):
        """Stop the service"""
        logger.info("Stopping Conductor Memory service...")
        self.running = False
        self.stop_event.set()
        
        # Wait for server thread to finish
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=10)
        
        logger.info("Conductor Memory service stopped")


if HAS_PYWIN32:
    class WindowsService(win32serviceutil.ServiceFramework):
        """
        Windows Service Framework wrapper for Conductor Memory
        
        This integrates with the Windows Service Control Manager (SCM).
        """
        
        _svc_name_ = "ConductorMemory"
        _svc_display_name_ = "Conductor Memory"
        _svc_description_ = "Semantic memory service with codebase indexing for AI agents"
        
        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.service = ConductorMemoryService()
        
        def SvcStop(self):
            """Called when the service is asked to stop"""
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            logger.info("Service stop requested")
            self.service.stop()
            win32event.SetEvent(self.stop_event)
        
        def SvcDoRun(self):
            """Called when the service is asked to start"""
            try:
                servicemanager.LogMsg(
                    servicemanager.EVENTLOG_INFORMATION_TYPE,
                    servicemanager.PYS_SERVICE_STARTED,
                    (self._svc_name_, '')
                )
                logger.info("Windows service starting...")
                self.service.start()
            except Exception as e:
                logger.exception(f"Service failed: {e}")
                servicemanager.LogErrorMsg(f"Service failed: {e}")


def check_admin():
    """Check if running with administrator privileges"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False


def main():
    """Main entry point for the Windows service"""
    if not HAS_PYWIN32:
        print("ERROR: pywin32 is required for Windows service functionality")
        print("Install it with: pip install pywin32")
        print("\nAfter installing, run: python -m pywin32_postinstall -install")
        print("\nAlternatively, use NSSM (recommended):")
        print("  .\\scripts\\install-service-nssm.ps1")
        sys.exit(1)
    
    if len(sys.argv) == 1:
        # Started by Windows SCM
        try:
            servicemanager.Initialize()
            servicemanager.PrepareToHostSingle(WindowsService)
            servicemanager.StartServiceCtrlDispatcher()
        except Exception as e:
            # Not started by SCM, show help
            print("Conductor Memory Windows Service")
            print("=" * 40)
            print("\nUsage:")
            print("  conductor-memory-service install   - Install the service")
            print("  conductor-memory-service start     - Start the service")
            print("  conductor-memory-service stop      - Stop the service")
            print("  conductor-memory-service remove    - Remove the service")
            print("  conductor-memory-service restart   - Restart the service")
            print("  conductor-memory-service debug     - Run in debug mode (foreground)")
            print("\nNote: install/remove/start/stop require Administrator privileges")
            print(f"\nConfiguration: {find_config_file() or 'Not found'}")
            print(f"Log file: {LOG_FILE}")
            print("\nRecommended: Use NSSM instead for more reliable service management:")
            print("  .\\scripts\\install-service-nssm.ps1")
    else:
        # Handle command line arguments
        cmd = sys.argv[1].lower()
        
        if cmd in ('install', 'remove', 'update') and not check_admin():
            print("ERROR: Administrator privileges required for this operation")
            print("Please run from an elevated Command Prompt or PowerShell")
            sys.exit(1)
        
        if cmd == 'debug':
            # Run in foreground for debugging
            print("Running in debug mode (Ctrl+C to stop)...")
            print("Dashboard: http://127.0.0.1:9820/")
            print("SSE MCP:   http://127.0.0.1:9820/sse")
            service = ConductorMemoryService()
            try:
                service.start()
            except KeyboardInterrupt:
                service.stop()
        else:
            # Use win32serviceutil to handle the command
            win32serviceutil.HandleCommandLine(WindowsService)


if __name__ == "__main__":
    main()
