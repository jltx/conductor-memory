"""
ChromaDB Server Manager

Manages ChromaDB as a subprocess when running in HTTP mode.
Handles startup, health checks, monitoring, and graceful shutdown.
"""

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ChromaServerManager:
    """Manages ChromaDB server as a subprocess using uvicorn."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        data_path: Optional[str] = None,
        startup_timeout: float = 30.0,
        health_check_interval: float = 5.0
    ):
        self.host = host
        self.port = port
        self.data_path = data_path or os.path.expanduser("~/.conductor-memory/chroma-server")
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        
        self._process: Optional[subprocess.Popen] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._log_thread: Optional[threading.Thread] = None
        self._shutdown_event = asyncio.Event()
        self._stop_log_thread = False
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def is_running(self) -> bool:
        """Check if the ChromaDB server process is running."""
        return self._process is not None and self._process.poll() is None
    
    async def is_healthy(self) -> bool:
        """Check if ChromaDB server is responding to health checks."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                # Try v1 heartbeat first, then v2
                for endpoint in ["/api/v1/heartbeat", "/api/v2/heartbeat"]:
                    try:
                        response = await client.get(f"{self.base_url}{endpoint}")
                        if response.status_code == 200:
                            return True
                    except Exception:
                        continue
                return False
        except Exception:
            return False
    
    def _is_healthy_sync(self) -> bool:
        """Synchronous health check for use during startup."""
        import urllib.request
        import urllib.error
        
        for endpoint in ["/api/v1/heartbeat", "/api/v2/heartbeat"]:
            try:
                req = urllib.request.Request(f"{self.base_url}{endpoint}")
                with urllib.request.urlopen(req, timeout=2.0) as response:
                    if response.status == 200:
                        return True
            except Exception:
                continue
        return False
    
    def start_sync(self) -> bool:
        """Start the ChromaDB server subprocess synchronously.
        
        This method is safe to call from asyncio.run() as it doesn't create
        any async tasks that would persist after the call completes.
        """
        # Check if already running
        if self._is_healthy_sync():
            logger.info(f"ChromaDB server already running at {self.base_url}")
            return True
        
        # Ensure data directory exists
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for ChromaDB
        env = os.environ.copy()
        env["IS_PERSISTENT"] = "TRUE"
        env["PERSIST_DIRECTORY"] = self.data_path
        env["ANONYMIZED_TELEMETRY"] = "FALSE"
        env["CHROMA_SERVER_HOST"] = self.host
        env["CHROMA_SERVER_HTTP_PORT"] = str(self.port)
        
        # Build uvicorn command to run chromadb.app
        cmd = [
            sys.executable, "-m", "uvicorn",
            "chromadb.app:app",
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "warning"
        ]
        
        logger.info(f"Starting ChromaDB server: {' '.join(cmd)}")
        logger.info(f"ChromaDB data path: {self.data_path}")
        
        try:
            # Start subprocess
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                env=env
            )
            
            # Start log reader in a separate thread (not async task)
            self._stop_log_thread = False
            self._log_thread = threading.Thread(target=self._read_output_sync, daemon=True)
            self._log_thread.start()
            
            # Wait for server to be ready (synchronously)
            if self._wait_for_ready_sync():
                logger.info(f"ChromaDB server ready at {self.base_url}")
                return True
            else:
                logger.error("ChromaDB server failed to start within timeout")
                self.stop_sync()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start ChromaDB server: {e}")
            return False
    
    def _wait_for_ready_sync(self) -> bool:
        """Wait for ChromaDB server to be ready (synchronous)."""
        start_time = time.time()
        
        while time.time() - start_time < self.startup_timeout:
            if not self.is_running():
                if self._process:
                    returncode = self._process.poll()
                    logger.error(f"ChromaDB process exited unexpectedly with code {returncode}")
                return False
            
            if self._is_healthy_sync():
                return True
            
            time.sleep(0.5)
        
        return False
    
    def _read_output_sync(self) -> None:
        """Read and log subprocess output (runs in separate thread)."""
        if not self._process or not self._process.stdout:
            return
        
        try:
            while not self._stop_log_thread and self.is_running():
                line = self._process.stdout.readline()
                if line:
                    line = line.rstrip()
                    if "error" in line.lower() or "exception" in line.lower():
                        logger.error(f"[ChromaDB] {line}")
                    else:
                        logger.debug(f"[ChromaDB] {line}")
                elif not self.is_running():
                    break
        except Exception as e:
            logger.debug(f"ChromaDB output reader stopped: {e}")
    
    def stop_sync(self) -> None:
        """Stop the ChromaDB server subprocess synchronously."""
        self._stop_log_thread = True
        
        if self._process:
            logger.info("Stopping ChromaDB server...")
            self._process.terminate()
            
            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("ChromaDB server did not stop gracefully, killing...")
                self._process.kill()
            
            self._process = None
            logger.info("ChromaDB server stopped")
    
    # Async versions for use in the main event loop
    
    async def start(self) -> bool:
        """Start the ChromaDB server subprocess using uvicorn."""
        if await self.is_healthy():
            logger.info(f"ChromaDB server already running at {self.base_url}")
            return True
        
        # Use sync version to avoid async task issues
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.start_sync)
    
    def start_monitoring(self) -> None:
        """Start the background monitoring task. Call this from the main async context."""
        if self._monitor_task is None or self._monitor_task.done():
            self._shutdown_event.clear()
            self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def _monitor_loop(self) -> None:
        """Monitor ChromaDB server and restart if needed."""
        restart_count = 0
        max_restarts = 5
        
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.health_check_interval)
            
            if self._shutdown_event.is_set():
                break
            
            if not self.is_running() or not await self.is_healthy():
                if restart_count >= max_restarts:
                    logger.error(f"ChromaDB server failed {max_restarts} times, giving up")
                    break
                
                restart_count += 1
                logger.warning(f"ChromaDB server not healthy, restarting (attempt {restart_count}/{max_restarts})")
                
                # Kill existing process if any
                self.stop_sync()
                
                # Restart
                loop = asyncio.get_event_loop()
                if await loop.run_in_executor(None, self.start_sync):
                    restart_count = 0  # Reset on successful restart
            else:
                restart_count = 0  # Reset on healthy check
    
    async def stop(self) -> None:
        """Stop the ChromaDB server subprocess."""
        self._shutdown_event.set()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.stop_sync)
