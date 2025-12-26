"""
Ollama client for background summarization.

Ollama provides a REST API at http://localhost:11434
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional

import aiohttp

from .base import LLMClient, LLMResponse, LLMConnectionError, LLMResponseError

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Ollama client using native Ollama API."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434", 
        model: str = "qwen2.5-coder:1.5b",
        timeout: float = 30.0
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            model: Model name (default: qwen2.5-coder:1.5b - fast, code-focused)
            timeout: Request timeout in seconds
        """
        super().__init__(base_url, model, timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session.
        
        Note: We don't set a timeout here because aiohttp's ClientTimeout
        uses asyncio.timeout() which requires being inside a Task. When running
        in a thread with loop.run_until_complete(), there's no Task wrapper.
        Instead, we handle timeouts at the request level with asyncio.wait_for().
        
        Also tracks which event loop created the session to avoid cross-loop issues
        when running in a background thread.
        """
        current_loop = asyncio.get_event_loop()
        
        # Check if session exists and is for the current loop
        if self._session is not None and not self._session.closed:
            if self._session_loop is current_loop:
                return self._session
            else:
                # Session was created in a different loop - close it and create new one
                try:
                    await self._session.close()
                except Exception:
                    pass  # Ignore errors closing session from different loop
        
        # Create new session for this loop
        self._session = aiohttp.ClientSession()
        self._session_loop = current_loop
        return self._session
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (Ollama uses num_predict)
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # Prepare request payload for Ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            # Add max tokens (Ollama calls it num_predict)
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            else:
                payload["options"]["num_predict"] = 2048  # Default for summarization
            
            # Make request with explicit timeout using asyncio.wait_for
            # This works correctly in threads without a Task wrapper
            url = f"{self.base_url}/api/generate"
            
            async def do_request():
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMResponseError(f"Ollama API error {response.status}: {error_text}")
                    
                    response_data = await response.json()
                    
                    # Check for errors in response
                    if "error" in response_data:
                        raise LLMResponseError(f"Ollama error: {response_data['error']}")
                    
                    # Extract content
                    if "response" not in response_data:
                        raise LLMResponseError("No response field in Ollama response")
                    
                    return response_data
            
            try:
                response_data = await asyncio.wait_for(do_request(), timeout=self.timeout)
            except asyncio.TimeoutError:
                raise LLMConnectionError(f"Ollama request timed out after {self.timeout}s")
            
            content = response_data["response"]
            
            # Extract usage info (Ollama provides different metrics)
            tokens_used = None
            if "eval_count" in response_data:
                # eval_count is output tokens, prompt_eval_count is input tokens
                eval_count = response_data.get("eval_count", 0)
                prompt_eval_count = response_data.get("prompt_eval_count", 0)
                tokens_used = eval_count + prompt_eval_count
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                response_time_ms=response_time_ms,
                raw_response=response_data
            )
                
        except aiohttp.ClientError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}")
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Invalid JSON response from Ollama: {e}")
        except Exception as e:
            if isinstance(e, (LLMConnectionError, LLMResponseError)):
                raise
            raise LLMResponseError(f"Unexpected error calling Ollama: {e}")
    
    async def health_check(self, auto_pull: bool = False) -> bool:
        """
        Check if Ollama is available and the model is accessible.
        
        Args:
            auto_pull: If True, automatically pull the model if not found
        
        Returns:
            True if service is healthy and model is available
        """
        try:
            session = await self._get_session()
            
            # Check if service is running with explicit timeout
            url = f"{self.base_url}/api/tags"
            
            async def do_request():
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    return await response.json()
            
            try:
                tags_data = await asyncio.wait_for(do_request(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.debug("Ollama health check timed out")
                return False
            
            if tags_data is None:
                return False
            
            # Check if our model is available
            if "models" not in tags_data:
                return False
            
            available_models = [model["name"] for model in tags_data["models"]]
            
            # Check if our specific model is available
            if self.model not in available_models:
                logger.warning(f"Ollama model '{self.model}' not found. Available: {available_models}")
                
                if auto_pull:
                    logger.info(f"Attempting to pull model '{self.model}'...")
                    if await self.pull_model():
                        return True
                
                return False
            
            return True
                
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False
    
    async def pull_model(self, timeout: float = 600.0) -> bool:
        """
        Pull/download the model from Ollama registry.
        
        Args:
            timeout: Maximum time to wait for model download (default 10 minutes)
            
        Returns:
            True if model was pulled successfully
        """
        logger.info(f"Pulling Ollama model '{self.model}'... (this may take several minutes)")
        
        try:
            # Don't use aiohttp.ClientTimeout - it requires Task context
            # Instead use asyncio.wait_for for the entire operation
            async def do_pull():
                async with aiohttp.ClientSession() as session:
                    url = f"{self.base_url}/api/pull"
                    payload = {"name": self.model, "stream": True}
                    
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Failed to pull model: {error_text}")
                            return False
                        
                        # Stream the response to show progress
                        last_status = ""
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    status = data.get("status", "")
                                    
                                    # Log progress updates (but not too frequently)
                                    if status != last_status:
                                        if "pulling" in status.lower():
                                            # Extract progress percentage if available
                                            completed = data.get("completed", 0)
                                            total = data.get("total", 0)
                                            if total > 0:
                                                pct = (completed / total) * 100
                                                logger.info(f"  Downloading: {pct:.1f}%")
                                        elif status:
                                            logger.info(f"  {status}")
                                        last_status = status
                                    
                                    # Check for completion or error
                                    if data.get("error"):
                                        logger.error(f"Pull error: {data['error']}")
                                        return False
                                        
                                except json.JSONDecodeError:
                                    continue
                        
                        logger.info(f"Successfully pulled model '{self.model}'")
                        return True
            
            return await asyncio.wait_for(do_pull(), timeout=timeout)
                    
        except asyncio.TimeoutError:
            logger.error(f"Model pull timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    @staticmethod
    def check_ollama_installed() -> tuple[bool, str]:
        """
        Check if Ollama is installed on the system.
        
        Returns:
            Tuple of (is_installed: bool, message: str)
        """
        import shutil
        import subprocess
        
        # Check if ollama command exists
        ollama_path = shutil.which("ollama")
        
        if ollama_path:
            # Try to get version
            try:
                result = subprocess.run(
                    ["ollama", "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                version = result.stdout.strip() or result.stderr.strip()
                return (True, f"Ollama installed: {version}")
            except Exception:
                return (True, f"Ollama found at: {ollama_path}")
        
        # Not found - provide installation instructions
        import platform
        system = platform.system().lower()
        
        if system == "windows":
            install_msg = "Install from: https://ollama.com/download/windows"
        elif system == "darwin":
            install_msg = "Install with: brew install ollama  OR  https://ollama.com/download/mac"
        else:
            install_msg = "Install with: curl -fsSL https://ollama.com/install.sh | sh"
        
        return (False, f"Ollama not found. {install_msg}")
    
    @staticmethod
    def check_ollama_running(base_url: str = "http://localhost:11434") -> tuple[bool, str]:
        """
        Check if Ollama server is running (synchronous check).
        
        Returns:
            Tuple of (is_running: bool, message: str)
        """
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    return (True, "Ollama is running")
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        
        return (False, f"Ollama is not running. Start it with: ollama serve")
    
    async def warm_up(self, timeout: float = 120.0) -> bool:
        """
        Pre-load the model by sending a simple prompt.
        
        This ensures the model is loaded into memory before processing files,
        avoiding timeouts on the first real request.
        
        Args:
            timeout: Maximum time to wait for model to load (default 120s)
            
        Returns:
            True if model loaded successfully
        """
        logger.info(f"Warming up Ollama model '{self.model}'... (this may take a minute)")
        
        try:
            # Don't use aiohttp.ClientTimeout - it requires Task context
            # Instead use asyncio.wait_for for the entire operation
            async def do_warmup():
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model,
                        "prompt": "Say 'ready' in one word.",
                        "stream": False,
                        "options": {
                            "num_predict": 10,  # Very short response
                            "temperature": 0.0
                        }
                    }
                    
                    url = f"{self.base_url}/api/generate"
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            logger.info(f"Model '{self.model}' is warmed up and ready")
                            return True
                        else:
                            error_text = await response.text()
                            logger.warning(f"Model warm-up failed: {error_text}")
                            return False
            
            return await asyncio.wait_for(do_warmup(), timeout=timeout)
                        
        except asyncio.TimeoutError:
            logger.warning(f"Model warm-up timed out after {timeout}s")
            return False
        except Exception as e:
            logger.warning(f"Model warm-up error: {e}")
            return False
    
    async def list_models(self) -> list[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of available model names
        """
        try:
            session = await self._get_session()
            
            url = f"{self.base_url}/api/tags"
            
            async def do_request():
                async with session.get(url) as response:
                    if response.status != 200:
                        raise LLMConnectionError(f"Failed to list models: HTTP {response.status}")
                    return await response.json()
            
            try:
                tags_data = await asyncio.wait_for(do_request(), timeout=10.0)
            except asyncio.TimeoutError:
                raise LLMConnectionError("Timed out listing models")
            
            if "models" not in tags_data:
                return []
            
            return [model["name"] for model in tags_data["models"]]
                
        except aiohttp.ClientError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            if isinstance(e, LLMConnectionError):
                raise
            raise LLMResponseError(f"Error listing models: {e}")
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()