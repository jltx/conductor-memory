"""
LMStudio client for background summarization.

LMStudio provides an OpenAI-compatible API at http://localhost:1234/v1
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional

import aiohttp

from .base import LLMClient, LLMResponse, LLMConnectionError, LLMResponseError

logger = logging.getLogger(__name__)


class LMStudioClient(LLMClient):
    """LMStudio client using OpenAI-compatible API."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:1234/v1", 
        model: str = "auto",
        timeout: float = 30.0
    ):
        """
        Initialize LMStudio client.
        
        Args:
            base_url: LMStudio API base URL (default: http://localhost:1234/v1)
            model: Model name or "auto" to use currently loaded model
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
        Generate text using LMStudio.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (default: 2048)
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            else:
                payload["max_tokens"] = 2048  # Default for summarization
            
            # Make request with explicit timeout using asyncio.wait_for
            # This works correctly in threads without a Task wrapper
            url = f"{self.base_url}/chat/completions"
            
            async def do_request():
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMResponseError(f"LMStudio API error {response.status}: {error_text}")
                    
                    response_data = await response.json()
                    
                    # Extract content
                    if "choices" not in response_data or not response_data["choices"]:
                        raise LLMResponseError("No choices in LMStudio response")
                    
                    choice = response_data["choices"][0]
                    if "message" not in choice or "content" not in choice["message"]:
                        raise LLMResponseError("Invalid message format in LMStudio response")
                    
                    return response_data
            
            try:
                response_data = await asyncio.wait_for(do_request(), timeout=self.timeout)
            except asyncio.TimeoutError:
                raise LLMConnectionError(f"LMStudio request timed out after {self.timeout}s")
            
            choice = response_data["choices"][0]
            content = choice["message"]["content"]
            
            # Extract usage info
            tokens_used = None
            if "usage" in response_data and "total_tokens" in response_data["usage"]:
                tokens_used = response_data["usage"]["total_tokens"]
            
            # Get actual model name from response
            actual_model = response_data.get("model", self.model)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=content,
                model=actual_model,
                tokens_used=tokens_used,
                response_time_ms=response_time_ms,
                raw_response=response_data
            )
                
        except aiohttp.ClientError as e:
            raise LLMConnectionError(f"Failed to connect to LMStudio: {e}")
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Invalid JSON response from LMStudio: {e}")
        except Exception as e:
            if isinstance(e, (LLMConnectionError, LLMResponseError)):
                raise
            raise LLMResponseError(f"Unexpected error calling LMStudio: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if LMStudio is available and has a model loaded.
        
        Returns:
            True if service is healthy and model is loaded
        """
        try:
            session = await self._get_session()
            
            # Check if service is running with explicit timeout
            url = f"{self.base_url}/models"
            
            async def do_request():
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    return await response.json()
            
            try:
                models_data = await asyncio.wait_for(do_request(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.debug("LMStudio health check timed out")
                return False
            
            if models_data is None:
                return False
            
            # Check if any models are available
            if "data" not in models_data or not models_data["data"]:
                logger.warning("LMStudio is running but no models are loaded")
                return False
            
            return True
                
        except Exception as e:
            logger.debug(f"LMStudio health check failed: {e}")
            return False
    
    async def list_models(self) -> list[str]:
        """
        List available models in LMStudio.
        
        Returns:
            List of available model names
        """
        try:
            session = await self._get_session()
            
            url = f"{self.base_url}/models"
            
            async def do_request():
                async with session.get(url) as response:
                    if response.status != 200:
                        raise LLMConnectionError(f"Failed to list models: HTTP {response.status}")
                    return await response.json()
            
            try:
                models_data = await asyncio.wait_for(do_request(), timeout=10.0)
            except asyncio.TimeoutError:
                raise LLMConnectionError("Timed out listing models")
            
            if "data" not in models_data:
                return []
            
            return [model["id"] for model in models_data["data"]]
                
        except aiohttp.ClientError as e:
            raise LLMConnectionError(f"Failed to connect to LMStudio: {e}")
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