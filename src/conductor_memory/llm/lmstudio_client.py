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
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
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
            
            # Make request
            url = f"{self.base_url}/chat/completions"
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
            
            # Check if service is running
            url = f"{self.base_url}/models"
            async with session.get(url) as response:
                if response.status != 200:
                    return False
                
                models_data = await response.json()
                
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
            async with session.get(url) as response:
                if response.status != 200:
                    raise LLMConnectionError(f"Failed to list models: HTTP {response.status}")
                
                models_data = await response.json()
                
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