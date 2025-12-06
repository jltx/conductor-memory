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
            
            # Make request
            url = f"{self.base_url}/api/generate"
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
    
    async def health_check(self) -> bool:
        """
        Check if Ollama is available and the model is accessible.
        
        Returns:
            True if service is healthy and model is available
        """
        try:
            session = await self._get_session()
            
            # Check if service is running
            url = f"{self.base_url}/api/tags"
            async with session.get(url) as response:
                if response.status != 200:
                    return False
                
                tags_data = await response.json()
                
                # Check if our model is available
                if "models" not in tags_data:
                    return False
                
                available_models = [model["name"] for model in tags_data["models"]]
                
                # Check if our specific model is available
                if self.model not in available_models:
                    logger.warning(f"Ollama model '{self.model}' not found. Available: {available_models}")
                    return False
                
                return True
                
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False
    
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
            # Create a session with longer timeout for warm-up
            warm_up_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=warm_up_timeout) as session:
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
            async with session.get(url) as response:
                if response.status != 200:
                    raise LLMConnectionError(f"Failed to list models: HTTP {response.status}")
                
                tags_data = await response.json()
                
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