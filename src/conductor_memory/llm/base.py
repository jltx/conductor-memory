"""
Base LLM client interface for background summarization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to LLM service."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM returns invalid or error response."""
    pass


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    response_time_ms: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        """
        Initialize LLM client.
        
        Args:
            base_url: Base URL for the LLM service
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMConnectionError: If unable to connect to service
            LLMResponseError: If response is invalid or contains error
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        List available models.
        
        Returns:
            List of available model names
        """
        pass
    
    def get_provider_name(self) -> str:
        """Get the name of this LLM provider."""
        return self.__class__.__name__.replace('Client', '').lower()