"""
LLM integration package for background summarization.

Supports multiple LLM providers:
- LMStudio (default)
- Ollama
"""

from .base import LLMClient, LLMError
from .lmstudio_client import LMStudioClient
from .ollama_client import OllamaClient
from .summarizer import FileSummarizer

__all__ = [
    'LLMClient',
    'LLMError', 
    'LMStudioClient',
    'OllamaClient',
    'FileSummarizer'
]