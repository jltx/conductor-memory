"""
Summarization configuration management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SummarizationConfig:
    """Configuration for background summarization."""
    
    # Enable/disable summarization
    enabled: bool = True
    llm_enabled: bool = True
    
    # Ollama configuration
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:1.5b"  # Fast, code-focused model for summarization
    
    # Rate limiting and performance
    rate_limit_seconds: float = 0.1   # Local Ollama doesn't need rate limiting
    timeout_seconds: float = 15.0     # Fail faster if something hangs
    
    # Startup timing (fallback timeout if callback system fails)
    startup_delay_seconds: float = 60.0  # Fallback timeout for summarizer startup
    
    # File size limits
    max_file_lines: int = 600
    max_file_tokens: int = 4000
    
    # LLM parameters
    temperature: float = 0.1
    max_response_tokens: int = 384    # JSON response only needs ~100-150 tokens
    
    # File filtering
    skip_patterns: List[str] = field(default_factory=lambda: [
        "**/test/**", 
        "**/*_test.*", 
        "**/vendor/**", 
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/build/**",
        "**/dist/**"
    ])
    
    priority_patterns: List[str] = field(default_factory=lambda: [
        "**/src/**", 
        "**/lib/**", 
        "**/core/**"
    ])
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SummarizationConfig':
        """Create config from dictionary."""
        # Extract summarization section
        summarization_config = config_dict.get('summarization', {})
        
        return cls(
            enabled=summarization_config.get('enabled', True),
            llm_enabled=summarization_config.get('llm_enabled', True),
            ollama_url=summarization_config.get('ollama_url', "http://localhost:11434"),
            model=summarization_config.get('model', "qwen2.5-coder:1.5b"),
            rate_limit_seconds=summarization_config.get('rate_limit_seconds', 0.1),
            timeout_seconds=summarization_config.get('timeout_seconds', 15.0),
            startup_delay_seconds=summarization_config.get('startup_delay_seconds', 30.0),
            max_file_lines=summarization_config.get('max_file_lines', 600),
            max_file_tokens=summarization_config.get('max_file_tokens', 4000),
            temperature=summarization_config.get('temperature', 0.1),
            max_response_tokens=summarization_config.get('max_response_tokens', 384),
            skip_patterns=summarization_config.get('skip_patterns', [
                "**/test/**", "**/*_test.*", "**/vendor/**", "**/node_modules/**",
                "**/__pycache__/**", "**/build/**", "**/dist/**"
            ]),
            priority_patterns=summarization_config.get('priority_patterns', [
                "**/src/**", "**/lib/**", "**/core/**"
            ])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enabled': self.enabled,
            'llm_enabled': self.llm_enabled,
            'ollama_url': self.ollama_url,
            'model': self.model,
            'rate_limit_seconds': self.rate_limit_seconds,
            'timeout_seconds': self.timeout_seconds,

            'startup_delay_seconds': self.startup_delay_seconds,
            'max_file_lines': self.max_file_lines,
            'max_file_tokens': self.max_file_tokens,
            'temperature': self.temperature,
            'max_response_tokens': self.max_response_tokens,
            'skip_patterns': self.skip_patterns,
            'priority_patterns': self.priority_patterns
        }