"""
Base classes for content parsers.

Provides abstract interfaces for parsing different types of content
(code, social media, documents, etc.) into semantic chunks.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..chunking import ChunkMetadata


class ContentParser(ABC):
    """Base class for content parsers (code, social media, docs, etc.)"""
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file.
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            True if this parser can handle the file
        """
        pass
    
    @abstractmethod
    def parse(
        self, 
        content: str, 
        file_path: str, 
        commit_hash: Optional[str] = None
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Parse content into chunks with metadata.
        
        Args:
            content: Raw file content as string
            file_path: Path to the file being parsed
            commit_hash: Optional git commit hash
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        pass


class ParseError(Exception):
    """Raised when parsing fails."""
    pass


class UnsupportedLanguageError(ParseError):
    """Raised when a language is not supported by the parser."""
    pass