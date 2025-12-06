"""
Memory Database Interface for the Hybrid Local/Cloud LLM Orchestrator
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid

class RoleEnum(str, Enum):
    """Enumeration of message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryType(str, Enum):
    """
    Type of memory for filtering and categorization.
    
    - CODE: Indexed source code chunks from codebase
    - CONVERSATION: Chat/session context
    - DECISION: Architectural decisions (pinned by default)
    - LESSON: Debugging insights and lessons learned
    """
    CODE = "code"
    CONVERSATION = "conversation"
    DECISION = "decision"
    LESSON = "lesson"

@dataclass
class MemoryChunk:
    """Represents a chunk of memory with associated metadata"""
    id: str
    project_id: str
    role: RoleEnum
    prompt: str
    response: str
    doc_text: str
    embedding_id: str
    tags: List[str]
    pin: bool
    relevance_score: float
    created_at: datetime
    updated_at: datetime
    source: str  # git_commit, manual, cloud, local, codebase_indexing
    memory_type: MemoryType = MemoryType.CODE  # Default for backward compatibility
    expires_at: Optional[datetime] = None

class MemoryDB(ABC):
    """Abstract interface for memory database operations"""
    
    @abstractmethod
    def get(self, memory_id: str) -> MemoryChunk:
        """Retrieve a memory chunk by ID"""
        pass
    
    @abstractmethod
    def query(self, filters: Dict[str, Any]) -> List[MemoryChunk]:
        """Query memory chunks with filters"""
        pass
    
    @abstractmethod
    def insert(self, chunk: MemoryChunk) -> None:
        """Insert a new memory chunk"""
        pass
    
    @abstractmethod
    def update(self, chunk: MemoryChunk) -> None:
        """Update an existing memory chunk"""
        pass
    
    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Delete a memory chunk by ID"""
        pass