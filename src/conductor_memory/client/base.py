"""
Base classes for MCP Memory Client Tools

Duplicates the Tool and ToolResponse from conductor's tools.py to allow
conductor-memory to be used as a standalone package.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ToolResponse:
    """Standardized response format for all tool executions"""
    success: bool
    output: str
    structured_data: Dict[str, Any]
    errors: Optional[str] = None


class Tool(ABC):
    """Abstract base class for all MCP tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the tool with given parameters"""
        pass

    def validate_params(self, params: Dict[str, Any], required_keys: List[str]) -> Optional[str]:
        """Validate that required parameters are present"""
        missing = [key for key in required_keys if key not in params]
        if missing:
            return f"Missing required parameters: {', '.join(missing)}"
        return None

    @property
    def schema(self) -> Dict[str, Any]:
        """Return JSON schema for this tool's parameters"""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def get_openai_schema(self) -> Dict[str, Any]:
        """Convert MCP tool schema to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema
            }
        }
