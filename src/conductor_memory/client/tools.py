"""
MCP Tool Definitions for Memory Server

These tools can be used by agents to interact with the MCP Memory Server.
They provide a standardized interface for memory operations following the MCP protocol.
"""

import json
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .base import Tool, ToolResponse


@dataclass
class MemoryServerConfig:
    """Configuration for connecting to the MCP Memory Server"""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'MemoryServerConfig':
        """Create config from environment variables"""
        import os
        return cls(
            base_url=os.getenv("MCP_MEMORY_SERVER_URL", "http://localhost:8000"),
            timeout=int(os.getenv("MCP_MEMORY_SERVER_TIMEOUT", "30"))
        )


class MemorySearchTool(Tool):
    """Tool for searching memories using semantic similarity"""
    
    def __init__(self, config: Optional[MemoryServerConfig] = None):
        super().__init__(
            name="memory_search",
            description="Search for relevant context and memories using semantic similarity"
        )
        self.config = config or MemoryServerConfig.from_env()
    
    def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Search for relevant memories
        
        Parameters:
        - query (str): Search query for semantic similarity
        - project_id (str, optional): Filter by project ID
        - max_results (int, optional): Maximum number of results (default: 10)
        - min_relevance (float, optional): Minimum relevance score 0-1 (default: 0.1)
        - include_conversation (bool, optional): Include conversation history (default: true)
        """
        # Validate required parameters
        error = self.validate_params(params, ["query"])
        if error:
            return ToolResponse(success=False, output="", structured_data={}, errors=error)
        
        try:
            # Prepare request
            search_request = {
                "query": params["query"],
                "project_id": params.get("project_id"),
                "max_results": params.get("max_results", 10),
                "min_relevance": params.get("min_relevance", 0.1),
                "include_conversation": params.get("include_conversation", True)
            }
            
            # Remove None values
            search_request = {k: v for k, v in search_request.items() if v is not None}
            
            # Make API request
            response = requests.post(
                f"{self.config.base_url}/search",
                json=search_request,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Format output for agent consumption
            output_lines = [
                f"Found {result['total_found']} relevant memories (showing {len(result['results'])})",
                f"Query time: {result['query_time_ms']:.1f}ms",
                f"Context: {result['context_summary']}",
                ""
            ]
            
            # Add relevant memories
            for i, memory in enumerate(result['results'], 1):
                output_lines.append(f"=== Memory {i} ===")
                output_lines.append(f"ID: {memory['id']}")
                output_lines.append(f"Project: {memory['project_id']}")
                output_lines.append(f"Role: {memory['role']}")
                output_lines.append(f"Source: {memory['source']}")
                output_lines.append(f"Tags: {', '.join(memory['tags'])}")
                
                if memory['prompt']:
                    output_lines.append(f"Prompt: {memory['prompt'][:200]}...")
                if memory['response']:
                    output_lines.append(f"Response: {memory['response'][:200]}...")
                if memory['doc_text']:
                    output_lines.append(f"Content: {memory['doc_text'][:300]}...")
                
                output_lines.append("")
            
            return ToolResponse(
                success=True,
                output="\n".join(output_lines),
                structured_data=result
            )
            
        except requests.exceptions.RequestException as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Failed to connect to memory server: {str(e)}"
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Memory search failed: {str(e)}"
            )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for semantic similarity"
                },
                "project_id": {
                    "type": "string",
                    "description": "Filter by project ID (optional)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                },
                "min_relevance": {
                    "type": "number",
                    "description": "Minimum relevance score (0-1)",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "include_conversation": {
                    "type": "boolean",
                    "description": "Include conversation history in search",
                    "default": True
                }
            },
            "required": ["query"]
        }


class MemoryStoreTool(Tool):
    """Tool for storing new memories"""
    
    def __init__(self, config: Optional[MemoryServerConfig] = None):
        super().__init__(
            name="memory_store",
            description="Store a new memory chunk (conversation, code, or documentation)"
        )
        self.config = config or MemoryServerConfig.from_env()
    
    def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Store a new memory chunk
        
        Parameters:
        - project_id (str): Project identifier
        - role (str): Role of the memory (user, assistant, system, tool)
        - prompt (str, optional): User prompt or input
        - response (str, optional): Assistant response or output
        - doc_text (str, optional): Document text or code content
        - tags (list, optional): Tags for categorization
        - pin (bool, optional): Pin this memory to prevent pruning
        - source (str, optional): Source of the memory (default: "api")
        """
        # Validate required parameters
        error = self.validate_params(params, ["project_id", "role"])
        if error:
            return ToolResponse(success=False, output="", structured_data={}, errors=error)
        
        try:
            # Validate role
            valid_roles = ["user", "assistant", "system", "tool"]
            if params["role"] not in valid_roles:
                return ToolResponse(
                    success=False,
                    output="",
                    structured_data={},
                    errors=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
                )
            
            # Prepare request
            store_request = {
                "project_id": params["project_id"],
                "role": params["role"],
                "prompt": params.get("prompt", ""),
                "response": params.get("response", ""),
                "doc_text": params.get("doc_text", ""),
                "tags": params.get("tags", []),
                "pin": params.get("pin", False),
                "source": params.get("source", "api")
            }
            
            # Make API request
            response = requests.post(
                f"{self.config.base_url}/store",
                json=store_request,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Format output
            output = f"Memory stored successfully!\n"
            output += f"ID: {result['id']}\n"
            output += f"Project: {result['project_id']}\n"
            output += f"Role: {result['role']}\n"
            output += f"Tags: {', '.join(result['tags'])}\n"
            output += f"Created: {result['created_at']}"
            
            return ToolResponse(
                success=True,
                output=output,
                structured_data=result
            )
            
        except requests.exceptions.RequestException as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Failed to connect to memory server: {str(e)}"
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Memory storage failed: {str(e)}"
            )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier"
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "system", "tool"],
                    "description": "Role of the memory"
                },
                "prompt": {
                    "type": "string",
                    "description": "User prompt or input (optional)"
                },
                "response": {
                    "type": "string",
                    "description": "Assistant response or output (optional)"
                },
                "doc_text": {
                    "type": "string",
                    "description": "Document text or code content (optional)"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization (optional)"
                },
                "pin": {
                    "type": "boolean",
                    "description": "Pin this memory to prevent pruning",
                    "default": False
                },
                "source": {
                    "type": "string",
                    "description": "Source of the memory",
                    "default": "api"
                }
            },
            "required": ["project_id", "role"]
        }


class MemoryPruneTool(Tool):
    """Tool for pruning obsolete memories"""
    
    def __init__(self, config: Optional[MemoryServerConfig] = None):
        super().__init__(
            name="memory_prune",
            description="Prune obsolete memories based on age and relevance"
        )
        self.config = config or MemoryServerConfig.from_env()
    
    def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Prune obsolete memories
        
        Parameters:
        - project_id (str, optional): Filter by project ID
        - max_age_days (int, optional): Maximum age in days (default: 30)
        """
        try:
            # Prepare request parameters
            prune_params = {}
            if "project_id" in params:
                prune_params["project_id"] = params["project_id"]
            if "max_age_days" in params:
                prune_params["max_age_days"] = params["max_age_days"]
            
            # Make API request
            response = requests.post(
                f"{self.config.base_url}/prune",
                params=prune_params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Format output
            output = f"Memory pruning completed!\n"
            output += f"Pruned: {result['pruned']} memories\n"
            output += f"Kept: {result['kept']} memories\n"
            output += f"Total processed: {result['total_processed']} memories"
            
            return ToolResponse(
                success=True,
                output=output,
                structured_data=result
            )
            
        except requests.exceptions.RequestException as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Failed to connect to memory server: {str(e)}"
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Memory pruning failed: {str(e)}"
            )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Filter by project ID (optional)"
                },
                "max_age_days": {
                    "type": "integer",
                    "description": "Maximum age in days for memories to keep",
                    "default": 30,
                    "minimum": 1
                }
            },
            "required": []
        }


class MemoryStatusTool(Tool):
    """Tool for checking memory server status"""
    
    def __init__(self, config: Optional[MemoryServerConfig] = None):
        super().__init__(
            name="memory_status",
            description="Check memory server status and indexing progress"
        )
        self.config = config or MemoryServerConfig.from_env()
    
    def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Check memory server status"""
        try:
            # Make API request
            response = requests.get(
                f"{self.config.base_url}/status",
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Format output
            output_lines = ["Memory Server Status:"]
            
            # Indexing status
            indexing = result.get("indexing", {})
            output_lines.append(f"Indexing Status: {indexing.get('status', 'unknown')}")
            
            if indexing.get("status") == "indexing":
                progress = indexing.get("progress", 0) * 100
                output_lines.append(f"Progress: {progress:.1f}% ({indexing.get('files_processed', 0)}/{indexing.get('total_files', 0)} files)")
                if indexing.get("current_file"):
                    output_lines.append(f"Current file: {indexing['current_file']}")
            elif indexing.get("status") == "completed":
                output_lines.append(f"Indexed {indexing.get('files_processed', 0)} files")
            elif indexing.get("status") == "error":
                output_lines.append(f"Error: {indexing.get('error_message', 'Unknown error')}")
            
            # Codebase info
            if result.get("codebase_path"):
                output_lines.append(f"Codebase: {result['codebase_path']}")
            
            # Conversation projects
            projects = result.get("conversation_projects", [])
            if projects:
                output_lines.append(f"Active projects: {', '.join(projects)}")
            
            return ToolResponse(
                success=True,
                output="\n".join(output_lines),
                structured_data=result
            )
            
        except requests.exceptions.RequestException as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Failed to connect to memory server: {str(e)}"
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                output="",
                structured_data={},
                errors=f"Status check failed: {str(e)}"
            )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


# Tool registry for easy access
MEMORY_TOOLS = {
    "memory_search": MemorySearchTool,
    "memory_store": MemoryStoreTool,
    "memory_prune": MemoryPruneTool,
    "memory_status": MemoryStatusTool
}


def create_memory_tools(config: Optional[MemoryServerConfig] = None) -> Dict[str, Tool]:
    """Create all memory tools with the given configuration"""
    return {
        name: tool_class(config)
        for name, tool_class in MEMORY_TOOLS.items()
    }