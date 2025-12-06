"""
Server Configuration for MCP Memory Server

Supports multiple codebase indexing with a YAML/JSON configuration file.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodebaseConfig:
    """Configuration for a single codebase to index"""
    
    # Unique identifier for this codebase
    name: str
    
    # Path to the codebase root directory
    path: str
    
    # File extensions to index (defaults to common code extensions)
    extensions: List[str] = field(default_factory=lambda: [
        '.py', '.js', '.ts', '.tsx', '.jsx',
        '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php',
        '.md', '.txt', '.json', '.yaml', '.yml'
    ])
    
    # Patterns to ignore (directory or file patterns)
    ignore_patterns: List[str] = field(default_factory=lambda: [
        '__pycache__', '.git', 'node_modules', '.venv', 'venv',
        'build', 'dist', '.idea', '.vscode', '.pytest_cache',
        '*.pyc', '*.pyo', '*.egg-info'
    ])
    
    # Whether this codebase is enabled for indexing
    enabled: bool = True
    
    # Optional description
    description: str = ""
    
    def __post_init__(self):
        """Validate and normalize the configuration"""
        # Normalize path
        self.path = os.path.abspath(os.path.expanduser(self.path))
        
        # Ensure extensions start with dot
        self.extensions = [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in self.extensions
        ]
    
    def get_extension_set(self) -> Set[str]:
        """Get extensions as a set for efficient lookup"""
        return set(self.extensions)
    
    def should_ignore(self, path: str) -> bool:
        """
        Check if a path should be ignored based on ignore patterns.
        
        Pattern types:
        - `/pattern` - Root-relative: matches only at the codebase root
                       e.g., `/data` matches `data/file.csv` but NOT `src/data/file.py`
        - `pattern`  - Component match: matches the pattern as any path component
                       e.g., `__pycache__` matches `src/__pycache__/file.pyc`
        - `*.ext`    - Glob suffix: matches files ending with the pattern
                       e.g., `*.pyc` matches `any/path/file.pyc`
        
        Args:
            path: Relative path from the codebase root (e.g., "src/data/file.py")
        """
        path_str = str(path).replace('\\', '/')
        path_components = path_str.split('/')
        
        for pattern in self.ignore_patterns:
            if pattern.startswith('*'):
                # Glob-style suffix pattern (e.g., *.pyc)
                if path_str.endswith(pattern[1:]):
                    logger.debug(f"[{self.name}] Ignoring '{path_str}' - matched glob pattern '{pattern}'")
                    return True
            elif pattern.startswith('/'):
                # Root-relative pattern (e.g., /data matches only top-level data/)
                root_pattern = pattern[1:]  # Remove leading slash
                if path_components[0] == root_pattern:
                    logger.debug(f"[{self.name}] Ignoring '{path_str}' - matched root pattern '{pattern}'")
                    return True
            else:
                # Component match - pattern matches any path component
                if pattern in path_components:
                    logger.debug(f"[{self.name}] Ignoring '{path_str}' - matched component pattern '{pattern}'")
                    return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'path': self.path,
            'extensions': self.extensions,
            'ignore_patterns': self.ignore_patterns,
            'enabled': self.enabled,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodebaseConfig':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            path=data['path'],
            extensions=data.get('extensions', cls.__dataclass_fields__['extensions'].default_factory()),
            ignore_patterns=data.get('ignore_patterns', cls.__dataclass_fields__['ignore_patterns'].default_factory()),
            enabled=data.get('enabled', True),
            description=data.get('description', '')
        )


@dataclass
class ServerConfig:
    """Configuration for the MCP Memory Server"""
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Persistence directory for ChromaDB
    persist_directory: str = "./data/chroma"
    
    # List of codebases to index
    codebases: List[CodebaseConfig] = field(default_factory=list)
    
    # File watcher settings
    watch_interval: float = 5.0  # seconds between file change checks
    enable_file_watcher: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    def __post_init__(self):
        """Validate and normalize the configuration"""
        self.persist_directory = os.path.abspath(os.path.expanduser(self.persist_directory))
    
    def get_enabled_codebases(self) -> List[CodebaseConfig]:
        """Get list of enabled codebases"""
        return [cb for cb in self.codebases if cb.enabled]
    
    def get_codebase_by_name(self, name: str) -> Optional[CodebaseConfig]:
        """Get a codebase configuration by name"""
        for cb in self.codebases:
            if cb.name == name:
                return cb
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'host': self.host,
            'port': self.port,
            'persist_directory': self.persist_directory,
            'codebases': [cb.to_dict() for cb in self.codebases],
            'watch_interval': self.watch_interval,
            'enable_file_watcher': self.enable_file_watcher,
            'log_level': self.log_level,
            'embedding_model': self.embedding_model
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServerConfig':
        """Create from dictionary"""
        codebases = [
            CodebaseConfig.from_dict(cb) for cb in data.get('codebases', [])
        ]
        return cls(
            host=data.get('host', '127.0.0.1'),
            port=data.get('port', 8000),
            persist_directory=data.get('persist_directory', './data/chroma'),
            codebases=codebases,
            watch_interval=data.get('watch_interval', 5.0),
            enable_file_watcher=data.get('enable_file_watcher', True),
            log_level=data.get('log_level', 'INFO'),
            embedding_model=data.get('embedding_model', 'all-MiniLM-L6-v2')
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ServerConfig':
        """Load configuration from a JSON or YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine format based on extension
        if config_path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
        else:
            # Default to JSON
            data = json.loads(content)
        
        return cls.from_dict(data)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to a JSON or YAML file"""
        config_path = Path(config_path)
        
        data = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
                except ImportError:
                    raise ImportError("PyYAML is required to save YAML config files. Install with: pip install pyyaml")
            else:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved configuration to: {config_path}")
    
    @classmethod
    def create_default(cls, codebase_path: Optional[str] = None, codebase_name: str = "default") -> 'ServerConfig':
        """Create a default configuration, optionally with a single codebase"""
        config = cls()
        
        if codebase_path:
            config.codebases.append(CodebaseConfig(
                name=codebase_name,
                path=codebase_path,
                description=f"Codebase at {codebase_path}"
            ))
        
        return config


def generate_example_config(output_path: str = "memory_server_config.json") -> None:
    """Generate an example configuration file"""
    config = ServerConfig(
        host="127.0.0.1",
        port=8000,
        persist_directory="./data/chroma",
        codebases=[
            CodebaseConfig(
                name="project1",
                path="/path/to/project1",
                description="Main project codebase",
                extensions=['.py', '.js', '.ts', '.md'],
                ignore_patterns=['__pycache__', '.git', 'node_modules', 'venv']
            ),
            CodebaseConfig(
                name="project2",
                path="/path/to/project2",
                description="Secondary project",
                enabled=True
            )
        ],
        watch_interval=5.0,
        enable_file_watcher=True,
        log_level="INFO"
    )
    
    config.save_to_file(output_path)
    print(f"Example configuration saved to: {output_path}")
