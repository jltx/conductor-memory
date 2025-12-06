"""
Heuristic extraction for structured metadata without LLM calls.

Extracts class/interface names, function signatures, docstrings, annotations,
and import statements from source code using tree-sitter parsers.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tree_sitter import Node, Query, QueryCursor, Language, Parser
except ImportError:
    raise ImportError("tree-sitter not installed. Run: pip install tree-sitter")

from .parsers.language_configs import get_language_config, get_language_from_extension

logger = logging.getLogger(__name__)


@dataclass
class HeuristicMetadata:
    """Structured metadata extracted via heuristic analysis."""
    file_path: str
    language: str
    
    # Classes and interfaces
    classes: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Functions and methods
    functions: List[Dict[str, Any]] = field(default_factory=list)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    
    # Documentation
    docstrings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Annotations and imports
    annotations: List[str] = field(default_factory=list)
    imports: List[Dict[str, str]] = field(default_factory=list)
    
    # Additional metadata
    exports: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage."""
        return {
            'file_path': self.file_path,
            'language': self.language,
            'class_count': len(self.classes),
            'function_count': len(self.functions),
            'method_count': len(self.methods),
            'import_count': len(self.imports),
            'has_annotations': bool(self.annotations),
            'has_docstrings': bool(self.docstrings),
            'class_names': [cls['name'] for cls in self.classes],
            'function_names': [func['name'] for func in self.functions],
            'import_modules': [imp.get('module', '') for imp in self.imports],
            'annotations': self.annotations,
        }


class HeuristicExtractor:
    """Extracts structured metadata from source code without LLM calls."""
    
    def __init__(self):
        self._parsers: Dict[str, Parser] = {}  # Cached parsers
        self._languages: Dict[str, Language] = {}
    
    def extract_file_metadata(
        self, 
        file_path: str, 
        content: str
    ) -> Optional[HeuristicMetadata]:
        """
        Extract structured metadata from a file.
        
        Args:
            file_path: Path to the source file
            content: File content as string
            
        Returns:
            HeuristicMetadata object or None if extraction fails
        """
        try:
            # Detect language
            lang_name = get_language_from_extension(file_path)
            if not lang_name:
                logger.debug(f"Unsupported file extension: {file_path}")
                return None
            
            # Get language configuration
            lang_config = get_language_config(lang_name)
            if not lang_config:
                logger.debug(f"No configuration for language: {lang_name}")
                return None
            
            # Parse with tree-sitter
            parser = self._get_parser(lang_name, lang_config)
            code_bytes = content.encode('utf-8')
            tree = parser.parse(code_bytes)
            
            if not tree.root_node:
                logger.warning(f"Failed to parse AST for: {file_path}")
                return None
            
            # Initialize metadata object
            metadata = HeuristicMetadata(
                file_path=file_path,
                language=lang_name
            )
            
            # Extract different types of metadata
            self._extract_definitions(tree, code_bytes, lang_config, metadata)
            self._extract_annotations(tree, code_bytes, lang_config, metadata)
            self._extract_docstrings(tree, code_bytes, lang_config, metadata)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Heuristic extraction failed for {file_path}: {e}")
            return None
    
    def _get_parser(self, lang_name: str, lang_config) -> Parser:
        """Get or create cached parser for language."""
        if lang_name not in self._parsers:
            if lang_name not in self._languages:
                self._languages[lang_name] = Language(lang_config.module.language())
            self._parsers[lang_name] = Parser(self._languages[lang_name])
        return self._parsers[lang_name]
    
    def _extract_definitions(
        self, 
        tree, 
        code: bytes, 
        lang_config, 
        metadata: HeuristicMetadata
    ):
        """Extract all definitions using the language config's definition query."""
        try:
            # Use the language config's definition query
            query_string = lang_config.definition_query
            
            if not query_string.strip():
                return
            
            # Execute query
            query = Query(self._languages[lang_config.name], query_string)
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)
            
            for match_id, captures in matches:
                self._process_definition_match(captures, code, lang_config, metadata)
                
        except Exception as e:
            logger.debug(f"Definition extraction failed for {lang_config.name}: {e}")
    
    def _process_definition_match(
        self, 
        captures: Dict[str, List[Node]], 
        code: bytes, 
        lang_config,
        metadata: HeuristicMetadata
    ):
        """Process a definition match from the language config query."""
        try:
            def_node = None
            name_node = None
            node_type = None
            
            for capture_name, nodes in captures.items():
                if not nodes:
                    continue
                    
                node = nodes[0]
                
                if capture_name.endswith('.def'):
                    def_node = node
                    node_type = capture_name.replace('.def', '')
                elif capture_name.endswith('.name'):
                    name_node = node
                    # Extract node type from capture name (e.g., 'class.name' -> 'class')
                    if '.' in capture_name:
                        node_type = capture_name.split('.')[0]
                elif capture_name == 'import.name':
                    # Special case for Ruby imports - check if it's actually require/load
                    parent_node = node.parent
                    if parent_node and parent_node.type == 'call':
                        call_text = code[parent_node.start_byte:parent_node.end_byte].decode('utf-8', errors='replace')
                        if 'require' in call_text or 'load' in call_text:
                            def_node = parent_node
                            name_node = node
                            node_type = 'import'
            
            if not def_node:
                return
            
            # Extract name
            if name_node:
                name = code[name_node.start_byte:name_node.end_byte].decode('utf-8', errors='replace')
            else:
                # For imports and other nodes without explicit names
                name = node_type or def_node.type
            
            # Create definition info
            definition_info = {
                'name': name,
                'start_line': def_node.start_point[0] + 1,
                'end_line': def_node.end_point[0] + 1,
                'node_type': def_node.type,
                'signature': self._extract_signature(def_node, code, is_class=(node_type in ['class', 'interface', 'struct', 'type', 'obj', 'module']))
            }
            
            # Categorize the definition
            if node_type in ['class', 'struct', 'type', 'obj']:
                metadata.classes.append(definition_info)
            elif node_type in ['interface', 'module']:
                metadata.interfaces.append(definition_info)
            elif node_type in ['func', 'function']:
                metadata.functions.append(definition_info)
            elif node_type in ['method']:
                metadata.methods.append(definition_info)
            elif node_type == 'import':
                # Process import statement
                import_text = code[def_node.start_byte:def_node.end_byte].decode('utf-8', errors='replace')
                import_info = self._parse_import_statement(import_text, lang_config.name)
                if import_info:
                    metadata.imports.append(import_info)
                    
        except Exception as e:
            logger.debug(f"Failed to process definition match: {e}")
    
    def _extract_annotations(
        self, 
        tree, 
        code: bytes, 
        lang_config, 
        metadata: HeuristicMetadata
    ):
        """Extract annotations and decorators."""
        try:
            # Use language-specific annotation query
            query_string = lang_config.get_annotation_query()
            
            if not query_string.strip():
                return
            
            # Execute query
            query = Query(self._languages[lang_config.name], query_string)
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)
            
            for match_id, captures in matches:
                for capture_name, nodes in captures.items():
                    for node in nodes:
                        annotation_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
                        
                        # Clean up annotation text based on language
                        cleaned_text = self._clean_annotation_text(annotation_text, lang_config.name)
                        
                        if cleaned_text and cleaned_text not in metadata.annotations:
                            metadata.annotations.append(cleaned_text)
                            
        except Exception as e:
            logger.debug(f"Annotation extraction failed for {lang_config.name}: {e}")
    
    def _clean_annotation_text(self, text: str, language: str) -> str:
        """Clean annotation text based on language conventions."""
        text = text.strip()
        
        if language == 'python':
            # Python decorators: @decorator_name
            if text.startswith('@'):
                return text
            else:
                # Extract decorator name from decorator_list
                lines = text.split('\n')
                decorators = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('@'):
                        decorators.append(line)
                return ', '.join(decorators) if decorators else text
        
        elif language == 'java':
            # Java annotations: @Annotation
            if text.startswith('@'):
                return text.split('\n')[0].strip()  # Take first line only
            else:
                # Extract from annotation blocks
                if '@' in text:
                    parts = text.split('@')
                    if len(parts) > 1:
                        return '@' + parts[1].split()[0]
        
        elif language == 'csharp':
            # C# attributes: [Attribute]
            if text.startswith('[') and text.endswith(']'):
                return text
            elif '[' in text and ']' in text:
                start = text.find('[')
                end = text.find(']', start)
                if start != -1 and end != -1:
                    return text[start:end+1]
        
        # For other languages or fallback
        return text.split('\n')[0].strip() if text else ""
    
    def _extract_docstrings(
        self, 
        tree, 
        code: bytes, 
        lang_config, 
        metadata: HeuristicMetadata
    ):
        """Extract documentation strings."""
        try:
            # Query for string literals that could be docstrings
            docstring_types = [
                'string_literal',
                'expression_statement',
                'comment'
            ]
            
            query_parts = []
            for doc_type in docstring_types:
                query_parts.append(f"({doc_type}) @docstring")
            
            query_string = '\n'.join(query_parts)
            
            # Execute query
            query = Query(self._languages[lang_config.name], query_string)
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)
            
            for match_id, captures in matches:
                for capture_name, nodes in captures.items():
                    for node in nodes:
                        self._process_potential_docstring(node, code, metadata)
                        
        except Exception as e:
            logger.debug(f"Docstring extraction failed for {lang_config.name}: {e}")
    

    
    def _process_potential_docstring(
        self, 
        node: Node, 
        code: bytes, 
        metadata: HeuristicMetadata
    ):
        """Process a potential docstring node."""
        try:
            text = code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
            text = text.strip()
            
            # Basic heuristics to identify docstrings
            is_docstring = False
            docstring_content = ""
            
            # Multi-line strings (Python, etc.)
            if ('"""' in text or "'''" in text):
                is_docstring = True
                # Extract content between triple quotes
                if '"""' in text:
                    parts = text.split('"""')
                    if len(parts) >= 2:
                        docstring_content = parts[1].strip()
                elif "'''" in text:
                    parts = text.split("'''")
                    if len(parts) >= 2:
                        docstring_content = parts[1].strip()
                else:
                    docstring_content = text
            
            # KDoc/Javadoc style
            elif text.startswith('/**') and text.endswith('*/'):
                is_docstring = True
                docstring_content = text[3:-2].strip()
            
            # Single-line docstring patterns
            elif (text.startswith('"""') and text.endswith('"""') and len(text) > 6):
                is_docstring = True
                docstring_content = text[3:-3].strip()
            elif (text.startswith("'''") and text.endswith("'''") and len(text) > 6):
                is_docstring = True
                docstring_content = text[3:-3].strip()
            
            # Substantial comments that might be documentation
            elif (text.startswith('//') or text.startswith('#')):
                # Only consider substantial comments as docstrings
                if len(text) > 30 and any(keyword in text.lower() for keyword in 
                                        ['description', 'param', 'return', 'note', 'todo', 'fixme']):
                    is_docstring = True
                    docstring_content = text
            
            if is_docstring and docstring_content:
                docstring_info = {
                    'content': docstring_content,
                    'line': node.start_point[0] + 1,
                    'type': node.type,
                    'raw_text': text
                }
                metadata.docstrings.append(docstring_info)
                
        except Exception as e:
            logger.debug(f"Failed to process docstring: {e}")
    
    def _extract_signature(self, node: Node, code: bytes, is_class: bool = False) -> str:
        """Extract signature from a node."""
        try:
            full_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
            lines = full_text.split('\n')
            
            if is_class:
                # For classes, extract the class declaration line(s)
                signature_lines = []
                for line in lines:
                    signature_lines.append(line.strip())
                    if ':' in line or '{' in line:
                        break
                
                signature = ' '.join(signature_lines)
                # Clean up the signature
                if ':' in signature:
                    signature = signature.split(':')[0].strip()
                if '{' in signature:
                    signature = signature.split('{')[0].strip()
                
                return signature
            else:
                # For functions, extract signature until opening brace/colon
                signature_lines = []
                for line in lines:
                    signature_lines.append(line.strip())
                    if ':' in line or '{' in line or ';' in line:
                        # For function signatures, stop at the colon/brace but include it in parsing
                        if ':' in line:
                            parts = line.split(':')
                            signature_lines[-1] = parts[0].strip()
                        elif '{' in line:
                            parts = line.split('{')
                            signature_lines[-1] = parts[0].strip()
                        elif ';' in line:
                            parts = line.split(';')
                            signature_lines[-1] = parts[0].strip()
                        break
                
                return ' '.join(signature_lines)
                
        except Exception:
            return ""
    
    def _parse_import_statement(self, import_text: str, language: str) -> Optional[Dict[str, str]]:
        """Parse import statement to extract module information."""
        try:
            import_text = import_text.strip()
            
            if language == 'python':
                if import_text.startswith('from '):
                    # from module import ...
                    parts = import_text.split()
                    if len(parts) >= 2:
                        return {
                            'statement': import_text,
                            'module': parts[1],
                            'type': 'from_import'
                        }
                elif import_text.startswith('import '):
                    # import module
                    module = import_text[7:].split()[0].strip()
                    return {
                        'statement': import_text,
                        'module': module,
                        'type': 'import'
                    }
            
            elif language == 'java':
                if import_text.startswith('import '):
                    module = import_text[7:].strip(';').strip()
                    return {
                        'statement': import_text,
                        'module': module,
                        'type': 'import'
                    }
            
            elif language == 'kotlin':
                if import_text.startswith('import '):
                    module = import_text[7:].strip()
                    return {
                        'statement': import_text,
                        'module': module,
                        'type': 'import'
                    }
            
            elif language == 'go':
                if 'import' in import_text:
                    # Go import can be single line or block
                    # Extract package path from quotes
                    if '"' in import_text:
                        # Handle both single imports and import blocks
                        lines = import_text.split('\n')
                        modules = []
                        for line in lines:
                            line = line.strip()
                            if '"' in line:
                                start = line.find('"')
                                end = line.rfind('"')
                                if start != -1 and end != -1 and start != end:
                                    module = line[start+1:end]
                                    modules.append(module)
                        
                        if modules:
                            # For import blocks, return the first module found
                            # (individual imports will be processed separately)
                            return {
                                'statement': import_text.split('\n')[0].strip(),
                                'module': modules[0],
                                'type': 'import'
                            }
            
            elif language in ['c', 'objc']:
                if import_text.startswith('#include'):
                    # Extract header name
                    if '<' in import_text and '>' in import_text:
                        start = import_text.find('<')
                        end = import_text.find('>')
                        module = import_text[start+1:end]
                    elif '"' in import_text:
                        start = import_text.find('"')
                        end = import_text.rfind('"')
                        if start != end:
                            module = import_text[start+1:end]
                    else:
                        module = import_text[8:].strip()
                    
                    return {
                        'statement': import_text,
                        'module': module,
                        'type': 'include'
                    }
            
            elif language == 'csharp':
                if import_text.startswith('using '):
                    module = import_text[6:].strip(';').strip()
                    return {
                        'statement': import_text,
                        'module': module,
                        'type': 'using'
                    }
            
            elif language == 'swift':
                if import_text.startswith('import '):
                    module = import_text[7:].strip()
                    return {
                        'statement': import_text,
                        'module': module,
                        'type': 'import'
                    }
            
            elif language == 'ruby':
                # Only process actual require/load statements
                if (import_text.startswith('require') or 
                    import_text.startswith('load') or
                    'require(' in import_text or
                    'load(' in import_text):
                    
                    # Extract from quotes
                    if "'" in import_text:
                        start = import_text.find("'")
                        end = import_text.rfind("'")
                        if start != end:
                            module = import_text[start+1:end]
                        else:
                            return None
                    elif '"' in import_text:
                        start = import_text.find('"')
                        end = import_text.rfind('"')
                        if start != end:
                            module = import_text[start+1:end]
                        else:
                            return None
                    else:
                        # Try to extract module name after require
                        parts = import_text.replace('(', ' ').replace(')', ' ').split()
                        if len(parts) >= 2:
                            module = parts[1].strip("'\"")
                        else:
                            return None
                    
                    return {
                        'statement': import_text,
                        'module': module,
                        'type': 'require'
                    }
                else:
                    # Not an import statement, skip
                    return None
            
            # Fallback - return the statement as-is
            return {
                'statement': import_text,
                'module': import_text,
                'type': 'unknown'
            }
            
        except Exception as e:
            logger.debug(f"Failed to parse import statement '{import_text}': {e}")
            return None