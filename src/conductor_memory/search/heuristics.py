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
    
    # Implementation details for each method/function (Phase 1 enhancement)
    method_details: List['MethodImplementationDetail'] = field(default_factory=list)
    
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
            # Include method implementation details
            'method_details': [md.to_dict() for md in self.method_details],
            'method_details_count': len(self.method_details),
        }


@dataclass
class MethodImplementationDetail:
    """
    Implementation details extracted from a method/function body.
    
    Used for verification queries ("does X use pattern Y?") and 
    implementation detail queries ("how does method X work?").
    """
    name: str
    signature: str  # Full signature with type hints
    
    # Call patterns (no predefined list - extract everything)
    internal_calls: List[str] = field(default_factory=list)    # self.method(), self.attr.method()
    external_calls: List[str] = field(default_factory=list)    # module.func(), Class.method()
    
    # Data access patterns
    attribute_reads: List[str] = field(default_factory=list)   # self._df, self.config.value
    attribute_writes: List[str] = field(default_factory=list)  # self._cache = x
    subscript_access: List[str] = field(default_factory=list)  # df[key], series.iloc[idx], dict[key]
    
    # Parameter usage
    parameters_used: List[str] = field(default_factory=list)   # Which params appear in method body
    
    # Structural signals
    has_loop: bool = False
    has_conditional: bool = False
    has_try_except: bool = False
    is_async: bool = False
    line_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage."""
        return {
            'name': self.name,
            'signature': self.signature,
            'internal_calls': self.internal_calls,
            'external_calls': self.external_calls,
            'attribute_reads': self.attribute_reads,
            'attribute_writes': self.attribute_writes,
            'subscript_access': self.subscript_access,
            'parameters_used': self.parameters_used,
            'has_loop': self.has_loop,
            'has_conditional': self.has_conditional,
            'has_try_except': self.has_try_except,
            'is_async': self.is_async,
            'line_count': self.line_count,
        }
    
    def to_searchable_text(self) -> str:
        """
        Format implementation signals as searchable text.
        
        This text is appended to chunk content so signals become
        searchable via existing hybrid search.
        """
        lines = ["[Implementation Signals]"]
        
        # Combine calls for readability
        all_calls = self.internal_calls + self.external_calls
        if all_calls:
            lines.append(f"Calls: {', '.join(all_calls)}")
        
        if self.attribute_reads:
            lines.append(f"Reads: {', '.join(self.attribute_reads)}")
        
        if self.attribute_writes:
            lines.append(f"Writes: {', '.join(self.attribute_writes)}")
        
        if self.subscript_access:
            lines.append(f"Subscripts: {', '.join(self.subscript_access)}")
        
        if self.parameters_used:
            lines.append(f"Parameters used: {', '.join(self.parameters_used)}")
        
        # Structural signals as compact list
        structural = []
        if self.has_loop:
            structural.append("loop")
        if self.has_conditional:
            structural.append("conditional")
        if self.has_try_except:
            structural.append("try-except")
        if self.is_async:
            structural.append("async")
        
        if structural:
            lines.append(f"Structure: {', '.join(structural)}")
        
        return '\n'.join(lines)


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
            
            # Extract method implementation details (Phase 1 enhancement)
            self._extract_method_implementation_details(tree, code_bytes, lang_config, metadata)
            
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
    
    def _extract_method_implementation_details(
        self,
        tree,
        code: bytes,
        lang_config,
        metadata: HeuristicMetadata
    ):
        """
        Extract implementation details from all method/function bodies.
        
        Uses the language config's implementation query to extract:
        - Method calls (internal/external)
        - Attribute reads/writes
        - Subscript access patterns
        - Structural signals (loops, conditionals, try-except, async)
        """
        try:
            # Get the implementation query for this language
            impl_query_string = lang_config.get_implementation_query()
            
            if not impl_query_string.strip():
                logger.debug(f"No implementation query for {lang_config.name}")
                return
            
            # First, find all function/method nodes to process individually
            func_nodes = self._find_function_nodes(tree.root_node, lang_config)
            
            if not func_nodes:
                logger.debug(f"No function nodes found in file")
                return
            
            # Create query for implementation patterns
            impl_query = Query(self._languages[lang_config.name], impl_query_string)
            
            # Process each function/method
            for func_node, func_name, signature in func_nodes:
                detail = self._extract_single_method_details(
                    func_node, func_name, signature, code, impl_query, lang_config
                )
                if detail:
                    metadata.method_details.append(detail)
                    
        except Exception as e:
            logger.debug(f"Method implementation extraction failed for {lang_config.name}: {e}")
    
    def _find_function_nodes(self, root_node: Node, lang_config) -> List[Tuple[Node, str, str]]:
        """
        Find all function/method definition nodes in the AST.
        
        Returns:
            List of (node, name, signature) tuples
        """
        func_nodes = []
        
        # Determine which node types are functions/methods for this language
        target_types = set(lang_config.function_types + lang_config.method_types)
        
        def walk(node: Node):
            if node.type in target_types:
                # Extract function name and signature
                name = self._extract_function_name(node, lang_config)
                signature = lang_config.extract_method_signature(node, b'')  # Initial signature
                if name:
                    func_nodes.append((node, name, signature))
            
            # Recurse into children
            for child in node.children:
                walk(child)
        
        walk(root_node)
        return func_nodes
    
    def _extract_function_name(self, node: Node, lang_config) -> Optional[str]:
        """Extract function/method name from a definition node."""
        try:
            # Look for name identifier in children based on language
            for child in node.children:
                if child.type == 'identifier' or child.type == 'name':
                    return child.text.decode('utf-8', errors='replace') if child.text else None
                
                # For languages with specific name node types
                if child.type in ['field_identifier', 'simple_identifier', 'constant']:
                    return child.text.decode('utf-8', errors='replace') if child.text else None
                
                # Python: look for name field
                if lang_config.name == 'python' and child.type == 'identifier':
                    # Check if this is the name position (usually first identifier after 'def')
                    return child.text.decode('utf-8', errors='replace') if child.text else None
            
            # Fallback: try to extract from first line of node text
            if node.text:
                text = node.text.decode('utf-8', errors='replace')
                # Common patterns: "def name(", "func name(", "function name("
                import re
                patterns = [
                    r'(?:def|func|function|fun)\s+(\w+)',
                    r'(\w+)\s*\(',
                ]
                for pattern in patterns:
                    match = re.search(pattern, text.split('\n')[0])
                    if match:
                        return match.group(1)
            
            return None
        except Exception:
            return None
    
    def _extract_single_method_details(
        self,
        func_node: Node,
        func_name: str,
        signature: str,
        code: bytes,
        impl_query: Query,
        lang_config
    ) -> Optional[MethodImplementationDetail]:
        """
        Extract implementation details from a single method/function body.
        
        Args:
            func_node: The AST node for the function/method
            func_name: The name of the function/method
            signature: The function signature
            code: Full source code bytes
            impl_query: Compiled tree-sitter query for implementation patterns
            lang_config: Language configuration
            
        Returns:
            MethodImplementationDetail with all extracted signals
        """
        try:
            # Extract signature properly
            full_signature = self._extract_signature(func_node, code, is_class=False)
            if not full_signature:
                full_signature = signature
            
            # Initialize detail object
            detail = MethodImplementationDetail(
                name=func_name,
                signature=full_signature,
                line_count=func_node.end_point[0] - func_node.start_point[0] + 1
            )
            
            # Extract parameters from signature
            parameters = self._extract_parameters_from_signature(full_signature, lang_config)
            
            # Check for async
            detail.is_async = self._is_async_function(func_node, code, lang_config)
            
            # Run implementation query on this function's subtree
            cursor = QueryCursor(impl_query)
            matches = cursor.matches(func_node)
            
            # Track parameters used in body
            param_set = set(parameters)
            params_used = set()
            
            # Process all captures
            for match_id, captures in matches:
                self._process_implementation_capture(
                    captures, code, detail, param_set, params_used, lang_config
                )
            
            # Set parameters_used
            detail.parameters_used = sorted(params_used)
            
            # Deduplicate lists
            detail.internal_calls = sorted(set(detail.internal_calls))
            detail.external_calls = sorted(set(detail.external_calls))
            detail.attribute_reads = sorted(set(detail.attribute_reads))
            detail.attribute_writes = sorted(set(detail.attribute_writes))
            detail.subscript_access = sorted(set(detail.subscript_access))
            
            return detail
            
        except Exception as e:
            logger.debug(f"Failed to extract details for {func_name}: {e}")
            return None
    
    def _extract_parameters_from_signature(self, signature: str, lang_config) -> List[str]:
        """Extract parameter names from a function signature."""
        params = []
        try:
            # Find content within parentheses
            import re
            paren_match = re.search(r'\(([^)]*)\)', signature)
            if not paren_match:
                return params
            
            param_str = paren_match.group(1)
            
            # Split by comma and extract param names
            for part in param_str.split(','):
                part = part.strip()
                if not part:
                    continue
                
                # Handle different param formats:
                # Python: param, param: type, param: type = default, *args, **kwargs
                # Java/C#: Type param, final Type param
                # Go: param Type, param, param2 Type
                
                # Remove default values
                if '=' in part:
                    part = part.split('=')[0].strip()
                
                # Remove type annotations (Python style)
                if ':' in part:
                    part = part.split(':')[0].strip()
                
                # Handle *args, **kwargs
                part = part.lstrip('*')
                
                # For Java/C#/Go style, get the last word (the parameter name)
                words = part.split()
                if words:
                    # Filter out type keywords and modifiers
                    type_keywords = {'final', 'const', 'var', 'val', 'let', 'mut', 'ref', 'out', 'in', 'params'}
                    param_name = words[-1]  # Last word is usually the param name
                    
                    # Clean up array brackets and other syntax
                    param_name = re.sub(r'[\[\]<>]', '', param_name).strip()
                    
                    if param_name and param_name not in type_keywords:
                        params.append(param_name)
            
            return params
        except Exception:
            return params
    
    def _is_async_function(self, func_node: Node, code: bytes, lang_config) -> bool:
        """Check if a function is async."""
        try:
            func_text = code[func_node.start_byte:func_node.end_byte].decode('utf-8', errors='replace')
            first_line = func_text.split('\n')[0].lower()
            
            # Check for async keyword in various positions
            async_patterns = ['async ', 'async def', 'async function', 'async fun']
            for pattern in async_patterns:
                if pattern in first_line:
                    return True
            
            # Check node type for languages with specific async node types
            if 'async' in func_node.type.lower():
                return True
            
            return False
        except Exception:
            return False
    
    def _process_implementation_capture(
        self,
        captures: Dict[str, List[Node]],
        code: bytes,
        detail: MethodImplementationDetail,
        param_set: set,
        params_used: set,
        lang_config
    ):
        """Process captured nodes from implementation query."""
        for capture_name, nodes in captures.items():
            for node in nodes:
                try:
                    node_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
                    
                    # Check if any parameter appears in this node
                    for param in param_set:
                        if param in node_text:
                            params_used.add(param)
                    
                    # Process based on capture type
                    if capture_name.startswith('call.'):
                        self._process_call_capture(capture_name, node, node_text, code, detail, captures)
                    elif capture_name.startswith('subscript.'):
                        self._process_subscript_capture(capture_name, node, node_text, code, detail, captures)
                    elif capture_name.startswith('access.'):
                        self._process_access_capture(capture_name, node, node_text, code, detail, captures)
                    elif capture_name.startswith('write.'):
                        self._process_write_capture(capture_name, node, node_text, code, detail, captures)
                    elif capture_name.startswith('structure.'):
                        self._process_structure_capture(capture_name, detail)
                        
                except Exception as e:
                    logger.debug(f"Error processing capture {capture_name}: {e}")
    
    def _process_call_capture(
        self,
        capture_name: str,
        node: Node,
        node_text: str,
        code: bytes,
        detail: MethodImplementationDetail,
        captures: Dict[str, List[Node]]
    ):
        """Process method/function call captures."""
        if capture_name == 'call.method':
            # Get the full call expression
            full_call = node_text
            
            # Get receiver and method name from captures
            receiver_text = ""
            method_name = ""
            
            if 'call.receiver' in captures and captures['call.receiver']:
                receiver_node = captures['call.receiver'][0]
                receiver_text = code[receiver_node.start_byte:receiver_node.end_byte].decode('utf-8', errors='replace')
            
            if 'call.method_name' in captures and captures['call.method_name']:
                method_node = captures['call.method_name'][0]
                method_name = code[method_node.start_byte:method_node.end_byte].decode('utf-8', errors='replace')
            
            # Classify as internal (self.) or external
            if receiver_text.startswith('self') or receiver_text == 'this':
                # Internal call
                if receiver_text == 'self' or receiver_text == 'this':
                    call_repr = method_name
                else:
                    # self.attr.method() -> attr.method
                    call_repr = f"{receiver_text.replace('self.', '').replace('this.', '')}.{method_name}" if method_name else receiver_text.replace('self.', '').replace('this.', '')
                detail.internal_calls.append(call_repr)
            else:
                # External call
                call_repr = f"{receiver_text}.{method_name}" if method_name else receiver_text
                detail.external_calls.append(call_repr)
                
        elif capture_name == 'call.function':
            # Simple function call
            func_name = ""
            if 'call.function_name' in captures and captures['call.function_name']:
                func_node = captures['call.function_name'][0]
                func_name = code[func_node.start_byte:func_node.end_byte].decode('utf-8', errors='replace')
            else:
                func_name = node_text.split('(')[0] if '(' in node_text else node_text
            
            detail.external_calls.append(func_name)
    
    def _process_subscript_capture(
        self,
        capture_name: str,
        node: Node,
        node_text: str,
        code: bytes,
        detail: MethodImplementationDetail,
        captures: Dict[str, List[Node]]
    ):
        """Process subscript/index access captures."""
        if capture_name in ['subscript.simple', 'subscript.attribute']:
            # Capture the full subscript expression
            # e.g., "df[key]", "self.data.iloc[idx]"
            full_subscript = node_text
            
            # Clean up to reasonable length (avoid capturing huge slices)
            if len(full_subscript) > 100:
                # Truncate but keep structure
                full_subscript = full_subscript[:97] + "..."
            
            detail.subscript_access.append(full_subscript)
    
    def _process_access_capture(
        self,
        capture_name: str,
        node: Node,
        node_text: str,
        code: bytes,
        detail: MethodImplementationDetail,
        captures: Dict[str, List[Node]]
    ):
        """Process attribute access captures."""
        if capture_name == 'access.attribute':
            # Full attribute access expression
            full_access = node_text
            
            # Filter: only track self/this accesses as attribute reads
            # (Others are likely part of calls or external accesses)
            if full_access.startswith('self.') or full_access.startswith('this.'):
                # Avoid duplicating call receivers - check if this is part of a call
                parent = node.parent
                if parent and parent.type not in ['call', 'call_expression']:
                    detail.attribute_reads.append(full_access)
    
    def _process_write_capture(
        self,
        capture_name: str,
        node: Node,
        node_text: str,
        code: bytes,
        detail: MethodImplementationDetail,
        captures: Dict[str, List[Node]]
    ):
        """Process attribute write captures."""
        if capture_name in ['write.attribute', 'write.augmented']:
            # Get the target of the write
            target = ""
            
            if 'write.target_object' in captures and captures['write.target_object']:
                obj_node = captures['write.target_object'][0]
                obj_text = code[obj_node.start_byte:obj_node.end_byte].decode('utf-8', errors='replace')
                
                if 'write.attr_name' in captures and captures['write.attr_name']:
                    attr_node = captures['write.attr_name'][0]
                    attr_name = code[attr_node.start_byte:attr_node.end_byte].decode('utf-8', errors='replace')
                    target = f"{obj_text}.{attr_name}"
            elif 'write.aug_object' in captures and captures['write.aug_object']:
                obj_node = captures['write.aug_object'][0]
                obj_text = code[obj_node.start_byte:obj_node.end_byte].decode('utf-8', errors='replace')
                
                if 'write.aug_attr' in captures and captures['write.aug_attr']:
                    attr_node = captures['write.aug_attr'][0]
                    attr_name = code[attr_node.start_byte:attr_node.end_byte].decode('utf-8', errors='replace')
                    target = f"{obj_text}.{attr_name}"
            
            if target and (target.startswith('self.') or target.startswith('this.')):
                detail.attribute_writes.append(target)
    
    def _process_structure_capture(self, capture_name: str, detail: MethodImplementationDetail):
        """Process structural signal captures."""
        if capture_name in ['structure.for_loop', 'structure.while_loop', 'structure.do_while',
                            'structure.list_comp', 'structure.dict_comp', 'structure.generator',
                            'structure.enhanced_for', 'structure.foreach_loop', 'structure.range',
                            'structure.until_loop', 'structure.repeat_while']:
            detail.has_loop = True
        elif capture_name in ['structure.conditional', 'structure.switch', 'structure.ternary', 
                              'structure.when', 'structure.guard', 'structure.unless']:
            detail.has_conditional = True
        elif capture_name in ['structure.try_except', 'structure.try_catch', 'structure.do_catch',
                              'structure.try_with_resources', 'structure.begin']:
            detail.has_try_except = True
        elif capture_name in ['structure.await', 'structure.async']:
            detail.is_async = True