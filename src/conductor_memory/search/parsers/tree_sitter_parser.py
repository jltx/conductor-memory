"""
Multi-language AST parser using tree-sitter.

Provides semantic chunking for 9+ programming languages with accurate
domain tagging, class summaries, and proper chunk boundaries.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from tree_sitter import Language, Parser, Tree, Node, Query, QueryCursor
except ImportError:
    raise ImportError("tree-sitter not installed. Run: pip install tree-sitter")

from .base import ContentParser, ParseError, UnsupportedLanguageError
from .language_configs import get_language_config, get_language_from_extension, LanguageConfig
from .domain_detector import detect_domain
from ..chunking import ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class Definition:
    """Represents a code definition (class, function, method, etc.)"""
    node_type: str  # Capture name prefix (e.g., 'class', 'func', 'method')
    ast_node_type: str  # Actual AST node type (e.g., 'class_definition', 'function_definition')
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    annotations: List[str]
    modifiers: List[str]
    parent_name: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None


class TreeSitterParser(ContentParser):
    """Multi-language AST parser using tree-sitter."""
    
    SPLIT_THRESHOLD = 100  # Lines before splitting class into methods
    
    def __init__(self):
        self._parsers: Dict[str, Parser] = {}  # Cached parsers
        self._languages: Dict[str, Language] = {}
        self._configs: Dict[str, LanguageConfig] = {}
    
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file extension."""
        try:
            ext = Path(file_path).suffix.lower()
            lang = get_language_from_extension(ext)
            return lang is not None
        except Exception:
            return False
    
    def parse(
        self, 
        content: str, 
        file_path: str, 
        commit_hash: Optional[str] = None
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Parse source code into semantic chunks.
        
        Args:
            content: Source code as string
            file_path: Path to the source file
            commit_hash: Optional git commit hash
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        try:
            # Detect language
            ext = Path(file_path).suffix.lower()
            lang_name = get_language_from_extension(ext)
            if not lang_name:
                raise UnsupportedLanguageError(f"Unsupported file extension: {ext}")
            
            # Get language configuration
            lang_config = self._get_language_config(lang_name)
            
            # Parse with tree-sitter
            parser = self._get_parser(lang_name, lang_config)
            code_bytes = content.encode('utf-8')
            tree = parser.parse(code_bytes)
            
            if not tree.root_node:
                raise ParseError("Failed to parse AST")
            
            # Extract definitions
            definitions = self._extract_definitions(tree, lang_config, code_bytes)
            
            # Group definitions by type
            classes = [d for d in definitions if self._is_class_definition(d, lang_config)]
            functions = [d for d in definitions if self._is_function_definition(d, lang_config)]
            imports = [d for d in definitions if self._is_import_definition(d, lang_config)]
            
            # Generate chunks
            chunks = []
            
            # Process classes
            for class_def in classes:
                chunks.extend(self._chunk_class(class_def, code_bytes, file_path, lang_config, lang_name))
            
            # Process standalone functions (exclude methods inside classes)
            for func_def in functions:
                # Check if function is contained within any class (by byte range)
                is_method = False
                for class_def in classes:
                    if (func_def.start_byte >= class_def.start_byte and 
                        func_def.end_byte <= class_def.end_byte):
                        is_method = True
                        break
                
                if not is_method:
                    chunks.append(self._chunk_function(func_def, code_bytes, file_path, lang_config, lang_name))
            
            # Process imports (combine into single chunk)
            if imports:
                chunks.append(self._chunk_imports(imports, code_bytes, file_path))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
            raise ParseError(f"Failed to parse {file_path}: {e}")
    
    def _get_language_config(self, lang_name: str) -> LanguageConfig:
        """Get or cache language configuration."""
        if lang_name not in self._configs:
            self._configs[lang_name] = get_language_config(lang_name)
        return self._configs[lang_name]
    
    def _get_parser(self, lang_name: str, lang_config: LanguageConfig) -> Parser:
        """Get or create cached parser for language."""
        if lang_name not in self._parsers:
            if lang_name not in self._languages:
                self._languages[lang_name] = Language(lang_config.module.language())
            self._parsers[lang_name] = Parser(self._languages[lang_name])
        return self._parsers[lang_name]
    
    def _extract_definitions(self, tree: Tree, lang_config: LanguageConfig, code: bytes) -> List[Definition]:
        """Extract all definitions from AST using language-specific queries."""
        definitions = []
        
        try:
            # Ensure language is available
            if lang_config.name not in self._languages:
                self._languages[lang_config.name] = Language(lang_config.module.language())
            
            # Create query and cursor (tree-sitter 0.25.x API)
            query = Query(self._languages[lang_config.name], lang_config.definition_query)
            cursor = QueryCursor(query)  # Pass query to cursor constructor
            
            # Execute query - returns list of (pattern_idx, {capture_name: [nodes]})
            matches = cursor.matches(tree.root_node)
            
            for match_id, captures in matches:
                definition = self._process_query_match(captures, code, lang_config)
                if definition:
                    definitions.append(definition)
                    
        except Exception as e:
            logger.warning(f"Query execution failed for {lang_config.name}: {e}")
            # Log the tree structure for debugging
            self._log_tree_structure(tree.root_node, lang_config.name)
        
        return definitions
    
    def _log_tree_structure(self, node: Node, lang_name: str, max_depth: int = 3):
        """Log the tree structure for debugging query issues."""
        def get_structure(n: Node, depth: int = 0) -> str:
            if depth > max_depth:
                return ""
            indent = "  " * depth
            children_str = ""
            for child in n.children:
                if child.is_named:
                    children_str += get_structure(child, depth + 1)
            return f"{indent}{n.type}\n{children_str}"
        
        structure = get_structure(node)
        logger.debug(f"AST structure for {lang_name} (first {max_depth} levels):\n{structure[:1000]}")
    
    def _process_query_match(self, captures: Dict[str, List[Node]], code: bytes, lang_config: LanguageConfig) -> Optional[Definition]:
        """Process a single query match into a Definition."""
        try:
            # Find the main definition node and name node
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
            
            if not def_node:
                return None
            
            # Extract name - from name_node if available, otherwise from def_node type
            if name_node:
                name = code[name_node.start_byte:name_node.end_byte].decode('utf-8', errors='replace')
            else:
                # For imports and other nodes without explicit names
                name = node_type or def_node.type
            
            # Extract annotations and modifiers
            annotations = self._extract_annotations(def_node, code, lang_config)
            modifiers = self._extract_modifiers(def_node, code, lang_config)
            
            # Extract docstring
            docstring = self._extract_docstring(def_node, code, lang_config)
            
            # Extract signature for functions/methods
            signature = None
            if node_type in ['function', 'method', 'func']:
                signature = lang_config.extract_method_signature(def_node, code)
            
            return Definition(
                node_type=node_type,
                ast_node_type=def_node.type,  # Store actual AST node type
                name=name,
                start_line=def_node.start_point[0] + 1,
                end_line=def_node.end_point[0] + 1,
                start_byte=def_node.start_byte,
                end_byte=def_node.end_byte,
                annotations=annotations,
                modifiers=modifiers,
                docstring=docstring,
                signature=signature
            )
            
        except Exception as e:
            logger.debug(f"Failed to process query match: {e}")
            return None
    
    def _extract_annotations(self, node: Node, code: bytes, lang_config: LanguageConfig) -> List[str]:
        """Extract annotations/decorators from a node."""
        annotations = []
        
        # Look for annotation nodes as siblings or children
        if hasattr(node, 'children'):
            for child in node.children:
                if child.type in ['decorator', 'annotation', 'marker_annotation', 'attribute']:
                    try:
                        annotation_text = code[child.start_byte:child.end_byte].decode('utf-8', errors='replace')
                        # Clean up annotation text
                        annotation_text = annotation_text.strip('@[]()').strip()
                        if annotation_text:
                            annotations.append(annotation_text)
                    except Exception:
                        pass
        
        return annotations
    
    def _extract_modifiers(self, node: Node, code: bytes, lang_config: LanguageConfig) -> List[str]:
        """Extract modifiers (public, private, static, etc.) from a node."""
        modifiers = []
        
        # Look for modifier nodes
        if hasattr(node, 'children'):
            for child in node.children:
                if child.type in ['modifiers', 'visibility_modifier', 'modifier']:
                    try:
                        modifier_text = code[child.start_byte:child.end_byte].decode('utf-8', errors='replace')
                        modifiers.extend(modifier_text.split())
                    except Exception:
                        pass
        
        return modifiers
    
    def _extract_docstring(self, node: Node, code: bytes, lang_config: LanguageConfig) -> Optional[str]:
        """Extract docstring/documentation from a node."""
        # Look for string literals at the beginning of the body
        if hasattr(node, 'children'):
            for child in node.children:
                if child.type == 'block' or child.type == 'body':
                    # Look for first string literal in the body
                    for grandchild in child.children:
                        if grandchild.type in ['string_literal', 'expression_statement']:
                            try:
                                text = code[grandchild.start_byte:grandchild.end_byte].decode('utf-8', errors='replace')
                                # Basic docstring detection
                                if ('"""' in text or "'''" in text or 
                                    text.strip().startswith('"') or 
                                    text.strip().startswith("'")):
                                    return text.strip()
                            except Exception:
                                pass
                            break
                    break
        
        return None
    
    def _is_class_definition(self, definition: Definition, lang_config: LanguageConfig) -> bool:
        """Check if definition is a class."""
        # node_type comes from capture names like 'class.def', 'struct.def', 'obj.def', 'type.def'
        return definition.node_type in ['class', 'struct', 'interface', 'object', 'obj', 'type']
    
    def _is_function_definition(self, definition: Definition, lang_config: LanguageConfig) -> bool:
        """Check if definition is a function/method."""
        # node_type comes from capture names like 'func.def', 'method.def'
        return definition.node_type in ['function', 'method', 'func']
    
    def _is_import_definition(self, definition: Definition, lang_config: LanguageConfig) -> bool:
        """Check if definition is an import."""
        return definition.node_type == 'import'
    
    def _chunk_class(
        self, 
        class_def: Definition, 
        code: bytes, 
        file_path: str, 
        lang_config: LanguageConfig,
        lang_name: str
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Chunk a class definition.
        
        If <100 lines: single chunk
        If >=100 lines: class summary + individual methods
        """
        line_count = class_def.end_line - class_def.start_line + 1
        
        if line_count < self.SPLIT_THRESHOLD:
            # Small class - single chunk
            return [self._create_single_class_chunk(class_def, code, file_path, lang_name)]
        else:
            # Large class - summary + methods
            chunks = []
            
            # Extract methods from this class
            methods = self._extract_class_methods(class_def, code, lang_config)
            
            # Create class summary
            summary_chunk = self._create_class_summary_chunk(class_def, methods, code, file_path, lang_config, lang_name)
            chunks.append(summary_chunk)
            
            # Create method chunks
            for method in methods:
                method_chunk = self._create_method_chunk(method, code, file_path, lang_name, class_def.name)
                chunks.append(method_chunk)
            
            return chunks
    
    def _create_single_class_chunk(
        self, 
        class_def: Definition, 
        code: bytes, 
        file_path: str,
        lang_name: str
    ) -> Tuple[str, ChunkMetadata]:
        """Create a single chunk for a small class."""
        chunk_text = code[class_def.start_byte:class_def.end_byte].decode('utf-8', errors='replace')
        
        domain = detect_domain(
            node_type=class_def.ast_node_type,  # Use actual AST node type
            name=class_def.name,
            file_path=file_path,
            annotations=class_def.annotations,
            modifiers=class_def.modifiers,
            language=lang_name
        )
        
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=class_def.start_line,
            end_line=class_def.end_line,
            token_count=len(chunk_text.split()),
            domain=domain,
            module=self._extract_module_name(file_path)
        )
        
        return (chunk_text, metadata)
    
    def _create_class_summary_chunk(
        self,
        class_def: Definition,
        methods: List[Definition],
        code: bytes,
        file_path: str,
        lang_config: LanguageConfig,
        lang_name: str
    ) -> Tuple[str, ChunkMetadata]:
        """Create a summary chunk for a large class."""
        summary = self._generate_class_summary(class_def, methods, code, lang_config)
        
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=class_def.start_line,
            end_line=class_def.end_line,
            token_count=len(summary.split()),
            domain='class_summary',
            module=self._extract_module_name(file_path)
        )
        
        return (summary, metadata)
    
    def _generate_class_summary(
        self,
        class_def: Definition,
        methods: List[Definition],
        code: bytes,
        lang_config: LanguageConfig
    ) -> str:
        """Generate a class summary with method signatures."""
        lines = []
        
        # Class declaration
        class_header = code[class_def.start_byte:class_def.start_byte + 200].decode('utf-8', errors='replace')
        first_line = class_header.split('\n')[0].strip()
        lines.append(first_line)
        
        # Docstring if present
        if class_def.docstring:
            lines.append(f'    """{class_def.docstring.strip()}"""')
        
        lines.append('')
        
        # Methods section
        if methods:
            lines.append('    Methods:')
            for method in methods:
                if method.signature:
                    lines.append(f'    - {method.signature}')
                else:
                    lines.append(f'    - {method.name}(...)')
        
        # TODO: Add inheritance and nested classes info
        
        return '\n'.join(lines)
    
    def _extract_class_methods(
        self,
        class_def: Definition,
        code: bytes,
        lang_config: LanguageConfig
    ) -> List[Definition]:
        """Extract method definitions from within a class."""
        # This is a simplified implementation
        # In a full implementation, we'd parse the class body and extract methods
        return []
    
    def _create_method_chunk(
        self,
        method_def: Definition,
        code: bytes,
        file_path: str,
        lang_name: str,
        parent_class: str
    ) -> Tuple[str, ChunkMetadata]:
        """Create a chunk for an individual method."""
        chunk_text = code[method_def.start_byte:method_def.end_byte].decode('utf-8', errors='replace')
        
        domain = detect_domain(
            node_type=method_def.ast_node_type,  # Use actual AST node type
            name=method_def.name,
            file_path=file_path,
            annotations=method_def.annotations,
            modifiers=method_def.modifiers,
            language=lang_name
        )
        
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=method_def.start_line,
            end_line=method_def.end_line,
            token_count=len(chunk_text.split()),
            domain=domain,
            module=self._extract_module_name(file_path),
            parent_class=parent_class
        )
        
        return (chunk_text, metadata)
    
    def _chunk_function(
        self,
        func_def: Definition,
        code: bytes,
        file_path: str,
        lang_config: LanguageConfig,
        lang_name: str
    ) -> Tuple[str, ChunkMetadata]:
        """Create a chunk for a standalone function."""
        chunk_text = code[func_def.start_byte:func_def.end_byte].decode('utf-8', errors='replace')
        
        domain = detect_domain(
            node_type=func_def.ast_node_type,  # Use actual AST node type
            name=func_def.name,
            file_path=file_path,
            annotations=func_def.annotations,
            modifiers=func_def.modifiers,
            language=lang_name
        )
        
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=func_def.start_line,
            end_line=func_def.end_line,
            token_count=len(chunk_text.split()),
            domain=domain,
            module=self._extract_module_name(file_path)
        )
        
        return (chunk_text, metadata)
    
    def _chunk_imports(
        self,
        imports: List[Definition],
        code: bytes,
        file_path: str
    ) -> Tuple[str, ChunkMetadata]:
        """Combine all imports into a single chunk."""
        if not imports:
            return None
        
        # Combine all import statements
        import_lines = []
        start_line = min(imp.start_line for imp in imports)
        end_line = max(imp.end_line for imp in imports)
        
        for imp in imports:
            import_text = code[imp.start_byte:imp.end_byte].decode('utf-8', errors='replace')
            import_lines.append(import_text.strip())
        
        chunk_text = '\n'.join(import_lines)
        
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            token_count=len(chunk_text.split()),
            domain='imports',
            module=self._extract_module_name(file_path)
        )
        
        return (chunk_text, metadata)
    
    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        path = Path(file_path)
        # Remove extension and convert path separators to dots
        module = str(path.with_suffix('')).replace('/', '.').replace('\\', '.')
        # Remove leading dots
        return module.lstrip('.')


# Export the main parser class
__all__ = ['TreeSitterParser']