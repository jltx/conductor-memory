"""
Chunking Strategy and Memory Refresh Mechanism for the Hybrid Local/Cloud LLM Orchestrator
"""

import ast
import logging
import warnings
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .heuristics import MethodImplementationDetail

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Enumeration of available chunking strategies"""
    FUNCTION_CLASS = "function_class"
    AST_PYTHON = "ast_python"  # Proper AST-based chunking for Python
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TOKEN_WINDOW = "token_window"
    JSONL_CONVERSATION = "jsonl_conversation"  # Claude Code conversation files

@dataclass
class ChunkMetadata:
    """Metadata for a memory chunk"""
    file_path: str
    start_line: int
    end_line: int
    commit_hash: Optional[str] = None
    token_count: int = 0
    domain: Optional[str] = None
    module: Optional[str] = None
    parent_class: Optional[str] = None  # For nested classes and methods in split classes
    # Implementation details for verification queries (Phase 1 enhancement)
    implementation_details: Optional[List['MethodImplementationDetail']] = field(default=None)
    
    def get_searchable_signals(self) -> str:
        """
        Return formatted implementation signals text for search indexing.
        
        Combines the searchable text from all implementation details,
        making method calls, attribute access, and subscripts searchable.
        
        Returns:
            Formatted string with all implementation signals, or empty string if none.
        """
        if not self.implementation_details:
            return ""
        
        signals_parts = []
        for detail in self.implementation_details:
            text = detail.to_searchable_text()
            if text and text != "[Implementation Signals]":
                signals_parts.append(text)
        
        return "\n\n".join(signals_parts)
    
    def get_signal_tags(self) -> List[str]:
        """
        Generate searchable tags from implementation signals.
        
        Creates tags in formats like:
        - calls:method_name (for internal and external calls)
        - subscript:iloc (for subscript access patterns)
        - reads:self._attr (for attribute reads)
        - writes:self._attr (for attribute writes)
        - param:name (for parameter usage)
        
        Returns:
            List of tag strings for filtering and search.
        """
        if not self.implementation_details:
            return []
        
        tags = []
        for detail in self.implementation_details:
            # Add call tags (both internal and external)
            for call in detail.internal_calls:
                tags.append(f"calls:{call}")
            for call in detail.external_calls:
                tags.append(f"calls:{call}")
            
            # Add attribute read tags
            for attr in detail.attribute_reads:
                tags.append(f"reads:{attr}")
            
            # Add attribute write tags
            for attr in detail.attribute_writes:
                tags.append(f"writes:{attr}")
            
            # Add subscript tags - extract the access pattern (e.g., "iloc" from "df.iloc[x]")
            for subscript in detail.subscript_access:
                # Extract the key part for the tag (e.g., "iloc" from "df.iloc[bar_index]")
                if '.' in subscript:
                    # Get the last attribute before the bracket
                    parts = subscript.split('[')[0].split('.')
                    if len(parts) >= 2:
                        tags.append(f"subscript:{parts[-1]}")
                    else:
                        tags.append(f"subscript:{subscript.split('[')[0]}")
                else:
                    tags.append(f"subscript:{subscript.split('[')[0]}")
            
            # Add parameter usage tags
            for param in detail.parameters_used:
                tags.append(f"param:{param}")
        
        # Deduplicate while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags

class ChunkingManager:
    """Manages chunking strategies for memory management"""
    
    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.FUNCTION_CLASS):
        self.strategy = strategy
        self._tree_sitter_parser = None  # Lazy initialization
        self._heuristic_extractor = None  # Lazy initialization for implementation signal extraction
    
    @property
    def tree_sitter_parser(self):
        """Get or create tree-sitter parser instance."""
        if self._tree_sitter_parser is None:
            try:
                from .parsers import TreeSitterParser
                self._tree_sitter_parser = TreeSitterParser()
            except ImportError as e:
                logger.warning(f"Tree-sitter not available: {e}")
                self._tree_sitter_parser = False  # Mark as unavailable
        return self._tree_sitter_parser if self._tree_sitter_parser is not False else None
    
    @property
    def heuristic_extractor(self):
        """Get or create heuristic extractor for implementation signal extraction."""
        if self._heuristic_extractor is None:
            try:
                from .heuristics import HeuristicExtractor
                self._heuristic_extractor = HeuristicExtractor()
            except ImportError as e:
                logger.warning(f"HeuristicExtractor not available: {e}")
                self._heuristic_extractor = False  # Mark as unavailable
        return self._heuristic_extractor if self._heuristic_extractor is not False else None
    
    def chunk_text(self, text: str, file_path: str, commit_hash: Optional[str] = None) -> List[Tuple[str, ChunkMetadata]]:
        """
        Split text into chunks based on the configured strategy.

        After chunking, enhances each chunk with implementation signals
        (method calls, attribute access, subscripts) for better searchability.

        Returns list of (chunk_text, metadata) tuples
        """
        # Check for conversation files first (JSONL format)
        if file_path.endswith('.jsonl'):
            try:
                from .parsers import ConversationParser
                parser = ConversationParser()
                return parser.parse(text, file_path, commit_hash)
            except Exception as e:
                logger.warning(f"Conversation parsing failed for {file_path}: {e}, falling back to token window")
                return self._chunk_by_token_window(text, file_path, commit_hash)

        # Try tree-sitter first for supported languages
        if self.tree_sitter_parser and self.tree_sitter_parser.supports(file_path):
            try:
                chunks = self.tree_sitter_parser.parse(text, file_path, commit_hash)
                # Enhance chunks with implementation signals
                return self._enhance_chunks_with_signals(chunks, text, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}, falling back to legacy chunking")
        
        # Fallback to existing chunking strategies
        is_python = file_path.endswith('.py')
        
        if self.strategy == ChunkingStrategy.AST_PYTHON and is_python:
            chunks = self._chunk_by_ast_python(text, file_path, commit_hash)
        elif self.strategy == ChunkingStrategy.FUNCTION_CLASS:
            # Use AST for Python files, fallback to naive for others
            if is_python:
                chunks = self._chunk_by_ast_python(text, file_path, commit_hash)
            else:
                chunks = self._chunk_by_function_class(text, file_path, commit_hash)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(text, file_path, commit_hash)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(text, file_path, commit_hash)
        elif self.strategy == ChunkingStrategy.TOKEN_WINDOW:
            chunks = self._chunk_by_token_window(text, file_path, commit_hash)
        else:
            # Default: use AST for Python, naive function/class for others
            if is_python:
                chunks = self._chunk_by_ast_python(text, file_path, commit_hash)
            else:
                chunks = self._chunk_by_function_class(text, file_path, commit_hash)
        
        # Enhance chunks with implementation signals
        return self._enhance_chunks_with_signals(chunks, text, file_path)
    
    def _enhance_chunks_with_signals(
        self, 
        chunks: List[Tuple[str, ChunkMetadata]], 
        file_content: str, 
        file_path: str
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Enhance all chunks in a file with implementation signals.
        
        Extracts file metadata once and distributes signals to matching chunks.
        This is more efficient than extracting per-chunk.
        
        Args:
            chunks: List of (chunk_text, metadata) tuples
            file_content: Full file content
            file_path: Path to the file
            
        Returns:
            List of enhanced (chunk_text, metadata) tuples
        """
        if not self.heuristic_extractor or not chunks:
            return chunks
        
        try:
            # Extract file metadata once for all chunks
            file_metadata = self.heuristic_extractor.extract_file_metadata(file_path, file_content)
            
            if not file_metadata or not file_metadata.method_details:
                return chunks
            
            # Enhance each chunk
            enhanced_chunks = []
            for chunk_text, metadata in chunks:
                enhanced_text = self._match_and_enhance_chunk(
                    chunk_text, metadata, file_metadata.method_details
                )
                enhanced_chunks.append((enhanced_text, metadata))
            
            return enhanced_chunks
            
        except Exception as e:
            logger.debug(f"Batch implementation signal extraction failed for {file_path}: {e}")
            return chunks
    
    def _match_and_enhance_chunk(
        self,
        chunk_text: str,
        metadata: ChunkMetadata,
        all_method_details: List['MethodImplementationDetail']
    ) -> str:
        """
        Match method details to a specific chunk and enhance its content.
        
        Args:
            chunk_text: Original chunk content
            metadata: ChunkMetadata for this chunk (will be modified)
            all_method_details: All extracted method details from the file
            
        Returns:
            Enhanced chunk content with appended signals
        """
        from .heuristics import MethodImplementationDetail
        
        # Filter method details to those within this chunk
        chunk_details: List[MethodImplementationDetail] = []
        
        for detail in all_method_details:
            # Check if the method definition appears in this chunk
            # Look for common definition patterns
            def_patterns = [
                f"def {detail.name}(",       # Python function
                f"def {detail.name} (",      # Python with space
                f"async def {detail.name}(", # Python async function
                f"async def {detail.name} (",
                f"func {detail.name}(",      # Go function
                f"func ({detail.name}",      # Go method receiver (partial match)
                f"function {detail.name}(",  # JavaScript/TypeScript function
                f"fun {detail.name}(",       # Kotlin function
            ]
            
            if any(pattern in chunk_text for pattern in def_patterns):
                chunk_details.append(detail)
        
        if not chunk_details:
            return chunk_text
        
        # Store implementation details in metadata
        metadata.implementation_details = chunk_details
        
        # Get formatted signals text
        signals_text = metadata.get_searchable_signals()
        
        if not signals_text:
            return chunk_text
        
        # Append signals to chunk content
        return f"{chunk_text}\n\n{signals_text}"
    
    def _chunk_by_function_class(self, text: str, file_path: str, commit_hash: Optional[str]) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk by functions and classes - naive implementation for non-Python files"""
        # This is a simplified implementation for non-Python files
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_start_line = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith(('def ', 'class ', 'function ', 'export ')) and current_chunk:
                # End of previous chunk
                chunk_text = '\n'.join(current_chunk)
                metadata = ChunkMetadata(
                    file_path=file_path,
                    start_line=current_start_line + 1,
                    end_line=i,
                    commit_hash=commit_hash,
                    token_count=len(chunk_text.split()),
                )
                chunks.append((chunk_text, metadata))
                current_chunk = [line]
                current_start_line = i
            else:
                current_chunk.append(line)
        
        # Add the final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=current_start_line + 1,
                end_line=len(lines),
                commit_hash=commit_hash,
                token_count=len(chunk_text.split()),
            )
            chunks.append((chunk_text, metadata))
        
        return chunks
    
    def _chunk_by_ast_python(self, text: str, file_path: str, commit_hash: Optional[str]) -> List[Tuple[str, ChunkMetadata]]:
        """
        AST-based chunking for Python files.
        
        Creates separate chunks for:
        - Module-level docstrings and imports (header chunk)
        - Each top-level function (with decorators)
        - Each top-level class (with all methods and docstring)
        - Module-level constants and assignments
        
        This provides better semantic boundaries than naive line-based splitting.
        """
        # Strip BOM (Byte Order Mark) if present - common in files saved by some editors
        if text.startswith('\ufeff'):
            text = text[1:]
        
        try:
            # Suppress SyntaxWarnings from invalid escape sequences in source files
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
        except SyntaxError as e:
            logger.warning(f"AST parse failed for {file_path}: {e}. Falling back to naive chunking.")
            return self._chunk_by_function_class(text, file_path, commit_hash)
        
        lines = text.split('\n')
        chunks = []
        
        # Extract module name from file path
        module_name = file_path.replace('/', '.').replace('\\', '.').rstrip('.py')
        if module_name.startswith('.'):
            module_name = module_name[1:]
        
        # Collect all top-level definitions with their line ranges
        definitions = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Top-level function
                start_line = self._get_decorator_start(node)
                end_line = node.end_lineno or node.lineno
                
                # Get function signature for metadata
                func_name = node.name
                args = self._get_function_args(node)
                
                definitions.append({
                    'type': 'function',
                    'name': func_name,
                    'start': start_line,
                    'end': end_line,
                    'args': args
                })
                
            elif isinstance(node, ast.ClassDef):
                # Top-level class with all its methods
                start_line = self._get_decorator_start(node)
                end_line = node.end_lineno or node.lineno
                
                # Get method names for metadata
                method_names = [
                    n.name for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node
                ]
                
                definitions.append({
                    'type': 'class',
                    'name': node.name,
                    'start': start_line,
                    'end': end_line,
                    'methods': method_names
                })
        
        # Sort by start line
        definitions.sort(key=lambda d: d['start'])
        
        # Build chunks
        if not definitions:
            # No functions/classes - treat entire file as one chunk
            chunk_text = text.strip()
            if chunk_text:
                metadata = ChunkMetadata(
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    commit_hash=commit_hash,
                    token_count=len(chunk_text.split()),
                    module=module_name
                )
                chunks.append((chunk_text, metadata))
            return chunks
        
        # Create header chunk (imports, module docstring, etc.) if content before first definition
        first_def_start = definitions[0]['start']
        if first_def_start > 1:
            header_lines = lines[:first_def_start - 1]
            header_text = '\n'.join(header_lines).strip()
            if header_text and len(header_text) > 30:  # Only if substantial
                metadata = ChunkMetadata(
                    file_path=file_path,
                    start_line=1,
                    end_line=first_def_start - 1,
                    commit_hash=commit_hash,
                    token_count=len(header_text.split()),
                    module=module_name,
                    domain='imports'
                )
                chunks.append((header_text, metadata))
        
        # Create chunk for each definition
        for defn in definitions:
            # Extract the code for this definition (1-indexed to 0-indexed)
            chunk_lines = lines[defn['start'] - 1:defn['end']]
            chunk_text = '\n'.join(chunk_lines).strip()
            
            if not chunk_text or len(chunk_text) < 20:
                continue
            
            # Determine domain based on definition type and name
            domain = None
            if defn['type'] == 'class':
                domain = 'class'
            elif defn['type'] == 'function':
                # Categorize by common patterns
                name = defn['name'].lower()
                if name.startswith('test_') or name.startswith('_test'):
                    domain = 'test'
                elif name.startswith('_'):
                    domain = 'private'
                elif any(name.startswith(p) for p in ['get_', 'set_', 'is_', 'has_']):
                    domain = 'accessor'
                else:
                    domain = 'function'
            
            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=defn['start'],
                end_line=defn['end'],
                commit_hash=commit_hash,
                token_count=len(chunk_text.split()),
                module=module_name,
                domain=domain
            )
            chunks.append((chunk_text, metadata))
        
        return chunks
    
    def _get_decorator_start(self, node: ast.AST) -> int:
        """Get the start line including decorators"""
        if hasattr(node, 'decorator_list') and node.decorator_list:
            return node.decorator_list[0].lineno
        return node.lineno
    
    def _get_function_args(self, node: ast.FunctionDef) -> List[str]:
        """Extract function argument names"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return args
    
    def _chunk_by_paragraph(self, text: str, file_path: str, commit_hash: Optional[str]) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk by paragraphs"""
        # Split by empty lines to identify paragraphs
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=(i * 2) + 1,  # Approximate line numbers
                end_line=((i * 2) + 1) + len(paragraph.split('\n')),
                commit_hash=commit_hash,
                token_count=len(paragraph.split()),
            )
            chunks.append((paragraph, metadata))
        
        return chunks
    
    def _chunk_by_sentence(self, text: str, file_path: str, commit_hash: Optional[str]) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk by sentences"""
        # Simple sentence splitting (may need improvement for production)
        import re
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        start_line = 1
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                sentence = sentence.strip() + '.'
                metadata = ChunkMetadata(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line + len(sentence.split('\n')) - 1,
                    commit_hash=commit_hash,
                    token_count=len(sentence.split()),
                )
                chunks.append((sentence, metadata))
                start_line += len(sentence.split('\n'))
        
        return chunks
    
    def _chunk_by_token_window(self, text: str, file_path: str, commit_hash: Optional[str]) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk by token windows (recommended for embedding size control)"""
        # For demonstration, we'll use a fixed window size
        # In practice, this would integrate with actual tokenization
        tokens = text.split()
        window_size = 512  # 512 tokens as per project spec
        chunks = []
        
        start_line = 1
        for i in range(0, len(tokens), window_size):
            window_tokens = tokens[i:i + window_size]
            chunk_text = ' '.join(window_tokens)
            
            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=start_line,
                end_line=start_line + len(chunk_text.split('\n')) - 1,
                commit_hash=commit_hash,
                token_count=len(window_tokens),
            )
            chunks.append((chunk_text, metadata))
            start_line += len(chunk_text.split('\n'))
        
        return chunks

class MemoryRefreshMechanism:
    """Handles memory refresh and updating triggers"""
    
    def __init__(self):
        self.triggers = {
            'file_change': self._on_file_change,
            'commit_push': self._on_commit_push,
            'llm_pruning': self._on_llm_pruning,
        }
    
    def _on_file_change(self, file_path: str, old_content: str, new_content: str) -> None:
        """
        Triggered when a file changes
        Should update related memory chunks and embeddings
        """
        # In practice, this would update the corresponding memory chunks
        print(f"File change detected: {file_path}")
    
    def _on_commit_push(self, commit_hash: str, files_changed: List[str]) -> None:
        """
        Triggered when a commit is pushed
        Should update memory with commit context
        """
        # In practice, this would update memory chunks with commit metadata
        print(f"Commit push detected: {commit_hash}")
    
    def _on_llm_pruning(self) -> None:
        """
        Triggered when LLM pruning occurs
        Should refresh relevance scores and memory state
        """
        # In practice, this would refresh memory relevance scores
        print("LLM pruning triggered memory refresh")
    
    def trigger_refresh(self, trigger_type: str, **kwargs) -> None:
        """Trigger a memory refresh based on the specified type"""
        if trigger_type in self.triggers:
            self.triggers[trigger_type](**kwargs)
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")