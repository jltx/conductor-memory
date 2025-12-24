"""
Method call graph construction for tracking method-to-method relationships.

Enables "what calls X?" and "what does X call?" queries by tracking
caller-callee relationships extracted from AST analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import networkx as nx

if TYPE_CHECKING:
    from .heuristics import HeuristicMetadata, MethodImplementationDetail

logger = logging.getLogger(__name__)


@dataclass
class MethodNode:
    """
    Represents a method or function node in the call graph.
    
    Attributes:
        qualified_name: Fully qualified name, e.g., "MyClass._generate_features" 
                        or "module.function" for standalone functions
        file_path: Path to the source file containing this method
        class_name: Name of the containing class, or None for standalone functions
        method_name: Just the method/function name without class prefix
        line_number: Line number where the method is defined
    """
    qualified_name: str
    file_path: str
    class_name: Optional[str]
    method_name: str
    line_number: int
    
    def __hash__(self):
        return hash(self.qualified_name)
    
    def __eq__(self, other):
        if not isinstance(other, MethodNode):
            return False
        return self.qualified_name == other.qualified_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the node to a dictionary for storage/transmission."""
        return {
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "line_number": self.line_number,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodNode":
        """Deserialize a node from a dictionary."""
        return cls(
            qualified_name=data["qualified_name"],
            file_path=data["file_path"],
            class_name=data.get("class_name"),
            method_name=data["method_name"],
            line_number=data["line_number"],
        )


class MethodCallGraph:
    """
    Tracks method-to-method call relationships for a codebase.
    
    Uses NetworkX internally for efficient graph operations, similar to ImportGraph.
    Supports queries like "what calls this method?" and "what does this method call?".
    
    Attributes:
        nodes: Dictionary mapping qualified_name to MethodNode
        edges: Dictionary mapping caller qualified_name to list of callee qualified_names
        reverse_edges: Dictionary mapping callee qualified_name to list of caller qualified_names
    """
    
    def __init__(self):
        self.nodes: Dict[str, MethodNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.reverse_edges: Dict[str, List[str]] = {}
        self._graph: Optional[nx.DiGraph] = None
    
    def add_node(self, node: MethodNode) -> None:
        """
        Add a method node to the graph.
        
        Args:
            node: The MethodNode to add
        """
        if node.qualified_name not in self.nodes:
            self.nodes[node.qualified_name] = node
            self.edges[node.qualified_name] = []
            self.reverse_edges[node.qualified_name] = []
            # Invalidate cached graph
            self._graph = None
            logger.debug(f"Added method node: {node.qualified_name}")
    
    def add_edge(self, caller: str, callee: str, allow_self_loops: bool = False) -> bool:
        """
        Add a call relationship edge from caller to callee.
        
        Both caller and callee should already exist as nodes. If they don't,
        the edge is still recorded for flexibility (callee might be external).
        
        Args:
            caller: Qualified name of the calling method
            callee: Qualified name of the called method
            allow_self_loops: If False (default), skip edges where caller == callee
            
        Returns:
            True if edge was added, False if skipped (self-loop or duplicate)
        """
        # Skip self-loops unless explicitly allowed (prevents trivial cycles)
        if caller == callee and not allow_self_loops:
            logger.debug(f"Skipping self-loop: {caller}")
            return False
        
        # Initialize edges lists if not present
        if caller not in self.edges:
            self.edges[caller] = []
        if callee not in self.reverse_edges:
            self.reverse_edges[callee] = []
        
        # Avoid duplicate edges
        if callee not in self.edges[caller]:
            self.edges[caller].append(callee)
            logger.debug(f"Added edge: {caller} -> {callee}")
        else:
            return False  # Duplicate edge
        
        if caller not in self.reverse_edges[callee]:
            self.reverse_edges[callee].append(caller)
        
        # Invalidate cached graph
        self._graph = None
        return True
    
    def get_node(self, qualified_name: str) -> Optional[MethodNode]:
        """
        Get a method node by its qualified name.
        
        Args:
            qualified_name: The fully qualified method name
            
        Returns:
            The MethodNode if found, None otherwise
        """
        return self.nodes.get(qualified_name)
    
    def get_callers(self, qualified_name: str) -> List[MethodNode]:
        """
        Get all methods that call the specified method.
        
        Args:
            qualified_name: The qualified name of the method to find callers for
            
        Returns:
            List of MethodNodes that call this method
        """
        caller_names = self.reverse_edges.get(qualified_name, [])
        return [self.nodes[name] for name in caller_names if name in self.nodes]
    
    def get_callees(self, qualified_name: str) -> List[MethodNode]:
        """
        Get all methods called by the specified method.
        
        Args:
            qualified_name: The qualified name of the calling method
            
        Returns:
            List of MethodNodes that are called by this method
        """
        callee_names = self.edges.get(qualified_name, [])
        return [self.nodes[name] for name in callee_names if name in self.nodes]
    
    def get_caller_names(self, qualified_name: str) -> List[str]:
        """
        Get qualified names of all methods that call the specified method.
        
        Args:
            qualified_name: The qualified name of the method to find callers for
            
        Returns:
            List of qualified names (includes external methods not in nodes)
        """
        return self.reverse_edges.get(qualified_name, []).copy()
    
    def get_callee_names(self, qualified_name: str) -> List[str]:
        """
        Get qualified names of all methods called by the specified method.
        
        Args:
            qualified_name: The qualified name of the calling method
            
        Returns:
            List of qualified names (includes external methods not in nodes)
        """
        return self.edges.get(qualified_name, []).copy()
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from the edges."""
        if self._graph is not None:
            return self._graph
        
        self._graph = nx.DiGraph()
        
        # Add all nodes
        for qualified_name, node in self.nodes.items():
            self._graph.add_node(qualified_name, **node.to_dict())
        
        # Add all edges
        for caller, callees in self.edges.items():
            for callee in callees:
                self._graph.add_edge(caller, callee)
        
        logger.debug(f"Built NetworkX graph: {len(self._graph.nodes)} nodes, {len(self._graph.edges)} edges")
        return self._graph
    
    def find_call_path(self, from_method: str, to_method: str) -> Optional[List[str]]:
        """
        Find a call path between two methods if one exists.
        
        Args:
            from_method: Qualified name of the starting method
            to_method: Qualified name of the target method
            
        Returns:
            List of qualified names representing the call path, or None if no path exists
        """
        graph = self._build_networkx_graph()
        
        try:
            if from_method in graph and to_method in graph:
                path = nx.shortest_path(graph, from_method, to_method)
                return path
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            pass
        
        return None
    
    def find_call_chain(
        self, 
        from_method: str, 
        to_method: str, 
        max_depth: int = 10
    ) -> Optional[List[str]]:
        """
        Find a call chain between two methods with depth limit.
        
        Similar to find_call_path but with configurable depth limit to prevent
        traversing extremely long paths in large codebases.
        
        Args:
            from_method: Qualified name of the starting method
            to_method: Qualified name of the target method
            max_depth: Maximum path length to search (default 10)
            
        Returns:
            List of qualified names representing the call chain, or None if no path
            exists within the depth limit
        """
        graph = self._build_networkx_graph()
        
        if from_method not in graph or to_method not in graph:
            return None
        
        try:
            # Use BFS with depth limit via cutoff parameter
            # shortest_path with cutoff isn't available, so use all_simple_paths
            # with a generator that stops at first result
            for path in nx.all_simple_paths(graph, from_method, to_method, cutoff=max_depth):
                return list(path)  # Return first path found
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            pass
        
        return None
    
    def get_transitive_callers(
        self, 
        qualified_name: str, 
        max_depth: int = 5
    ) -> List[MethodNode]:
        """
        Get all methods that directly or transitively call the specified method.
        
        Uses BFS traversal which naturally handles cycles by not revisiting nodes.
        
        Args:
            qualified_name: The qualified name of the method to find callers for
            max_depth: Maximum depth of caller chain to traverse (default 5)
            
        Returns:
            List of MethodNodes representing all transitive callers
        """
        caller_names = self.get_reachable_methods(
            qualified_name, 
            direction="callers", 
            max_depth=max_depth
        )
        return [self.nodes[name] for name in caller_names if name in self.nodes]
    
    def get_transitive_callees(
        self, 
        qualified_name: str, 
        max_depth: int = 5
    ) -> List[MethodNode]:
        """
        Get all methods directly or transitively called by the specified method.
        
        Uses BFS traversal which naturally handles cycles by not revisiting nodes.
        
        Args:
            qualified_name: The qualified name of the calling method
            max_depth: Maximum depth of callee chain to traverse (default 5)
            
        Returns:
            List of MethodNodes representing all transitive callees
        """
        callee_names = self.get_reachable_methods(
            qualified_name, 
            direction="callees", 
            max_depth=max_depth
        )
        return [self.nodes[name] for name in callee_names if name in self.nodes]
    
    def find_methods_by_name(
        self, 
        name_pattern: str, 
        case_sensitive: bool = False
    ) -> List[MethodNode]:
        """
        Find methods matching a name pattern (substring or simple wildcard).
        
        Supports:
        - Exact match: "process_data"
        - Substring match: "process" matches "process_data", "data_processor"
        - Wildcard prefix: "*_handler" matches "error_handler", "request_handler"
        - Wildcard suffix: "get_*" matches "get_user", "get_data"
        - Wildcard both: "*cache*" matches "build_cache", "cache_manager", "get_cached_data"
        
        Args:
            name_pattern: Pattern to match against method names
            case_sensitive: If False (default), match case-insensitively
            
        Returns:
            List of MethodNodes with names matching the pattern
        """
        import fnmatch
        
        # Normalize pattern for case-insensitive matching
        pattern = name_pattern if case_sensitive else name_pattern.lower()
        
        # Determine matching strategy
        has_wildcard = '*' in pattern or '?' in pattern
        
        matches: List[MethodNode] = []
        
        for node in self.nodes.values():
            method_name = node.method_name if case_sensitive else node.method_name.lower()
            qualified_name = node.qualified_name if case_sensitive else node.qualified_name.lower()
            
            matched = False
            
            if has_wildcard:
                # Use fnmatch for wildcard patterns
                if fnmatch.fnmatch(method_name, pattern):
                    matched = True
                elif fnmatch.fnmatch(qualified_name, pattern):
                    matched = True
            else:
                # Substring match
                if pattern in method_name:
                    matched = True
                elif pattern in qualified_name:
                    matched = True
            
            if matched:
                matches.append(node)
        
        # Sort by qualified name for consistent ordering
        matches.sort(key=lambda n: n.qualified_name)
        return matches
    
    def get_call_depth(self, qualified_name: str, direction: str = "callees") -> int:
        """
        Get the maximum call depth from a method.
        
        Args:
            qualified_name: The method to start from
            direction: "callees" for downstream depth, "callers" for upstream depth
            
        Returns:
            Maximum depth of calls from this method
        """
        graph = self._build_networkx_graph()
        
        if qualified_name not in graph:
            return 0
        
        if direction == "callers":
            graph = graph.reverse()
        
        try:
            # Use BFS to find all reachable nodes and their distances
            lengths = nx.single_source_shortest_path_length(graph, qualified_name)
            return max(lengths.values()) if lengths else 0
        except Exception as e:
            logger.debug(f"Failed to calculate call depth: {e}")
            return 0
    
    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the call graph.
        
        Useful for detecting recursive call patterns and potential infinite loops.
        Uses NetworkX's simple_cycles which handles the graph safely.
        
        Returns:
            List of cycles, where each cycle is a list of qualified method names
        """
        graph = self._build_networkx_graph()
        
        try:
            # simple_cycles finds all elementary cycles in a directed graph
            # It's safe and won't infinite loop even with complex cycle structures
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                logger.debug(f"Found {len(cycles)} cycles in call graph")
            return cycles
        except Exception as e:
            logger.warning(f"Failed to find cycles: {e}")
            return []
    
    def has_cycles(self) -> bool:
        """
        Check if the call graph contains any cycles.
        
        More efficient than find_cycles() when you only need to know if cycles exist.
        
        Returns:
            True if the graph contains at least one cycle
        """
        graph = self._build_networkx_graph()
        
        try:
            # is_directed_acyclic_graph is O(V + E) and stops early
            return not nx.is_directed_acyclic_graph(graph)
        except Exception as e:
            logger.warning(f"Failed to check for cycles: {e}")
            return False
    
    def get_reachable_methods(
        self, 
        qualified_name: str, 
        direction: str = "callees",
        max_depth: Optional[int] = None
    ) -> List[str]:
        """
        Get all methods reachable from a starting method, with cycle-safe traversal.
        
        Uses BFS which naturally handles cycles by not revisiting nodes.
        
        Args:
            qualified_name: The method to start from
            direction: "callees" for downstream, "callers" for upstream
            max_depth: Maximum traversal depth (None for unlimited)
            
        Returns:
            List of reachable method qualified names (excluding the starting method)
        """
        graph = self._build_networkx_graph()
        
        if qualified_name not in graph:
            return []
        
        if direction == "callers":
            graph = graph.reverse()
        
        try:
            if max_depth is not None:
                # BFS with depth limit
                lengths = nx.single_source_shortest_path_length(
                    graph, qualified_name, cutoff=max_depth
                )
            else:
                lengths = nx.single_source_shortest_path_length(graph, qualified_name)
            
            # Exclude the starting node itself
            return [node for node in lengths.keys() if node != qualified_name]
        except Exception as e:
            logger.debug(f"Failed to get reachable methods: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        graph = self._build_networkx_graph()
        
        total_edges = sum(len(callees) for callees in self.edges.values())
        
        return {
            "total_methods": len(self.nodes),
            "total_edges": total_edges,
            "methods_with_callers": sum(1 for v in self.reverse_edges.values() if v),
            "methods_with_callees": sum(1 for v in self.edges.values() if v),
            "isolated_methods": sum(
                1 for name in self.nodes 
                if not self.edges.get(name) and not self.reverse_edges.get(name)
            ),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire graph to a dictionary."""
        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": {caller: list(callees) for caller, callees in self.edges.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodCallGraph":
        """Deserialize a graph from a dictionary."""
        graph = cls()
        
        # Reconstruct nodes
        for name, node_data in data.get("nodes", {}).items():
            graph.nodes[name] = MethodNode.from_dict(node_data)
            graph.edges[name] = []
            graph.reverse_edges[name] = []
        
        # Reconstruct edges
        for caller, callees in data.get("edges", {}).items():
            for callee in callees:
                graph.add_edge(caller, callee)
        
        return graph
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.reverse_edges.clear()
        self._graph = None
        logger.debug("Cleared method call graph")


class CallGraphBuilder:
    """
    Builds a MethodCallGraph from heuristic metadata.
    
    Takes HeuristicMetadata objects (containing method_details) and constructs
    a graph of method-to-method call relationships.
    
    Usage:
        builder = CallGraphBuilder()
        builder.build_from_heuristics("src/service.py", heuristic_metadata)
        builder.build_from_heuristics("src/utils.py", another_metadata)
        graph = builder.get_graph()
        
    Or for bulk operations:
        graph = builder.build_from_multiple_files([
            ("src/service.py", service_metadata),
            ("src/utils.py", utils_metadata),
        ])
    """
    
    def __init__(self, graph: Optional[MethodCallGraph] = None):
        """
        Initialize the builder.
        
        Args:
            graph: Optional existing graph to add to. If None, creates a new graph.
        """
        self._graph = graph or MethodCallGraph()
        
        # Mapping from simple name to qualified names for call resolution
        # e.g., {"_generate_features": ["MyClass._generate_features", "OtherClass._generate_features"]}
        self._name_to_qualified: Dict[str, List[str]] = {}
        
        # Track which files have been processed
        self._processed_files: set = set()
    
    def get_graph(self) -> MethodCallGraph:
        """Get the constructed graph."""
        return self._graph
    
    def build_from_heuristics(
        self, 
        file_path: str, 
        heuristic_metadata: 'HeuristicMetadata'
    ) -> None:
        """
        Build graph nodes and edges from a file's heuristic metadata.
        
        Extracts methods from heuristic_metadata.method_details and creates:
        - A MethodNode for each method
        - Edges for each method's internal_calls and external_calls
        
        Args:
            file_path: Path to the source file
            heuristic_metadata: HeuristicMetadata containing method_details
        """
        from .heuristics import HeuristicMetadata, MethodImplementationDetail
        
        if file_path in self._processed_files:
            logger.debug(f"File already processed: {file_path}")
            return
        
        self._processed_files.add(file_path)
        
        if not heuristic_metadata.method_details:
            logger.debug(f"No method details in {file_path}")
            return
        
        # Build a map of method names to their containing classes for this file
        # This helps determine qualified names
        method_to_class = self._build_method_class_map(heuristic_metadata)
        
        # First pass: Create nodes for all methods
        for detail in heuristic_metadata.method_details:
            node = self._create_method_node(
                detail=detail,
                file_path=file_path,
                method_to_class=method_to_class
            )
            
            if node:
                self._graph.add_node(node)
                
                # Track name mapping for call resolution
                if node.method_name not in self._name_to_qualified:
                    self._name_to_qualified[node.method_name] = []
                self._name_to_qualified[node.method_name].append(node.qualified_name)
        
        # Second pass: Create edges for all calls
        for detail in heuristic_metadata.method_details:
            caller_qname = self._get_qualified_name(
                detail.name, method_to_class, file_path
            )
            
            if not caller_qname:
                continue
            
            # Process internal calls (self.method(), self.attr.method())
            for call in detail.internal_calls:
                callee_name = self._normalize_internal_call(call)
                callee_qname = self._resolve_call(
                    callee_name, 
                    caller_qname, 
                    method_to_class, 
                    file_path,
                    is_internal=True
                )
                
                if callee_qname:
                    self._graph.add_edge(caller_qname, callee_qname)
            
            # Process external calls (module.func(), Class.method())
            for call in detail.external_calls:
                callee_qname = self._resolve_call(
                    call,
                    caller_qname,
                    method_to_class,
                    file_path,
                    is_internal=False
                )
                
                if callee_qname:
                    self._graph.add_edge(caller_qname, callee_qname)
        
        logger.debug(
            f"Processed {file_path}: {len(heuristic_metadata.method_details)} methods"
        )
    
    def build_from_multiple_files(
        self, 
        files: List[Tuple[str, 'HeuristicMetadata']]
    ) -> MethodCallGraph:
        """
        Build a complete graph from multiple files at once.
        
        This is more efficient than calling build_from_heuristics repeatedly
        as it can do better cross-file call resolution.
        
        Args:
            files: List of (file_path, HeuristicMetadata) tuples
            
        Returns:
            The constructed MethodCallGraph
        """
        from .heuristics import HeuristicMetadata
        
        # First pass: collect all method nodes from all files
        # This enables better cross-file resolution
        for file_path, metadata in files:
            if file_path in self._processed_files:
                continue
            
            method_to_class = self._build_method_class_map(metadata)
            
            for detail in metadata.method_details:
                node = self._create_method_node(
                    detail=detail,
                    file_path=file_path,
                    method_to_class=method_to_class
                )
                
                if node:
                    self._graph.add_node(node)
                    
                    if node.method_name not in self._name_to_qualified:
                        self._name_to_qualified[node.method_name] = []
                    self._name_to_qualified[node.method_name].append(node.qualified_name)
        
        # Second pass: create edges now that we know all methods
        for file_path, metadata in files:
            if not metadata.method_details:
                continue
            
            self._processed_files.add(file_path)
            method_to_class = self._build_method_class_map(metadata)
            
            for detail in metadata.method_details:
                caller_qname = self._get_qualified_name(
                    detail.name, method_to_class, file_path
                )
                
                if not caller_qname:
                    continue
                
                # Process internal calls
                for call in detail.internal_calls:
                    callee_name = self._normalize_internal_call(call)
                    callee_qname = self._resolve_call(
                        callee_name,
                        caller_qname,
                        method_to_class,
                        file_path,
                        is_internal=True
                    )
                    
                    if callee_qname:
                        self._graph.add_edge(caller_qname, callee_qname)
                
                # Process external calls
                for call in detail.external_calls:
                    callee_qname = self._resolve_call(
                        call,
                        caller_qname,
                        method_to_class,
                        file_path,
                        is_internal=False
                    )
                    
                    if callee_qname:
                        self._graph.add_edge(caller_qname, callee_qname)
        
        logger.info(
            f"Built call graph from {len(files)} files: "
            f"{len(self._graph.nodes)} methods, "
            f"{sum(len(e) for e in self._graph.edges.values())} edges"
        )
        
        return self._graph
    
    def _build_method_class_map(
        self, 
        metadata: 'HeuristicMetadata'
    ) -> Dict[str, Optional[str]]:
        """
        Build a map from method names to their containing class names.
        
        Uses the classes and methods lists in HeuristicMetadata to determine
        which methods belong to which classes based on line number ranges.
        
        Args:
            metadata: HeuristicMetadata with classes and methods info
            
        Returns:
            Dict mapping method name to class name (or None for standalone functions)
        """
        method_to_class: Dict[str, Optional[str]] = {}
        
        # Build class ranges from metadata
        class_ranges: List[Tuple[str, int, int]] = []
        for cls in metadata.classes:
            class_ranges.append((
                cls['name'],
                cls.get('start_line', 0),
                cls.get('end_line', float('inf'))
            ))
        
        # Map methods to their containing classes
        for detail in metadata.method_details:
            # Try to find the method's line number from method_details
            # The signature often contains enough info, but we can also
            # check the methods/functions lists
            method_line = self._find_method_line(detail.name, metadata)
            
            containing_class = None
            if method_line:
                for class_name, start, end in class_ranges:
                    if start <= method_line <= end:
                        containing_class = class_name
                        break
            
            method_to_class[detail.name] = containing_class
        
        return method_to_class
    
    def _find_method_line(
        self, 
        method_name: str, 
        metadata: 'HeuristicMetadata'
    ) -> Optional[int]:
        """Find the line number where a method is defined."""
        # Check in methods list first
        for method in metadata.methods:
            if method.get('name') == method_name:
                return method.get('start_line')
        
        # Check in functions list
        for func in metadata.functions:
            if func.get('name') == method_name:
                return func.get('start_line')
        
        return None
    
    def _create_method_node(
        self,
        detail: 'MethodImplementationDetail',
        file_path: str,
        method_to_class: Dict[str, Optional[str]]
    ) -> Optional[MethodNode]:
        """
        Create a MethodNode from a MethodImplementationDetail.
        
        Args:
            detail: The method implementation detail
            file_path: Path to the source file
            method_to_class: Mapping from method name to class name
            
        Returns:
            MethodNode or None if creation fails
        """
        from .heuristics import MethodImplementationDetail
        
        class_name = method_to_class.get(detail.name)
        
        # Build qualified name
        if class_name:
            qualified_name = f"{class_name}.{detail.name}"
        else:
            # Standalone function - use just the name
            qualified_name = detail.name
        
        # Extract line number from signature or estimate from line_count
        line_number = self._extract_line_from_signature(detail.signature)
        if line_number is None:
            line_number = 1  # Default
        
        return MethodNode(
            qualified_name=qualified_name,
            file_path=file_path,
            class_name=class_name,
            method_name=detail.name,
            line_number=line_number
        )
    
    def _extract_line_from_signature(self, signature: str) -> Optional[int]:
        """Extract line number if embedded in signature (usually not)."""
        # Signatures don't typically contain line numbers
        # This is a placeholder for future enhancement
        return None
    
    def _get_qualified_name(
        self,
        method_name: str,
        method_to_class: Dict[str, Optional[str]],
        file_path: str
    ) -> Optional[str]:
        """Get the qualified name for a method."""
        class_name = method_to_class.get(method_name)
        
        if class_name:
            return f"{class_name}.{method_name}"
        else:
            return method_name
    
    def _normalize_internal_call(self, call: str) -> str:
        """
        Normalize an internal call to just the method name.
        
        Internal calls come in forms like:
        - "method_name" (already normalized)
        - "attr.method_name" (self.attr.method())
        
        We want just the method name part.
        
        Args:
            call: The internal call string
            
        Returns:
            The normalized method name
        """
        # If it's a chained call like "attr.method", take the last part
        if '.' in call:
            parts = call.split('.')
            return parts[-1]
        return call
    
    def _resolve_call(
        self,
        call: str,
        caller_qname: str,
        method_to_class: Dict[str, Optional[str]],
        file_path: str,
        is_internal: bool
    ) -> Optional[str]:
        """
        Resolve a call to a qualified method name.
        
        For internal calls (self.method()), we look for methods in the same class.
        For external calls, we try to find a matching method in the graph.
        
        Args:
            call: The call string (e.g., "method_name" or "Class.method")
            caller_qname: Qualified name of the calling method
            method_to_class: Method-to-class mapping for the current file
            file_path: Current file path
            is_internal: True if this is an internal (self.) call
            
        Returns:
            Qualified name of the callee, or None if unresolvable
        """
        if is_internal:
            # Internal call - look for method in same class
            caller_class = None
            if '.' in caller_qname:
                caller_class = caller_qname.rsplit('.', 1)[0]
            
            if caller_class:
                # Try same-class method first
                same_class_qname = f"{caller_class}.{call}"
                if same_class_qname in self._graph.nodes:
                    return same_class_qname
                
                # Check if we know about this method at all
                if call in self._name_to_qualified:
                    candidates = self._name_to_qualified[call]
                    # Prefer same-class match
                    for candidate in candidates:
                        if candidate.startswith(f"{caller_class}."):
                            return candidate
                    # Otherwise take first available
                    return candidates[0] if candidates else None
            
            # Standalone caller - look for standalone function
            if call in self._graph.nodes:
                return call
            
            if call in self._name_to_qualified:
                return self._name_to_qualified[call][0]
            
            # Unknown internal call - could be inherited or from mixin
            # Create a placeholder qualified name
            if caller_class:
                return f"{caller_class}.{call}"
            return call
        
        else:
            # External call - could be Class.method, module.func, or just func
            
            # Already qualified (e.g., "SomeClass.method")
            if '.' in call:
                # Check if this exact qualified name exists
                if call in self._graph.nodes:
                    return call
                
                # Try to find method part in our known methods
                parts = call.split('.')
                method_name = parts[-1]
                
                if method_name in self._name_to_qualified:
                    # Check for exact class match
                    class_prefix = '.'.join(parts[:-1])
                    for candidate in self._name_to_qualified[method_name]:
                        if candidate.endswith(f".{method_name}") and class_prefix in candidate:
                            return candidate
                    
                    # Return first match
                    return self._name_to_qualified[method_name][0]
                
                # External method not in our codebase
                # Return as-is for tracking external dependencies
                return call
            
            else:
                # Simple function name
                if call in self._graph.nodes:
                    return call
                
                if call in self._name_to_qualified:
                    return self._name_to_qualified[call][0]
                
                # Unknown external - return as-is
                return call
    
    def clear(self) -> None:
        """Clear the builder state."""
        self._graph.clear()
        self._name_to_qualified.clear()
        self._processed_files.clear()
