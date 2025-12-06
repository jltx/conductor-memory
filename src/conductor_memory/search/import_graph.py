"""
Import graph construction and centrality calculation.

Builds dependency graphs from import statements to identify "hub" files
that should be prioritized for LLM summarization.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ImportNode:
    """Represents a file node in the import graph."""
    file_path: str
    module_name: str
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    centrality_score: float = 0.0
    
    def __hash__(self):
        return hash(self.file_path)


class ImportGraph:
    """
    Builds and analyzes import dependency graphs for centrality calculation.
    
    Used to prioritize files for LLM summarization based on their importance
    in the codebase's dependency structure.
    """
    
    def __init__(self):
        self.nodes: Dict[str, ImportNode] = {}
        self.graph: Optional[nx.DiGraph] = None
        self._centrality_calculated = False
    
    def add_file(self, file_path: str, imports: List[Dict[str, str]]) -> None:
        """
        Add a file and its imports to the graph.
        
        Args:
            file_path: Path to the source file
            imports: List of import dictionaries from heuristic extraction
        """
        try:
            # Normalize file path
            normalized_path = str(Path(file_path).resolve())
            
            # Extract module name from file path
            module_name = self._path_to_module(file_path)
            
            # Create or update node
            if normalized_path not in self.nodes:
                self.nodes[normalized_path] = ImportNode(
                    file_path=normalized_path,
                    module_name=module_name
                )
            
            node = self.nodes[normalized_path]
            
            # Process imports
            for import_info in imports:
                imported_module = import_info.get('module', '')
                if imported_module and imported_module not in node.imports:
                    node.imports.append(imported_module)
            
            # Mark that centrality needs recalculation
            self._centrality_calculated = False
            
        except Exception as e:
            logger.warning(f"Failed to add file to import graph: {file_path}: {e}")
    
    def build_graph(self) -> None:
        """Build the NetworkX directed graph from import relationships."""
        try:
            self.graph = nx.DiGraph()
            
            # Add all nodes
            for file_path, node in self.nodes.items():
                self.graph.add_node(file_path, **{
                    'module_name': node.module_name,
                    'import_count': len(node.imports)
                })
            
            # Add edges for import relationships
            for file_path, node in self.nodes.items():
                for imported_module in node.imports:
                    # Try to resolve imported module to a file path
                    target_file = self._resolve_import_to_file(imported_module)
                    if target_file and target_file in self.nodes:
                        # Add edge: file_path imports target_file
                        self.graph.add_edge(file_path, target_file)
                        
                        # Update imported_by relationship
                        if file_path not in self.nodes[target_file].imported_by:
                            self.nodes[target_file].imported_by.append(file_path)
            
            logger.info(f"Built import graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to build import graph: {e}")
            self.graph = nx.DiGraph()  # Empty graph as fallback
    
    def calculate_centrality(self) -> Dict[str, float]:
        """
        Calculate centrality scores for all files.
        
        Uses a combination of PageRank and in-degree centrality to identify
        important "hub" files that are heavily imported by others.
        
        Returns:
            Dictionary mapping file paths to centrality scores (0.0 to 1.0)
        """
        if self.graph is None:
            self.build_graph()
        
        if len(self.graph.nodes) == 0:
            return {}
        
        try:
            centrality_scores = {}
            
            # Calculate PageRank (considers the importance of importers)
            try:
                pagerank_scores = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                # Fallback to uniform scores if PageRank fails
                pagerank_scores = {node: 1.0 / len(self.graph.nodes) for node in self.graph.nodes}
            
            # Calculate in-degree centrality (number of files importing this file)
            in_degree_scores = dict(self.graph.in_degree())
            max_in_degree = max(in_degree_scores.values()) if in_degree_scores else 1
            
            # Normalize in-degree scores
            normalized_in_degree = {
                node: score / max_in_degree 
                for node, score in in_degree_scores.items()
            }
            
            # Combine PageRank and in-degree centrality
            # Weight: 60% PageRank, 40% in-degree
            for node in self.graph.nodes:
                pagerank_score = pagerank_scores.get(node, 0.0)
                in_degree_score = normalized_in_degree.get(node, 0.0)
                
                combined_score = (0.6 * pagerank_score) + (0.4 * in_degree_score)
                centrality_scores[node] = combined_score
                
                # Update node centrality score
                if node in self.nodes:
                    self.nodes[node].centrality_score = combined_score
            
            self._centrality_calculated = True
            
            logger.info(f"Calculated centrality for {len(centrality_scores)} files")
            return centrality_scores
            
        except Exception as e:
            logger.error(f"Failed to calculate centrality: {e}")
            # Return uniform scores as fallback
            return {node: 0.5 for node in self.graph.nodes}
    
    def get_priority_queue(self, max_files: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get files sorted by centrality score for prioritized processing.
        
        Args:
            max_files: Maximum number of files to return (None for all)
            
        Returns:
            List of (file_path, centrality_score) tuples, sorted by score descending
        """
        if not self._centrality_calculated:
            self.calculate_centrality()
        
        # Sort files by centrality score (descending)
        priority_list = [
            (node.file_path, node.centrality_score)
            for node in self.nodes.values()
        ]
        priority_list.sort(key=lambda x: x[1], reverse=True)
        
        if max_files is not None:
            priority_list = priority_list[:max_files]
        
        return priority_list
    
    def get_file_stats(self, file_path: str) -> Optional[Dict[str, any]]:
        """
        Get statistics for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file statistics or None if file not found
        """
        normalized_path = str(Path(file_path).resolve())
        
        if normalized_path not in self.nodes:
            return None
        
        node = self.nodes[normalized_path]
        
        return {
            'file_path': node.file_path,
            'module_name': node.module_name,
            'imports_count': len(node.imports),
            'imported_by_count': len(node.imported_by),
            'centrality_score': node.centrality_score,
            'imports': node.imports,
            'imported_by': node.imported_by
        }
    
    def get_graph_stats(self) -> Dict[str, any]:
        """Get overall graph statistics."""
        if self.graph is None:
            self.build_graph()
        
        stats = {
            'total_files': len(self.nodes),
            'total_edges': len(self.graph.edges) if self.graph else 0,
            'centrality_calculated': self._centrality_calculated
        }
        
        if self._centrality_calculated:
            scores = [node.centrality_score for node in self.nodes.values()]
            if scores:
                stats.update({
                    'avg_centrality': sum(scores) / len(scores),
                    'max_centrality': max(scores),
                    'min_centrality': min(scores)
                })
        
        return stats
    
    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        try:
            path = Path(file_path)
            # Remove extension and convert path separators to dots
            module = str(path.with_suffix('')).replace('/', '.').replace('\\', '.')
            # Remove leading dots and common prefixes
            module = module.lstrip('.')
            
            # Remove common source directory prefixes
            for prefix in ['src.', 'lib.', 'app.', 'source.']:
                if module.startswith(prefix):
                    module = module[len(prefix):]
                    break
            
            return module
        except Exception:
            return file_path
    
    def _resolve_import_to_file(self, imported_module: str) -> Optional[str]:
        """
        Attempt to resolve an imported module name to a file path.
        
        This is a best-effort approach that works for relative imports
        and common patterns. For complex module resolution, this would
        need to be enhanced with language-specific logic.
        """
        try:
            # Look for exact module name matches
            for file_path, node in self.nodes.items():
                if node.module_name == imported_module:
                    return file_path
            
            # Look for partial matches (e.g., import might be a submodule)
            for file_path, node in self.nodes.items():
                # Check if the file's module is a parent of the imported module
                if imported_module.startswith(node.module_name + '.'):
                    return file_path
                # Check if the imported module is a parent of the file's module
                if node.module_name.startswith(imported_module + '.'):
                    return file_path
            
            # Look for file name matches (without path)
            imported_name = imported_module.split('.')[-1]
            for file_path, node in self.nodes.items():
                file_name = Path(file_path).stem
                if file_name == imported_name:
                    return file_path
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to resolve import '{imported_module}': {e}")
            return None
    
    def export_graph(self, format: str = 'gexf') -> Optional[str]:
        """
        Export the graph to a file for visualization.
        
        Args:
            format: Export format ('gexf', 'graphml', 'json')
            
        Returns:
            Exported graph data as string, or None if export fails
        """
        if self.graph is None:
            self.build_graph()
        
        try:
            if format == 'gexf':
                import io
                buffer = io.StringIO()
                nx.write_gexf(self.graph, buffer)
                return buffer.getvalue()
            elif format == 'graphml':
                import io
                buffer = io.StringIO()
                nx.write_graphml(self.graph, buffer)
                return buffer.getvalue()
            elif format == 'json':
                import json
                data = nx.node_link_data(self.graph)
                return json.dumps(data, indent=2)
            else:
                logger.warning(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return None