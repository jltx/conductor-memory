"""
Multi-language code parsers for semantic chunking.

This package provides AST-based parsing for multiple programming languages
using tree-sitter, enabling accurate domain tagging and chunk boundaries.
"""

from .tree_sitter_parser import TreeSitterParser

__all__ = ['TreeSitterParser']