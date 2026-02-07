"""
Multi-language code parsers for semantic chunking.

This package provides AST-based parsing for multiple programming languages
using tree-sitter, enabling accurate domain tagging and chunk boundaries.

Also includes specialized parsers for non-code content like conversations.
"""

from .tree_sitter_parser import TreeSitterParser
from .conversation_parser import ConversationParser

__all__ = ['TreeSitterParser', 'ConversationParser']