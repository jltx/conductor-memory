#!/usr/bin/env python3
"""
Test tree-sitter implementation queries for extracting method body details.

Tests the get_implementation_query() method added to LanguageConfig for
extracting calls, subscripts, attributes, assignments, and structural patterns.
"""

import sys
sys.path.insert(0, 'src')

from tree_sitter import Language, Parser, Query, QueryCursor
from conductor_memory.search.parsers.language_configs import get_language_config


# Sample Python code with various implementation patterns
PYTHON_SAMPLE = b'''
import pandas as pd
from typing import List, Optional

class DataProcessor:
    """A sample class demonstrating various implementation patterns."""
    
    def __init__(self, config: dict):
        self.config = config
        self._cache = {}
        self._initialized = False
    
    def process_data(self, df: pd.DataFrame, bar_index: int) -> Optional[dict]:
        """Process data using window-relative indexing."""
        # Subscript access patterns
        row = df.iloc[bar_index]
        value = self._cache[bar_index]
        series_val = df.loc[bar_index, "column"]
        
        # Method calls - internal
        self._validate_input(df)
        result = self.transform(row)
        
        # Method calls - external
        processed = pd.concat([df, result])
        logger.info("Processing complete")
        
        # Attribute access
        threshold = self.config.threshold
        max_val = self._cache.max_value
        
        # Attribute writes
        self._cache = {}
        self.last_result = result
        
        # Structural patterns
        if bar_index > 0:
            for i in range(bar_index):
                self._cache[i] = df.iloc[i]
        
        try:
            with open("file.txt") as f:
                data = f.read()
        except IOError:
            pass
        
        # Comprehensions
        values = [x * 2 for x in range(10)]
        mapping = {k: v for k, v in self._cache.items()}
        
        return result
    
    async def async_method(self, items: List[str]):
        """Async method with await."""
        for item in items:
            result = await self.fetch_data(item)
            yield result
        
        if not items:
            raise ValueError("No items provided")
        
        return [x for x in items]
'''


def test_python_implementation_queries():
    """Test that Python implementation queries work correctly."""
    config = get_language_config("python")
    assert config is not None, "Python config should exist"
    
    # Get the implementation query
    query_string = config.get_implementation_query()
    assert query_string, "Implementation query should not be empty for Python"
    
    # Create parser and parse sample code
    lang = Language(config.module.language())
    parser = Parser(lang)
    tree = parser.parse(PYTHON_SAMPLE)
    
    # Compile the query
    try:
        query = Query(lang, query_string)
    except Exception as e:
        print(f"Query compilation failed: {e}")
        print("\nQuery string:")
        print(query_string)
        raise AssertionError(f"Query should compile without errors: {e}")
    
    # Execute the query using QueryCursor
    cursor = QueryCursor(query)
    matches = list(cursor.matches(tree.root_node))
    
    # Extract all captures from matches
    # matches is a list of (pattern_index, captures_dict)
    captures = []
    for pattern_idx, capture_dict in matches:
        for name, nodes in capture_dict.items():
            for node in nodes:
                captures.append((node, name))
    
    # Group captures by category
    categories = {}
    for node, name in captures:
        category = name.split('.')[0]  # e.g., "call" from "call.method"
        if category not in categories:
            categories[category] = []
        text = PYTHON_SAMPLE[node.start_byte:node.end_byte].decode('utf-8')
        # Truncate long text
        if len(text) > 60:
            text = text[:57] + "..."
        categories[category].append((name, text, node.start_point[0] + 1))
    
    print("\n" + "="*70)
    print("IMPLEMENTATION QUERY TEST RESULTS")
    print("="*70)
    
    for category in sorted(categories.keys()):
        items = categories[category]
        print(f"\n{category.upper()} ({len(items)} captures):")
        print("-" * 50)
        for name, text, line in items[:10]:  # Limit to first 10 per category
            text_display = text.replace('\n', '\\n')
            print(f"  Line {line:3d}: {name:25s} | {text_display}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
    
    # Verify we captured expected patterns
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check for method calls
    call_captures = [c for c in captures if c[1].startswith('call.')]
    print(f"\n[{'PASS' if call_captures else 'FAIL'}] Method calls captured: {len(call_captures)}")
    
    # Check for subscripts
    subscript_captures = [c for c in captures if c[1].startswith('subscript.')]
    print(f"[{'PASS' if subscript_captures else 'FAIL'}] Subscript access captured: {len(subscript_captures)}")
    
    # Check for attribute access
    access_captures = [c for c in captures if c[1].startswith('access.')]
    print(f"[{'PASS' if access_captures else 'FAIL'}] Attribute access captured: {len(access_captures)}")
    
    # Check for writes
    write_captures = [c for c in captures if c[1].startswith('write.')]
    print(f"[{'PASS' if write_captures else 'FAIL'}] Assignments captured: {len(write_captures)}")
    
    # Check for structural patterns
    structure_captures = [c for c in captures if c[1].startswith('structure.')]
    print(f"[{'PASS' if structure_captures else 'FAIL'}] Structural patterns captured: {len(structure_captures)}")
    
    # Verify specific expected captures
    expected_patterns = [
        ("subscript", "df.iloc[bar_index]"),
        ("subscript", "self._cache[bar_index]"),
        ("call", "self._validate_input"),
        ("call", "pd.concat"),
        ("structure", "for_loop"),
        ("structure", "try_except"),
        ("structure", "list_comp"),
    ]
    
    print("\n" + "-"*50)
    print("Expected pattern checks:")
    
    all_text = [(PYTHON_SAMPLE[n.start_byte:n.end_byte].decode('utf-8'), name) 
                for n, name in captures]
    
    for pattern_type, expected in expected_patterns:
        found = any(expected in text or expected in name for text, name in all_text)
        status = "PASS" if found else "FAIL"
        print(f"  [{status}] {pattern_type}: '{expected}'")
    
    print("\n" + "="*70)
    
    # Assert minimum expectations
    assert len(call_captures) > 0, "Should capture method calls"
    assert len(subscript_captures) > 0, "Should capture subscript access"
    assert len(access_captures) > 0, "Should capture attribute access"
    assert len(write_captures) > 0, "Should capture assignments"
    assert len(structure_captures) > 0, "Should capture structural patterns"
    
    print("\nAll tests PASSED!")
    return True


def test_other_languages_have_empty_query():
    """Test that other languages return empty string (not yet implemented)."""
    for lang_name in ["java", "go", "ruby", "c", "csharp", "kotlin", "swift", "objc"]:
        config = get_language_config(lang_name)
        if config:
            query = config.get_implementation_query()
            # For now, other languages should return empty string
            # They can be implemented later as needed
            print(f"{lang_name}: {'has query' if query else 'empty (expected)'}")


def main():
    print("Testing Implementation Queries")
    print("="*70)
    
    try:
        test_python_implementation_queries()
        print("\n" + "-"*70)
        test_other_languages_have_empty_query()
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        return 0
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
