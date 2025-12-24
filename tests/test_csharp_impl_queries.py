#!/usr/bin/env python3
"""
Test C# implementation queries for tree-sitter.

Tests the get_implementation_query() method for CSharpConfig.
"""

import sys
sys.path.insert(0, 'src')

from tree_sitter import Language, Parser, Query, QueryCursor
from conductor_memory.search.parsers.language_configs import get_language_config


# Comprehensive C# sample code with various implementation patterns
CSHARP_SAMPLE = b'''
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

public class DataService
{
    private Dictionary<string, int> _cache;
    private ILogger _logger;
    
    public DataService(ILogger logger)
    {
        this._logger = logger;
        this._cache = new Dictionary<string, int>();
    }
    
    public async Task<List<string>> ProcessDataAsync(List<Item> items)
    {
        // Method calls on objects
        var result = items.Select(x => x.Name).ToList();
        this._logger.LogInfo("Processing started");
        
        // Static method call
        var formatted = String.Format("Count: {0}", items.Count);
        
        // Property access
        var count = items.Count;
        var first = this._cache["first"];
        
        // Property assignment
        this._cache["key"] = 42;
        this.LastUpdate = DateTime.Now;
        
        // Index access
        var item = items[0];
        var cached = _cache["test"];
        
        // For loop
        for (int i = 0; i < items.Count; i++)
        {
            var current = items[i];
        }
        
        // Foreach loop
        foreach (var item2 in items)
        {
            Console.WriteLine(item2.Name);
        }
        
        // While loop
        int counter = 0;
        while (counter < 10)
        {
            counter++;
        }
        
        // If statement
        if (items.Count > 0)
        {
            return result;
        }
        
        // Try-catch-finally
        try
        {
            await Task.Delay(100);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex.Message);
        }
        finally
        {
            _cache.Clear();
        }
        
        // Switch statement
        switch (count)
        {
            case 0:
                return new List<string>();
            default:
                break;
        }
        
        // Using statement
        using (var stream = new MemoryStream())
        {
            stream.WriteByte(0);
        }
        
        // LINQ expression
        var filtered = from i in items
                       where i.Value > 0
                       select i.Name;
        
        // Await expression
        var data = await FetchDataAsync();
        
        return result;
    }
    
    private Task<string> FetchDataAsync()
    {
        return Task.FromResult("data");
    }
    
    public DateTime LastUpdate { get; set; }
}

public class Item
{
    public string Name { get; set; }
    public int Value { get; set; }
}
'''


def test_csharp_implementation_queries():
    """Test that C# implementation queries work correctly."""
    config = get_language_config("csharp")
    assert config is not None, "C# config should exist"
    
    # Get the implementation query
    query_string = config.get_implementation_query()
    assert query_string, "Implementation query should not be empty for C#"
    
    print(f"Query length: {len(query_string)} characters")
    
    # Create parser and parse sample code
    lang = Language(config.module.language())
    parser = Parser(lang)
    tree = parser.parse(CSHARP_SAMPLE)
    
    # Compile the query
    try:
        query = Query(lang, query_string)
    except Exception as e:
        print(f"\nQuery compilation failed: {e}")
        print("\nQuery string (showing near error):")
        # Try to show relevant part of query
        lines = query_string.split('\n')
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
        raise AssertionError(f"Query should compile without errors: {e}")
    
    # Execute the query using QueryCursor
    cursor = QueryCursor(query)
    matches = list(cursor.matches(tree.root_node))
    
    # Extract all captures from matches
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
        text = CSHARP_SAMPLE[node.start_byte:node.end_byte].decode('utf-8')
        # Truncate long text
        if len(text) > 60:
            text = text[:57] + "..."
        categories[category].append((name, text, node.start_point[0] + 1))
    
    print("\n" + "="*70)
    print("C# IMPLEMENTATION QUERY TEST RESULTS")
    print("="*70)
    
    for category in sorted(categories.keys()):
        items = categories[category]
        print(f"\n{category.upper()} ({len(items)} captures):")
        print("-" * 50)
        for name, text, line in items[:10]:  # Limit to first 10 per category
            text_display = text.replace('\n', '\\n')
            print(f"  Line {line:3d}: {name:25s} | {text_display[:40]}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
    
    # Verify we captured expected patterns
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check for method calls
    call_captures = [c for c in captures if c[1].startswith('call.')]
    print(f"\n[{'PASS' if call_captures else 'FAIL'}] Method calls captured: {len(call_captures)}")
    
    # Check for subscripts (element access)
    subscript_captures = [c for c in captures if c[1].startswith('subscript.')]
    print(f"[{'PASS' if subscript_captures else 'FAIL'}] Element access captured: {len(subscript_captures)}")
    
    # Check for property/attribute access
    access_captures = [c for c in captures if c[1].startswith('access.')]
    print(f"[{'PASS' if access_captures else 'FAIL'}] Property access captured: {len(access_captures)}")
    
    # Check for writes
    write_captures = [c for c in captures if c[1].startswith('write.')]
    print(f"[{'PASS' if write_captures else 'FAIL'}] Assignments captured: {len(write_captures)}")
    
    # Check for structural patterns
    structure_captures = [c for c in captures if c[1].startswith('structure.')]
    print(f"[{'PASS' if structure_captures else 'FAIL'}] Structural patterns captured: {len(structure_captures)}")
    
    # Verify specific expected captures
    expected_patterns = [
        ("call", "this._logger.LogInfo"),
        ("call", "String.Format"),
        ("subscript", "items[0]"),
        ("subscript", "_cache"),
        ("access", "items.Count"),
        ("structure", "for_loop"),
        ("structure", "foreach_loop"),
        ("structure", "while_loop"),
        ("structure", "conditional"),
        ("structure", "try_catch"),
        ("structure", "using"),
        ("structure", "linq_query"),
        ("structure", "await"),
        ("structure", "lambda"),
    ]
    
    print("\n" + "-"*50)
    print("Expected pattern checks:")
    
    all_text = [(CSHARP_SAMPLE[n.start_byte:n.end_byte].decode('utf-8'), name) 
                for n, name in captures]
    
    for pattern_type, expected in expected_patterns:
        found = any(expected in text or expected in name for text, name in all_text)
        status = "PASS" if found else "FAIL"
        print(f"  [{status}] {pattern_type}: '{expected}'")
    
    print("\n" + "="*70)
    
    # Assert minimum expectations
    assert len(call_captures) > 0, "Should capture method calls"
    assert len(subscript_captures) > 0, "Should capture element access"
    assert len(access_captures) > 0, "Should capture property access"
    assert len(write_captures) > 0, "Should capture assignments"
    assert len(structure_captures) > 0, "Should capture structural patterns"
    
    print("\nAll tests PASSED!")
    return True


def main():
    print("Testing C# Implementation Queries")
    print("="*70)
    
    try:
        test_csharp_implementation_queries()
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
