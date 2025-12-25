#!/usr/bin/env python3
"""
Test script to verify tree-sitter queries for all supported languages.

This script:
1. Tests that each language module can be loaded
2. Prints the actual AST node types for sample code
3. Tests that our queries work correctly
4. Reports which queries pass/fail
"""

import sys
sys.path.insert(0, '../src')

from tree_sitter import Language, Parser, Query, QueryCursor


# Sample code for each language
SAMPLE_CODE = {
    'python': b'''
import os
from typing import List

class MyClass:
    """A sample class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        return self.name

def standalone_function():
    pass

def test_something():
    assert True
''',
    
    'java': b'''
import java.util.List;
import java.util.ArrayList;

public class MyClass {
    private String name;
    
    public MyClass(String name) {
        this.name = name;
    }
    
    public String getName() {
        return name;
    }
    
    @Test
    public void testSomething() {
        assertTrue(true);
    }
}
''',
    
    'kotlin': b'''
import kotlin.collections.List

class MyClass(private val name: String) {
    fun getName(): String {
        return name
    }
    
    @Test
    fun testSomething() {
        assertTrue(true)
    }
}

object MySingleton {
    val instance = "singleton"
}

fun standaloneFunction(): Unit {
    println("Hello")
}
''',
    
    'go': b'''
package main

import (
    "fmt"
    "os"
)

type MyStruct struct {
    Name string
}

func (m *MyStruct) GetName() string {
    return m.Name
}

func standaloneFunction() {
    fmt.Println("Hello")
}

func TestSomething(t *testing.T) {
    // test
}
''',
    
    'ruby': b'''
require 'json'
require_relative 'helper'

class MyClass
  def initialize(name)
    @name = name
  end
  
  def get_name
    @name
  end
end

def standalone_function
  puts "Hello"
end

describe "MyClass" do
  it "should work" do
    expect(true).to be true
  end
end
''',
    
    'c': b'''
#include <stdio.h>
#include <stdlib.h>

struct MyStruct {
    char* name;
};

void my_function(struct MyStruct* s) {
    printf("%s", s->name);
}

int main() {
    return 0;
}
''',
    
    'csharp': b'''
using System;
using System.Collections.Generic;

public class MyClass {
    private string name;
    
    public MyClass(string name) {
        this.name = name;
    }
    
    public string GetName() {
        return name;
    }
    
    [Test]
    public void TestSomething() {
        Assert.True(true);
    }
}
''',
    
    'swift': b'''
import Foundation

class MyClass {
    private var name: String
    
    init(name: String) {
        self.name = name
    }
    
    func getName() -> String {
        return name
    }
}

struct MyStruct {
    var value: Int
}

func standaloneFunction() {
    print("Hello")
}
''',
    
    'objc': b'''
#import <Foundation/Foundation.h>

@interface MyClass : NSObject
@property (nonatomic, strong) NSString *name;
- (instancetype)initWithName:(NSString *)name;
- (NSString *)getName;
@end

@implementation MyClass
- (instancetype)initWithName:(NSString *)name {
    self = [super init];
    if (self) {
        _name = name;
    }
    return self;
}

- (NSString *)getName {
    return self.name;
}
@end
'''
}


def get_language_module(lang_name):
    """Import and return the tree-sitter language module."""
    modules = {
        'python': 'tree_sitter_python',
        'java': 'tree_sitter_java',
        'kotlin': 'tree_sitter_kotlin',
        'go': 'tree_sitter_go',
        'ruby': 'tree_sitter_ruby',
        'c': 'tree_sitter_c',
        'csharp': 'tree_sitter_c_sharp',
        'swift': 'tree_sitter_swift',
        'objc': 'tree_sitter_objc',
    }
    
    module_name = modules.get(lang_name)
    if not module_name:
        return None
    
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"  [ERROR] Could not import {module_name}: {e}")
        return None


def print_tree(node, indent=0, max_depth=4):
    """Print the AST tree structure."""
    if indent > max_depth:
        return
    
    # Get node text (truncated)
    try:
        text = node.text[:40].decode('utf-8', errors='replace').replace('\n', '\\n')
    except:
        text = "<error>"
    
    prefix = "  " * indent
    print(f"{prefix}{node.type}: {text!r}")
    
    for child in node.children:
        if child.is_named:
            print_tree(child, indent + 1, max_depth)


def validate_query(lang, query_string, description=""):
    """Validate if a query is valid."""
    try:
        query = Query(lang, query_string)
        return True, None
    except Exception as e:
        return False, str(e)


def find_node_types(node, target_types, results=None, depth=0, max_depth=10):
    """Find all occurrences of specific node types."""
    if results is None:
        results = []
    
    if depth > max_depth:
        return results
    
    if node.type in target_types:
        results.append((node.type, depth, node.start_point[0]))
    
    for child in node.children:
        find_node_types(child, target_types, results, depth + 1, max_depth)
    
    return results


def check_language(lang_name):
    """Check a single language's tree-sitter support."""
    print(f"\n{'='*60}")
    print(f"Testing: {lang_name.upper()}")
    print('='*60)
    
    # Get module
    module = get_language_module(lang_name)
    if not module:
        print(f"  [SKIP] Language module not available")
        return False
    
    # Create language and parser
    try:
        lang = Language(module.language())
        parser = Parser(lang)
    except Exception as e:
        print(f"  [ERROR] Failed to create parser: {e}")
        return False
    
    # Get sample code
    code = SAMPLE_CODE.get(lang_name)
    if not code:
        print(f"  [SKIP] No sample code available")
        return False
    
    # Parse the code
    try:
        tree = parser.parse(code)
    except Exception as e:
        print(f"  [ERROR] Failed to parse code: {e}")
        return False
    
    # Print the tree structure
    print("\n  AST Structure (first 4 levels):")
    print("  " + "-"*40)
    print_tree(tree.root_node)
    
    # Determine what node types we're looking for
    target_types = ['class', 'function', 'method', 'import', 'struct', 'interface', 
                    'class_definition', 'function_definition', 'method_definition',
                    'class_declaration', 'method_declaration', 'function_declaration',
                    'import_statement', 'import_from_statement', 'import_declaration',
                    'import', 'import_header', 'using_directive', 'preproc_include',
                    'object_declaration', 'type_declaration', 'struct_specifier',
                    'class_interface', 'class_implementation']
    
    found = find_node_types(tree.root_node, target_types)
    if found:
        print("\n  Key node types found:")
        for node_type, depth, line in found:
            print(f"    - {node_type} (depth {depth}, line {line})")
    
    # Test queries
    print("\n  Query Tests:")
    print("  " + "-"*40)
    
    # Basic queries to test
    basic_queries = [
        ("(class_definition) @cls", "Python-style class"),
        ("(class_declaration) @cls", "Java/Kotlin-style class"),
        ("(function_definition) @func", "Python-style function"),
        ("(function_declaration) @func", "Kotlin/Go-style function"),
        ("(method_declaration) @method", "Java/C#-style method"),
        ("(import_statement) @imp", "Python import"),
        ("(import_from_statement) @imp", "Python from-import"),
        ("(import_declaration) @imp", "Java import"),
        ("(import) @imp", "Kotlin import"),
        ("(import_header) @imp", "Kotlin import header"),
        ("(using_directive) @imp", "C# using"),
        ("(preproc_include) @imp", "C include"),
        ("(object_declaration) @obj", "Kotlin object"),
        ("(struct_specifier) @struct", "C struct"),
        ("(type_declaration) @type", "Go type"),
    ]
    
    working_queries = []
    for query_str, desc in basic_queries:
        success, error = validate_query(lang, query_str, desc)
        if success:
            # Also test if it actually matches anything
            try:
                query = Query(lang, query_str)
                cursor = QueryCursor(query)
                matches = list(cursor.matches(tree.root_node))
                if matches:
                    print(f"    [OK] {query_str} - {len(matches)} matches")
                    working_queries.append((query_str, len(matches)))
                else:
                    print(f"    [--] {query_str} - valid but no matches")
            except Exception as e:
                print(f"    [??] {query_str} - error executing: {e}")
        else:
            if "Invalid node type" in str(error):
                pass  # Don't spam with invalid node types
            else:
                print(f"    [FAIL] {query_str} - {error}")
    
    # Test field access queries
    print("\n  Field Access Queries:")
    field_queries = [
        "(class_definition name: (identifier) @name) @cls",
        "(class_declaration name: (identifier) @name) @cls",
        "(function_definition name: (identifier) @name) @func",
        "(function_declaration name: (identifier) @name) @func",
        "(method_declaration name: (identifier) @name) @method",
    ]
    
    for query_str in field_queries:
        success, error = validate_query(lang, query_str)
        if success:
            try:
                query = Query(lang, query_str)
                cursor = QueryCursor(query)
                matches = list(cursor.matches(tree.root_node))
                if matches:
                    print(f"    [OK] {query_str} - {len(matches)} matches")
            except:
                pass
    
    print("\n  Summary of working queries:")
    for q, count in working_queries:
        print(f"    - {q} ({count} matches)")
    
    return True


def main():
    print("Tree-sitter Query Test Script")
    print("="*60)
    print("This script tests tree-sitter queries for all supported languages.")
    print("It will show which node types exist and which queries work.")
    
    languages = ['python', 'java', 'kotlin', 'go', 'ruby', 'c', 'csharp', 'swift', 'objc']
    
    results = {}
    for lang in languages:
        try:
            results[lang] = check_language(lang)
        except Exception as e:
            print(f"\n[ERROR] Failed to test {lang}: {e}")
            import traceback
            traceback.print_exc()
            results[lang] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for lang, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {lang}")


if __name__ == "__main__":
    main()
