#!/usr/bin/env python3
"""
Basic test for TreeSitterParser implementation.

Tests that the tree-sitter parser can be imported and handles basic cases.
"""

import sys
sys.path.insert(0, '../src')

def test_imports():
    """Test that all tree-sitter components can be imported."""
    try:
        from conductor_memory.search.parsers import TreeSitterParser
        from conductor_memory.search.parsers.language_configs import get_language_config, get_language_from_extension
        from conductor_memory.search.parsers.domain_detector import detect_domain
        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_language_detection():
    """Test language detection from file extensions."""
    try:
        from conductor_memory.search.parsers.language_configs import get_language_from_extension
        
        test_cases = [
            ('.py', 'python'),
            ('.java', 'java'),
            ('.kt', 'kotlin'),
            ('.go', 'go'),
            ('.rb', 'ruby'),
            ('.cs', 'csharp'),
            ('.swift', 'swift'),
            ('.c', 'c'),
            ('.m', 'objc'),
            ('.xyz', None)  # Unsupported
        ]
        
        for ext, expected in test_cases:
            result = get_language_from_extension(ext)
            if result != expected:
                print(f"[FAIL] Language detection failed: {ext} -> {result}, expected {expected}")
                return False
        
        print("[OK] Language detection working")
        return True
    except Exception as e:
        print(f"[FAIL] Language detection failed: {e}")
        return False

def test_domain_detection():
    """Test domain classification."""
    try:
        from conductor_memory.search.parsers.domain_detector import detect_domain
        
        test_cases = [
            # (node_type, name, file_path, annotations, modifiers, language, expected_domain)
            ('class_definition', 'MyClass', 'src/main.py', [], [], 'python', 'class'),
            ('function_definition', 'test_something', 'tests/test_main.py', [], [], 'python', 'test'),
            ('function_definition', '_private_func', 'src/main.py', [], [], 'python', 'private'),
            ('function_definition', 'get_value', 'src/main.py', [], [], 'python', 'accessor'),
            ('import_statement', '', 'src/main.py', [], [], 'python', 'imports'),
            ('method_declaration', 'doSomething', 'src/Main.java', ['Test'], [], 'java', 'test'),
        ]
        
        for node_type, name, file_path, annotations, modifiers, language, expected in test_cases:
            result = detect_domain(node_type, name, file_path, annotations, modifiers, language)
            if result != expected:
                print(f"[FAIL] Domain detection failed: {name} -> {result}, expected {expected}")
                return False
        
        print("[OK] Domain detection working")
        return True
    except Exception as e:
        print(f"[FAIL] Domain detection failed: {e}")
        return False

def test_parser_supports():
    """Test that parser correctly identifies supported files."""
    try:
        from conductor_memory.search.parsers import TreeSitterParser
        
        parser = TreeSitterParser()
        
        test_cases = [
            ('example.py', True),
            ('Example.java', True),
            ('example.kt', True),
            ('example.go', True),
            ('example.rb', True),
            ('example.cs', True),
            ('example.swift', True),
            ('example.c', True),
            ('example.m', True),
            ('example.txt', False),
            ('example.xyz', False)
        ]
        
        for file_path, expected in test_cases:
            result = parser.supports(file_path)
            if result != expected:
                print(f"[FAIL] Parser support check failed: {file_path} -> {result}, expected {expected}")
                return False
        
        print("[OK] Parser support detection working")
        return True
    except Exception as e:
        print(f"[FAIL] Parser support test failed: {e}")
        return False

def test_chunking_manager_integration():
    """Test that ChunkingManager can use TreeSitterParser."""
    try:
        from conductor_memory.search.chunking import ChunkingManager
        
        manager = ChunkingManager()
        
        # Test that tree_sitter_parser property works
        parser = manager.tree_sitter_parser
        if parser is None:
            print("[WARN] TreeSitterParser not available (missing dependencies)")
            return True  # Not a failure if dependencies aren't installed yet
        
        # Test that it supports files
        supports_python = parser.supports('test.py')
        if not supports_python:
            print("[FAIL] ChunkingManager integration failed: should support Python")
            return False
        
        print("[OK] ChunkingManager integration working")
        return True
    except Exception as e:
        print(f"[FAIL] ChunkingManager integration failed: {e}")
        return False

def run_all_tests():
    """Run all basic tests."""
    print("Running basic TreeSitter tests...\n")
    
    tests = [
        test_imports,
        test_language_detection,
        test_domain_detection,
        test_parser_supports,
        test_chunking_manager_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"[CRASH] Test {test.__name__} crashed: {e}\n")
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All basic tests passed!")
        return True
    else:
        print("Some tests failed - check dependencies or implementation")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)