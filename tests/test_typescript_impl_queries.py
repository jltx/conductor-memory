#!/usr/bin/env python3
"""
Test tree-sitter implementation queries for TypeScript.

Tests the get_implementation_query() method for TypeScriptConfig.
"""

import sys
sys.path.insert(0, 'src')

from tree_sitter import Language, Parser, Query, QueryCursor
from conductor_memory.search.parsers.language_configs import get_language_config


# Sample TypeScript code with various implementation patterns
TYPESCRIPT_SAMPLE = b'''
import { User, UserService } from './services';
import axios from 'axios';

interface CacheOptions {
    ttl: number;
    maxSize: number;
}

class UserManager {
    private cache: Map<string, User>;
    private options: CacheOptions;
    
    constructor(options: CacheOptions) {
        this.options = options;
        this.cache = new Map();
    }
    
    async getUser(id: string): Promise<User | null> {
        // Check cache first (member access)
        if (this.cache.has(id)) {
            return this.cache.get(id)!;
        }
        
        try {
            // Method call on imported service
            const user = await this.fetchUser(id);
            
            // Assignment to member
            this.cache.set(id, user);
            
            return user;
        } catch (error) {
            console.error('Failed to fetch user:', error);
            throw error;
        }
    }
    
    private async fetchUser(id: string): Promise<User> {
        const response = await axios.get(`/api/users/${id}`);
        const data = response.data as User;
        return data;
    }
    
    processUsers(users: User[]): void {
        // For-of loop
        for (const user of users) {
            console.log(user.name);
            this.handleUser(user);
        }
        
        // Traditional for loop with subscript
        for (let i = 0; i < users.length; i++) {
            const user = users[i];
            this.processIndex(i, user);
        }
        
        // For-in loop
        const obj = { a: 1, b: 2 };
        for (const key in obj) {
            console.log(key);
        }
        
        // While loop
        let idx = 0;
        while (idx < users.length) {
            idx++;
        }
    }
    
    handleSwitch(type: string): number {
        switch (type) {
            case 'admin':
                return 1;
            case 'user':
                return 2;
            default:
                return 0;
        }
    }
    
    assignmentTests(): void {
        // Member assignment
        this.cache = new Map();
        
        // Simple variable declarations
        const x = 5;
        let y = 10;
        
        // Subscript assignment
        const arr: number[] = [1, 2, 3];
        arr[0] = 99;
        
        // Ternary expression
        const result = x > 0 ? 'positive' : 'negative';
    }
    
    arrowFunctionTest(): void {
        // Arrow function
        const double = (x: number): number => x * 2;
        
        // Arrow with body
        const process = (items: string[]): string[] => {
            return items.map(item => item.toUpperCase());
        };
    }
}

function standalone(x: number): number {
    const result = Math.abs(x);
    return result;
}

// Top-level arrow function
const topLevelArrow = async (id: string): Promise<void> => {
    await fetch(`/api/${id}`);
};
'''


def test_typescript_implementation_queries():
    """Test that TypeScript implementation queries work correctly."""
    config = get_language_config("typescript")
    assert config is not None, "TypeScript config should exist"
    
    # Get the implementation query
    query_string = config.get_implementation_query()
    assert query_string, "Implementation query should not be empty for TypeScript"
    
    # Create parser and parse sample code
    lang = Language(config.module.language())
    parser = Parser(lang)
    tree = parser.parse(TYPESCRIPT_SAMPLE)
    
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
    captures = []
    for pattern_idx, capture_dict in matches:
        for name, nodes in capture_dict.items():
            for node in nodes:
                captures.append((node, name))
    
    # Group captures by category
    categories = {}
    for node, name in captures:
        category = name.split('.')[0]
        if category not in categories:
            categories[category] = []
        text = TYPESCRIPT_SAMPLE[node.start_byte:node.end_byte].decode('utf-8')
        if len(text) > 60:
            text = text[:57] + "..."
        categories[category].append((name, text, node.start_point[0] + 1))
    
    print("\n" + "=" * 70)
    print("TYPESCRIPT IMPLEMENTATION QUERY TEST RESULTS")
    print("=" * 70)
    
    for category in sorted(categories.keys()):
        items = categories[category]
        print(f"\n{category.upper()} ({len(items)} captures):")
        print("-" * 50)
        for name, text, line in items[:10]:
            text_display = text.replace('\n', '\\n')
            print(f"  Line {line:3d}: {name:25s} | {text_display}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
    
    # Verify we captured expected patterns
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Check for method calls
    call_captures = [c for c in captures if c[1].startswith('call.')]
    print(f"\n[{'PASS' if call_captures else 'FAIL'}] Method calls captured: {len(call_captures)}")
    
    # Check for subscript access
    subscript_captures = [c for c in captures if c[1].startswith('subscript.')]
    print(f"[{'PASS' if subscript_captures else 'FAIL'}] Subscript access captured: {len(subscript_captures)}")
    
    # Check for property access
    access_captures = [c for c in captures if c[1].startswith('access.')]
    print(f"[{'PASS' if access_captures else 'FAIL'}] Property access captured: {len(access_captures)}")
    
    # Check for writes
    write_captures = [c for c in captures if c[1].startswith('write.')]
    print(f"[{'PASS' if write_captures else 'FAIL'}] Writes captured: {len(write_captures)}")
    
    # Check for structural patterns
    structure_captures = [c for c in captures if c[1].startswith('structure.')]
    print(f"[{'PASS' if structure_captures else 'FAIL'}] Structural patterns captured: {len(structure_captures)}")
    
    # Detailed structural checks
    print("\n  Structural breakdown:")
    struct_types = {}
    for node, name in structure_captures:
        if name not in struct_types:
            struct_types[name] = 0
        struct_types[name] += 1
    
    for stype, count in sorted(struct_types.items()):
        print(f"    - {stype}: {count}")
    
    # Verify key patterns
    assertions = [
        (len(call_captures) > 10, "Should have multiple method calls"),
        (len(subscript_captures) >= 2, "Should have subscript access (users[i], arr[0])"),
        (len(access_captures) > 10, "Should have property access (this.cache, etc.)"),
        (len(write_captures) > 5, "Should have multiple write operations"),
        ('structure.for_loop' in struct_types, "Should have traditional for loop"),
        ('structure.for_in_of' in struct_types, "Should have for-of/for-in loops"),
        ('structure.while_loop' in struct_types, "Should have while loop"),
        ('structure.conditional' in struct_types, "Should have if statements"),
        ('structure.switch' in struct_types, "Should have switch statement"),
        ('structure.try_catch' in struct_types, "Should have try-catch"),
        ('structure.await' in struct_types, "Should have await expressions"),
        ('structure.arrow_function' in struct_types, "Should have arrow functions"),
        ('structure.throw' in struct_types, "Should have throw statements"),
        ('structure.return' in struct_types, "Should have return statements"),
    ]
    
    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)
    
    all_passed = True
    for condition, message in assertions:
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {message}")
        if not condition:
            all_passed = False
    
    assert all_passed, "Some assertions failed"
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


def test_typescript_definition_queries():
    """Test that TypeScript definition queries also work."""
    config = get_language_config("typescript")
    assert config is not None
    
    lang = Language(config.module.language())
    parser = Parser(lang)
    tree = parser.parse(TYPESCRIPT_SAMPLE)
    
    query = Query(lang, config.definition_query)
    cursor = QueryCursor(query)
    matches = list(cursor.matches(tree.root_node))
    
    captures = []
    for pattern_idx, capture_dict in matches:
        for name, nodes in capture_dict.items():
            for node in nodes:
                captures.append((node, name))
    
    print("\n" + "=" * 70)
    print("TYPESCRIPT DEFINITION QUERY TEST")
    print("=" * 70)
    
    for node, name in captures:
        text = TYPESCRIPT_SAMPLE[node.start_byte:node.end_byte].decode('utf-8')
        if len(text) > 60:
            text = text[:57] + "..."
        text = text.replace('\n', '\\n')
        print(f"  {name:20s} | {text}")
    
    # We should capture class, interface, methods, functions, imports
    assert len(captures) > 0, "Should capture definitions"
    print("\nDefinition query works correctly!")


if __name__ == "__main__":
    test_typescript_implementation_queries()
    test_typescript_definition_queries()
