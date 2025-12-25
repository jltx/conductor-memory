"""
Tests for Task E1: Simple File Detection.

Verifies that HeuristicExtractor correctly identifies "simple" files
that don't need LLM summarization across different languages.
"""

import pytest
from conductor_memory.search.heuristics import HeuristicExtractor


class TestPythonSimpleFileDetection:
    """Tests for Python simple file detection."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_python_init_is_simple(self, extractor):
        """Python __init__.py files are always detected as simple (barrel_reexport)."""
        content = """
from .module1 import Class1
from .module2 import func2
from .subpackage import SubClass

__all__ = ['Class1', 'func2', 'SubClass']
"""
        metadata = extractor.extract_file_metadata("package/__init__.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "barrel_reexport"
    
    def test_python_init_empty_is_simple(self, extractor):
        """Empty Python __init__.py files are detected as simple."""
        content = "# Empty init file\n"
        
        metadata = extractor.extract_file_metadata("mypackage/__init__.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "barrel_reexport"
    
    def test_python_init_with_functions_is_still_simple(self, extractor):
        """Python __init__.py with lazy loading functions is STILL simple.
        
        Design decision: __init__.py files are ALWAYS considered simple,
        even if they have functions. The rationale is that lazy loading 
        patterns are still fundamentally barrel/re-export behavior.
        """
        content = '''
"""Package with lazy loading."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .heavy_module import HeavyClass

__all__ = ['HeavyClass', 'get_heavy']

_lazy_imports = {}

def get_heavy():
    """Lazy load HeavyClass."""
    if 'heavy' not in _lazy_imports:
        from .heavy_module import HeavyClass
        _lazy_imports['heavy'] = HeavyClass
    return _lazy_imports['heavy']

def __getattr__(name):
    if name == 'HeavyClass':
        return get_heavy()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''
        metadata = extractor.extract_file_metadata("mypackage/__init__.py", content)
        
        assert metadata is not None
        # __init__.py is ALWAYS simple by design decision
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "barrel_reexport"
    
    def test_python_regular_file_with_functions_is_complex(self, extractor):
        """Python files with functions are NOT detected as simple."""
        content = '''
"""Module with actual implementation."""

def calculate_total(items):
    """Calculate the total value of items."""
    total = 0
    for item in items:
        total += item.value
    return total

def validate_input(data):
    """Validate the input data."""
    if not data:
        raise ValueError("Data cannot be empty")
    return True
'''
        metadata = extractor.extract_file_metadata("src/calculator.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None
    
    def test_python_file_with_classes_is_complex(self, extractor):
        """Python files with classes are NOT detected as simple."""
        content = '''
"""Module with class definition."""

class UserService:
    """Handles user operations."""
    
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id):
        return self.db.query(user_id)
'''
        metadata = extractor.extract_file_metadata("src/services/user_service.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None
    
    def test_python_constants_only_file_is_simple(self, extractor):
        """Python files with only constants are detected as simple (constants_only)."""
        content = '''
"""Configuration constants."""

API_URL = "https://api.example.com"
API_TIMEOUT = 30
MAX_RETRIES = 3
DEFAULT_LIMIT = 100

# Feature flags
ENABLE_CACHING = True
ENABLE_LOGGING = True
DEBUG_MODE = False
'''
        metadata = extractor.extract_file_metadata("src/config/constants.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "constants_only"
    
    def test_python_small_empty_file_is_simple(self, extractor):
        """Small Python files with no functions/classes are detected as simple."""
        content = '''
# Placeholder module
# TODO: Implement later
'''
        metadata = extractor.extract_file_metadata("src/placeholder.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        # Could be empty_module or constants_only depending on line count
        assert metadata.simple_file_reason in ["empty_module", "constants_only"]


class TestTypeScriptSimpleFileDetection:
    """Tests for TypeScript simple file detection."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_typescript_index_with_exports_is_simple(self, extractor):
        """TypeScript index.ts with only exports is detected as simple (barrel_reexport)."""
        content = '''
// Barrel file - re-exports from submodules
export { UserService } from './user-service';
export { ProductService } from './product-service';
export { OrderService } from './order-service';
export type { User, UserCreateInput } from './types';
'''
        metadata = extractor.extract_file_metadata("src/services/index.ts", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "barrel_reexport"
    
    def test_typescript_index_with_functions_is_complex(self, extractor):
        """TypeScript index.ts with functions is NOT detected as simple."""
        content = '''
// Index with utility functions
export { UserService } from './user-service';

export function initializeServices(config: Config): void {
    UserService.init(config);
}

export function teardownServices(): void {
    UserService.cleanup();
}
'''
        metadata = extractor.extract_file_metadata("src/services/index.ts", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None
    
    def test_typescript_types_only_file_currently_complex(self, extractor):
        """TypeScript files with type aliases are currently NOT detected as simple.
        
        NOTE: This documents current behavior. Type aliases (`type X = ...`) are 
        categorized as 'classes' by the heuristic extractor, which prevents 
        type_definitions detection. This could be enhanced in the future.
        
        For now, we test that pure interface files (without type aliases) ARE
        detected as simple - see test_typescript_pure_interfaces_is_simple.
        """
        content = '''
// Type definitions for User domain

export interface User {
    id: string;
    name: string;
    email: string;
}

export interface UserCreateInput {
    name: string;
    email: string;
}

export type UserRole = 'admin' | 'user' | 'guest';

export interface UserWithRole extends User {
    role: UserRole;
}
'''
        metadata = extractor.extract_file_metadata("src/types/user.ts", content)
        
        assert metadata is not None
        # Currently NOT detected as simple because type aliases are counted as classes
        assert metadata.is_simple_file is False
        # Type aliases are counted as classes
        assert len(metadata.classes) >= 1
    
    def test_typescript_pure_interfaces_is_simple(self, extractor):
        """TypeScript files with only interfaces (no type aliases) are simple."""
        content = '''
// Pure interface definitions (no type aliases)

export interface User {
    id: string;
    name: string;
    email: string;
}

export interface UserCreateInput {
    name: string;
    email: string;
}
'''
        metadata = extractor.extract_file_metadata("src/types/user-interfaces.ts", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "type_definitions"
    
    def test_typescript_regular_file_is_complex(self, extractor):
        """TypeScript files with classes/functions are NOT detected as simple."""
        content = '''
// User service implementation

export class UserService {
    private users: Map<string, User> = new Map();
    
    async getUser(id: string): Promise<User | null> {
        return this.users.get(id) ?? null;
    }
    
    async createUser(input: UserCreateInput): Promise<User> {
        const user: User = {
            id: generateId(),
            ...input
        };
        this.users.set(user.id, user);
        return user;
    }
}
'''
        metadata = extractor.extract_file_metadata("src/services/user-service.ts", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None


class TestJavaSimpleFileDetection:
    """Tests for Java simple file detection."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_java_package_info_is_simple(self, extractor):
        """Java package-info.java files are detected as simple (barrel_reexport)."""
        content = '''
/**
 * This package contains utility classes for data processing.
 * 
 * @since 1.0
 * @see com.example.core
 */
@NonNullApi
package com.example.utils;

import org.springframework.lang.NonNullApi;
'''
        metadata = extractor.extract_file_metadata("src/main/java/com/example/utils/package-info.java", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "barrel_reexport"
    
    def test_java_interface_with_methods_currently_complex(self, extractor):
        """Java interfaces with method declarations are currently NOT detected as simple.
        
        NOTE: This documents current behavior. Interface method declarations (abstract
        methods without bodies) are counted in metadata.methods, which increases 
        func_count and prevents type_definitions detection.
        
        Future enhancement: Distinguish between method declarations (in interfaces)
        vs method definitions (with bodies) to properly detect interface-only files.
        """
        content = '''
package com.example.api;

/**
 * Repository interface for User entities.
 */
public interface UserRepository {
    User findById(Long id);
    List<User> findAll();
    User save(User user);
    void delete(Long id);
}
'''
        metadata = extractor.extract_file_metadata("src/main/java/com/example/api/UserRepository.java", content)
        
        assert metadata is not None
        # Currently NOT detected as simple because interface methods are counted
        assert metadata.is_simple_file is False
        # Verify that interface IS detected, just methods are counted
        assert len(metadata.interfaces) == 1
        assert metadata.interfaces[0]['name'] == 'UserRepository'
        # Methods are counted (this is why it's not detected as simple)
        assert len(metadata.methods) == 4
    
    def test_java_marker_interface_is_simple(self, extractor):
        """Java marker interfaces (no methods) are detected as simple."""
        content = '''
package com.example.marker;

/**
 * Marker interface for serializable entities.
 */
public interface Serializable {
    // Marker interface - no methods
}
'''
        metadata = extractor.extract_file_metadata("src/main/java/com/example/marker/Serializable.java", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "type_definitions"
    
    def test_java_class_with_methods_is_complex(self, extractor):
        """Java files with class implementations are NOT detected as simple."""
        content = '''
package com.example.service;

public class UserService {
    private final UserRepository userRepository;
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    public User getUser(Long id) {
        return userRepository.findById(id);
    }
    
    public User createUser(String name, String email) {
        User user = new User(name, email);
        return userRepository.save(user);
    }
}
'''
        metadata = extractor.extract_file_metadata("src/main/java/com/example/service/UserService.java", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None


class TestGeneratedFileDetection:
    """Tests for generated file detection across languages."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_generated_file_do_not_edit_marker(self, extractor):
        """Files with 'DO NOT EDIT' marker are detected as simple (generated_code)."""
        content = '''
// Code generated by protoc-gen-go. DO NOT EDIT.
// source: user.proto

package proto

type User struct {
    Id    string `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`
    Name  string `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
    Email string `protobuf:"bytes,3,opt,name=email,proto3" json:"email,omitempty"`
}

func (x *User) GetId() string {
    if x != nil {
        return x.Id
    }
    return ""
}
'''
        metadata = extractor.extract_file_metadata("gen/proto/user.pb.go", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "generated_code"
    
    def test_generated_file_at_generated_marker(self, extractor):
        """Files with '@generated' marker are detected as simple (generated_code)."""
        content = '''
/**
 * @generated SignedSource<<abc123>>
 * @codegen-command: relay-compiler
 */

export type UserQuery = {
  readonly response: UserQuery$data;
  readonly variables: UserQuery$variables;
};
'''
        metadata = extractor.extract_file_metadata("__generated__/UserQuery.graphql.ts", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "generated_code"
    
    def test_generated_file_auto_generated_marker(self, extractor):
        """Files with 'auto-generated' marker are detected as simple (generated_code)."""
        content = '''
# This file is auto-generated by swagger-codegen
# Please do not modify this file manually

from __future__ import absolute_import

class ApiClient:
    def __init__(self, configuration=None):
        self.configuration = configuration or Configuration()
'''
        metadata = extractor.extract_file_metadata("client/api_client.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "generated_code"
    
    def test_generated_file_generated_by_marker(self, extractor):
        """Files with 'Generated by' marker are detected as simple (generated_code)."""
        content = '''
/* Generated by the protocol buffer compiler. */
/* Source: messages.proto */

#ifndef MESSAGES_PB_H
#define MESSAGES_PB_H

typedef struct _Message {
    int32_t id;
    char* content;
} Message;

#endif
'''
        metadata = extractor.extract_file_metadata("gen/messages.pb.h", content)
        
        # Note: C header files may not be supported by tree-sitter config
        # This test verifies behavior if parsing succeeds
        if metadata is not None:
            assert metadata.is_simple_file is True
            assert metadata.simple_file_reason == "generated_code"


class TestEmptyFileDetection:
    """Tests for empty/minimal file detection."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_empty_python_file_is_simple(self, extractor):
        """Empty Python files are detected as simple (empty_module)."""
        content = "# Empty file\n"
        
        metadata = extractor.extract_file_metadata("src/empty.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "empty_module"
    
    def test_only_comments_file_is_simple(self, extractor):
        """Files with only comments are detected as simple."""
        content = '''
# This module will contain future implementations
# Currently a placeholder
#
# TODO:
# - Implement feature X
# - Add tests for feature Y
'''
        metadata = extractor.extract_file_metadata("src/future.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason in ["empty_module", "constants_only"]
    
    def test_small_file_with_docstring_only(self, extractor):
        """Small files with only a module docstring are detected as simple."""
        content = '''
"""
This module provides utilities for data transformation.

Coming soon in version 2.0.
"""
'''
        metadata = extractor.extract_file_metadata("src/transforms.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason in ["empty_module", "constants_only"]


class TestComplexFilesNotDetectedAsSimple:
    """Negative tests ensuring complex files are NOT detected as simple."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_file_with_multiple_functions_is_complex(self, extractor):
        """Files with multiple function definitions are NOT simple."""
        content = '''
"""Utility functions for data processing."""

def parse_json(data: str) -> dict:
    """Parse JSON string to dictionary."""
    import json
    return json.loads(data)

def format_date(date_obj) -> str:
    """Format date object to ISO string."""
    return date_obj.isoformat()

def calculate_hash(content: bytes) -> str:
    """Calculate SHA256 hash of content."""
    import hashlib
    return hashlib.sha256(content).hexdigest()

def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
'''
        metadata = extractor.extract_file_metadata("src/utils.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None
    
    def test_file_with_class_and_methods_is_complex(self, extractor):
        """Files with classes and methods are NOT simple."""
        content = '''
"""Repository pattern implementation."""

class Repository:
    """Generic repository for data access."""
    
    def __init__(self, connection):
        self.connection = connection
        self._cache = {}
    
    def find(self, id):
        """Find entity by ID."""
        if id in self._cache:
            return self._cache[id]
        result = self.connection.query(id)
        self._cache[id] = result
        return result
    
    def save(self, entity):
        """Save entity to storage."""
        self.connection.insert(entity)
        self._cache[entity.id] = entity
        return entity
    
    def delete(self, id):
        """Delete entity by ID."""
        self.connection.delete(id)
        self._cache.pop(id, None)
'''
        metadata = extractor.extract_file_metadata("src/repository.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None
    
    def test_large_file_with_constants_and_functions_is_complex(self, extractor):
        """Large files with both constants AND functions are NOT simple."""
        content = '''
"""Configuration module with helper functions."""

# Constants
API_URL = "https://api.example.com"
TIMEOUT = 30
MAX_RETRIES = 3

def get_config_value(key: str, default=None):
    """Get configuration value by key."""
    config_map = {
        'api_url': API_URL,
        'timeout': TIMEOUT,
        'max_retries': MAX_RETRIES,
    }
    return config_map.get(key, default)

def validate_config():
    """Validate all configuration values."""
    if not API_URL:
        raise ValueError("API_URL must be set")
    if TIMEOUT <= 0:
        raise ValueError("TIMEOUT must be positive")
    return True
'''
        metadata = extractor.extract_file_metadata("src/config.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None


class TestCSharpSimpleFileDetection:
    """Tests for C# simple file detection."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_csharp_interface_with_methods_currently_complex(self, extractor):
        """C# interfaces with method declarations are currently NOT detected as simple.
        
        NOTE: This documents current behavior. Similar to Java, interface method 
        declarations are counted in metadata.methods, preventing type_definitions 
        detection.
        """
        content = '''
namespace MyApp.Contracts
{
    /// <summary>
    /// Interface for user operations.
    /// </summary>
    public interface IUserService
    {
        Task<User> GetUserAsync(int id);
        Task<IEnumerable<User>> GetAllUsersAsync();
        Task<User> CreateUserAsync(UserCreateDto dto);
        Task DeleteUserAsync(int id);
    }
}
'''
        metadata = extractor.extract_file_metadata("src/Contracts/IUserService.cs", content)
        
        assert metadata is not None
        # Currently NOT detected as simple because interface methods are counted
        assert metadata.is_simple_file is False
        # Verify interface IS detected
        assert len(metadata.interfaces) == 1
        assert metadata.interfaces[0]['name'] == 'IUserService'
        # Methods are counted
        assert len(metadata.methods) >= 1
    
    def test_csharp_marker_interface_is_simple(self, extractor):
        """C# marker interfaces (no methods) are detected as simple."""
        content = '''
namespace MyApp.Contracts
{
    /// <summary>
    /// Marker interface for entities.
    /// </summary>
    public interface IEntity
    {
        // Marker interface
    }
}
'''
        metadata = extractor.extract_file_metadata("src/Contracts/IEntity.cs", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "type_definitions"
    
    def test_csharp_class_with_methods_is_complex(self, extractor):
        """C# files with class implementations are NOT simple."""
        content = '''
namespace MyApp.Services
{
    public class UserService : IUserService
    {
        private readonly IUserRepository _repository;
        
        public UserService(IUserRepository repository)
        {
            _repository = repository;
        }
        
        public async Task<User> GetUserAsync(int id)
        {
            return await _repository.FindByIdAsync(id);
        }
    }
}
'''
        metadata = extractor.extract_file_metadata("src/Services/UserService.cs", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is False
        assert metadata.simple_file_reason is None


class TestEdgeCases:
    """Edge case tests for simple file detection."""
    
    @pytest.fixture
    def extractor(self):
        return HeuristicExtractor()
    
    def test_file_at_exactly_30_lines_threshold(self, extractor):
        """Files at exactly 30 lines with no functions are simple."""
        # Create a file with exactly 30 lines
        content = '\n'.join([
            "# Configuration constants",
            "",
            "API_URL = 'https://api.example.com'",
            "TIMEOUT = 30",
            "MAX_RETRIES = 3",
            "BATCH_SIZE = 100",
            "",
            "# Feature flags",
            "ENABLE_CACHE = True",
            "ENABLE_LOGGING = True",
            "",
            "# Database settings",
            "DB_HOST = 'localhost'",
            "DB_PORT = 5432",
            "DB_NAME = 'myapp'",
            "",
            "# Redis settings",
            "REDIS_HOST = 'localhost'",
            "REDIS_PORT = 6379",
            "",
            "# External service URLs",
            "AUTH_SERVICE = 'https://auth.example.com'",
            "PAYMENT_SERVICE = 'https://pay.example.com'",
            "",
            "# Timeouts in seconds",
            "CONNECT_TIMEOUT = 5",
            "READ_TIMEOUT = 30",
            "",
            "# Limits",
            "MAX_CONNECTIONS = 100",
        ])
        
        assert content.count('\n') == 29  # 30 lines
        
        metadata = extractor.extract_file_metadata("src/settings.py", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "constants_only"
    
    def test_file_just_over_30_lines_with_no_functions_is_complex(self, extractor):
        """Files just over 30 lines with no functions are NOT simple (exceed threshold)."""
        # Create a file with 35 lines
        lines = ["# Configuration constants", ""]
        for i in range(33):
            lines.append(f"CONSTANT_{i} = {i}")
        content = '\n'.join(lines)
        
        metadata = extractor.extract_file_metadata("src/many_constants.py", content)
        
        assert metadata is not None
        # Over 30 lines, so not simple even without functions
        assert metadata.is_simple_file is False
    
    def test_typescript_index_tsx_is_simple(self, extractor):
        """TypeScript index.tsx files with only exports are simple."""
        content = '''
// Re-export all components
export { Button } from './Button';
export { Input } from './Input';
export { Modal } from './Modal';
export type { ButtonProps, InputProps, ModalProps } from './types';
'''
        metadata = extractor.extract_file_metadata("src/components/index.tsx", content)
        
        assert metadata is not None
        assert metadata.is_simple_file is True
        assert metadata.simple_file_reason == "barrel_reexport"
    
    def test_javascript_files_not_supported(self, extractor):
        """JavaScript (.js/.jsx) files are not currently supported.
        
        NOTE: Only TypeScript (.ts/.tsx) is supported by the language config.
        JavaScript support would require adding tree_sitter_javascript.
        """
        content = '''
// Component barrel file
export { Header } from './Header';
export { Footer } from './Footer';
export { Sidebar } from './Sidebar';
'''
        # .js files return None - not supported
        metadata_js = extractor.extract_file_metadata("src/layout/index.js", content)
        assert metadata_js is None
        
        # .jsx files also return None - not supported
        metadata_jsx = extractor.extract_file_metadata("src/layout/index.jsx", content)
        assert metadata_jsx is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
