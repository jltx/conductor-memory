"""
Domain classification for code entities.

Determines the semantic domain (class, function, test, private, etc.) 
for code chunks based on AST node information and naming patterns.
"""

import logging
from typing import List, Optional
from tree_sitter import Node

logger = logging.getLogger(__name__)


class DomainDetector:
    """Determines domain tag for a code entity."""
    
    # Test file path patterns
    TEST_PATH_PATTERNS = [
        '/test/', '/tests/', '/spec/', '/specs/',
        '\\test\\', '\\tests\\', '\\spec\\', '\\specs\\',
        '__tests__', '.test.', '.spec.',
        'test_', '_test', 'spec_', '_spec'
    ]
    
    def detect(
        self, 
        node_type: str,
        name: str, 
        file_path: str,
        annotations: Optional[List[str]] = None,
        modifiers: Optional[List[str]] = None,
        language: str = ""
    ) -> str:
        """
        Determine the domain for a code entity.
        
        Args:
            node_type: AST node type (e.g., 'class_definition', 'function_definition')
            name: Entity name (e.g., 'MyClass', 'test_something')
            file_path: Path to the source file
            annotations: List of annotations/decorators (e.g., ['Test', 'Override'])
            modifiers: List of modifiers (e.g., ['private', 'static'])
            language: Programming language name
            
        Returns:
            Domain string: 'class', 'function', 'test', 'private', 'accessor', 
                          'imports', 'interface', 'constant', 'class_summary'
        """
        annotations = annotations or []
        modifiers = modifiers or []
        
        # Priority 1: Test detection (highest priority - overrides others)
        if self._is_test(name, file_path, annotations, language):
            return 'test'
        
        # Priority 2: Imports
        if self._is_import(node_type):
            return 'imports'
        
        # Priority 3: Interface/Protocol
        if self._is_interface(node_type):
            return 'interface'
        
        # Priority 4: Private entities
        if self._is_private(name, modifiers, language):
            return 'private'
        
        # Priority 5: Accessor methods (getters/setters)
        if self._is_accessor(name, node_type):
            return 'accessor'
        
        # Priority 6: Constants
        if self._is_constant(name, node_type, modifiers, language):
            return 'constant'
        
        # Priority 7: Classes
        if self._is_class(node_type):
            return 'class'
        
        # Priority 8: Functions/Methods
        if self._is_function_or_method(node_type):
            return 'function'
        
        # Default: function
        return 'function'
    
    def _is_test(self, name: str, file_path: str, annotations: List[str], language: str) -> bool:
        """Check if entity is a test."""
        # Check file path first (catches all tests in test directories)
        file_path_lower = file_path.lower()
        if any(pattern in file_path_lower for pattern in self.TEST_PATH_PATTERNS):
            return True
        
        # Check annotations (Java, Kotlin, C#, Swift)
        test_annotations = {
            'test', 'fact', 'theory', 'parameterizedtest', 'repeatedtest',
            'testcase', 'testmethod', 'unittest', 'pytest.mark'
        }
        for annotation in annotations:
            if annotation.lower() in test_annotations:
                return True
            # Handle pytest markers like @pytest.mark.parametrize
            if 'pytest.mark' in annotation.lower():
                return True
        
        # Check name patterns
        name_lower = name.lower()
        
        # Common test prefixes/suffixes
        if (name_lower.startswith(('test_', 'test', 'spec_', 'should_', 'when_', 'given_')) or
            name_lower.endswith(('_test', '_spec', 'test', 'spec'))):
            return True
        
        # Language-specific patterns
        if language == 'go' and name.startswith('Test') and len(name) > 4 and name[4].isupper():
            return True  # Go test functions: TestSomething
        
        if language == 'ruby':
            # RSpec patterns
            if name_lower in ['describe', 'it', 'context', 'before', 'after', 'let', 'subject']:
                return True
        
        if language == 'swift' and name_lower.startswith('test'):
            return True
        
        return False
    
    def _is_import(self, node_type: str) -> bool:
        """Check if entity is an import/include statement."""
        import_types = {
            'import_statement', 'import_from_statement',  # Python
            'import_declaration',  # Java, Kotlin, Go
            'using_directive',  # C#
            'preproc_include', 'preproc_import',  # C, Objective-C
            'import_header',  # Kotlin
            'call'  # Ruby (require calls)
        }
        return node_type in import_types
    
    def _is_interface(self, node_type: str) -> bool:
        """Check if entity is an interface/protocol."""
        interface_types = {
            'interface_declaration',  # Java, C#
            'protocol_declaration',  # Swift, Objective-C
            'trait_declaration'  # Some languages
        }
        return node_type in interface_types
    
    def _is_private(self, name: str, modifiers: List[str], language: str) -> bool:
        """Check if entity is private."""
        # Check modifiers first
        if 'private' in [m.lower() for m in modifiers]:
            return True
        
        # Python/Ruby convention: underscore prefix (but not dunder methods)
        if language in ['python', 'ruby']:
            if name.startswith('_') and not name.startswith('__'):
                return True
        
        # Swift: underscore prefix
        if language == 'swift' and name.startswith('_'):
            return True
        
        return False
    
    def _is_accessor(self, name: str, node_type: str) -> bool:
        """Check if entity is an accessor method (getter/setter)."""
        if not self._is_function_or_method(node_type):
            return False
        
        name_lower = name.lower()
        
        # Common accessor patterns
        accessor_prefixes = ['get', 'set', 'is', 'has', 'can', 'should', 'will']
        
        for prefix in accessor_prefixes:
            if name_lower.startswith(prefix) and len(name) > len(prefix):
                # Check if next character is uppercase (camelCase) or underscore
                next_char_idx = len(prefix)
                if next_char_idx < len(name):
                    next_char = name[next_char_idx]
                    if next_char.isupper() or next_char == '_':
                        return True
        
        return False
    
    def _is_constant(self, name: str, node_type: str, modifiers: List[str], language: str) -> bool:
        """Check if entity is a constant."""
        # Check modifiers
        modifier_set = {m.lower() for m in modifiers}
        if 'const' in modifier_set or 'final' in modifier_set or 'static' in modifier_set:
            # Additional check for naming convention
            if name.isupper() or (name.startswith('k') and len(name) > 1 and name[1].isupper()):
                return True
        
        # Naming conventions
        if language == 'python':
            # Python: ALL_CAPS
            if name.isupper() and '_' in name:
                return True
        
        if language in ['java', 'kotlin', 'csharp']:
            # Java/Kotlin/C#: ALL_CAPS or kConstantName
            if name.isupper() or (name.startswith('k') and len(name) > 1 and name[1].isupper()):
                return True
        
        if language == 'go':
            # Go: Exported constants are often ALL_CAPS
            if name.isupper():
                return True
        
        return False
    
    def _is_class(self, node_type: str) -> bool:
        """Check if entity is a class/struct."""
        class_types = {
            'class_definition', 'class_declaration',  # Python, Java, Kotlin, C#, Swift
            'struct_specifier', 'struct_declaration',  # C, Swift
            'type_declaration',  # Go (for structs)
            'class_interface', 'class_implementation',  # Objective-C
            'object_declaration',  # Kotlin
            'record_declaration',  # Java records
            'enum_declaration', 'enum_definition'  # Enums
        }
        return node_type in class_types
    
    def _is_function_or_method(self, node_type: str) -> bool:
        """Check if entity is a function or method."""
        function_types = {
            'function_definition', 'function_declaration',  # Python, Go, C
            'method_definition', 'method_declaration',  # Java, C#, Objective-C
            'function_item',  # Some languages
            'constructor_declaration',  # Java, C#
            'destructor_declaration',  # C++
            'init_declaration',  # Swift
            'deinit_declaration'  # Swift
        }
        return node_type in function_types


# Singleton instance for easy access
domain_detector = DomainDetector()


def detect_domain(
    node_type: str,
    name: str,
    file_path: str,
    annotations: Optional[List[str]] = None,
    modifiers: Optional[List[str]] = None,
    language: str = ""
) -> str:
    """
    Convenience function to detect domain using the singleton detector.
    
    Args:
        node_type: AST node type
        name: Entity name
        file_path: Source file path
        annotations: List of annotations/decorators
        modifiers: List of modifiers
        language: Programming language name
        
    Returns:
        Domain string
    """
    return domain_detector.detect(node_type, name, file_path, annotations, modifiers, language)