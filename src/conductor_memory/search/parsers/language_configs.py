"""
Multi-language AST parser configurations for tree-sitter.

Defines comprehensive configurations for 9 programming languages including:
- Tree-sitter module imports
- File extensions
- AST node types for classes, functions, imports, interfaces
- Tree-sitter query strings
- Test detection patterns
- Method signature extraction logic
"""

from dataclasses import dataclass
from typing import List, Any, Dict, Optional
import re

try:
    import tree_sitter_python as ts_python
    import tree_sitter_java as ts_java
    import tree_sitter_ruby as ts_ruby
    import tree_sitter_go as ts_go
    import tree_sitter_c as ts_c
    import tree_sitter_c_sharp as ts_csharp
    import tree_sitter_kotlin as ts_kotlin
    import tree_sitter_swift as ts_swift
    import tree_sitter_objc as ts_objc
    from tree_sitter import Node
except ImportError as e:
    raise ImportError(f"Missing tree-sitter language modules: {e}")


@dataclass
class LanguageConfig:
    """Configuration for a specific programming language."""
    name: str
    module: Any  # tree-sitter language module
    extensions: List[str]
    
    # AST node types
    class_types: List[str]
    function_types: List[str]
    method_types: List[str]
    import_types: List[str]
    interface_types: List[str]
    
    # Tree-sitter queries
    definition_query: str  # Query to extract all definitions
    
    # Test detection
    test_annotations: List[str]  # @Test, @Fact, etc.
    test_name_patterns: List[str]  # test_, Test, spec_, etc.
    
    def extract_method_signature(self, node: Node, code: bytes) -> str:
        """Extract full method signature with types."""
        return self._extract_signature_impl(node, code)
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Language-specific signature extraction implementation."""
        # Default implementation - extracts basic signature
        return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting annotations/decorators."""
        # Default implementation - override in subclasses for language-specific patterns
        return "(decorator) @annotation\n(annotation) @annotation\n(marker_annotation) @annotation"


class PythonConfig(LanguageConfig):
    """Python language configuration."""
    
    def __init__(self):
        super().__init__(
            name="python",
            module=ts_python,
            extensions=[".py", ".pyw", ".pyi"],
            class_types=["class_definition"],
            function_types=["function_definition"],
            method_types=["function_definition"],  # Methods are functions in class context
            import_types=["import_statement", "import_from_statement"],
            interface_types=["class_definition"],  # Python uses ABC for interfaces
            definition_query="""(class_definition name: (identifier) @class.name) @class.def
(function_definition name: (identifier) @func.name) @func.def
(import_statement) @import.def
(import_from_statement) @import.def""",
            test_annotations=["@pytest.mark.*", "@unittest.*", "@mock.*"],
            test_name_patterns=["test_", "Test"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract Python method signature with type hints."""
        try:
            # Get the full function definition
            signature_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract just the signature line (up to the colon)
            lines = signature_text.split('\n')
            signature_line = lines[0]
            
            # Handle multi-line signatures
            if ':' not in signature_line:
                for i, line in enumerate(lines[1:], 1):
                    signature_line += ' ' + line.strip()
                    if ':' in line:
                        break
            
            # Extract signature up to colon
            if ':' in signature_line:
                signature = signature_line.split(':')[0].strip()
            else:
                signature = signature_line.strip()
                
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting Python decorators."""
        return "(decorator) @annotation\n(decorator_list) @annotation"


class JavaConfig(LanguageConfig):
    """Java language configuration."""
    
    def __init__(self):
        super().__init__(
            name="java",
            module=ts_java,
            extensions=[".java"],
            class_types=["class_declaration", "record_declaration", "enum_declaration"],
            function_types=["method_declaration", "constructor_declaration"],
            method_types=["method_declaration"],
            import_types=["import_declaration"],
            interface_types=["interface_declaration", "annotation_type_declaration"],
            definition_query="""(class_declaration name: (identifier) @class.name) @class.def
(interface_declaration name: (identifier) @interface.name) @interface.def
(method_declaration name: (identifier) @method.name) @method.def
(import_declaration) @import.def""",
            test_annotations=["@Test", "@ParameterizedTest", "@RepeatedTest", "@TestFactory", "@BeforeEach", "@AfterEach"],
            test_name_patterns=["test", "Test", "should"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract Java method signature with modifiers and throws clause."""
        try:
            # Get the method declaration text
            method_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract signature (everything before the opening brace or semicolon)
            lines = method_text.split('\n')
            signature_lines = []
            
            for line in lines:
                signature_lines.append(line.strip())
                if '{' in line or ';' in line:
                    break
            
            signature = ' '.join(signature_lines)
            
            # Remove the opening brace and everything after
            if '{' in signature:
                signature = signature.split('{')[0].strip()
            if ';' in signature:
                signature = signature.split(';')[0].strip()
                
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting Java annotations."""
        return "(annotation) @annotation\n(marker_annotation) @annotation\n(modifiers) @annotation"


class RubyConfig(LanguageConfig):
    """Ruby language configuration."""
    
    def __init__(self):
        super().__init__(
            name="ruby",
            module=ts_ruby,
            extensions=[".rb", ".rbw", ".rake", ".gemspec"],
            class_types=["class", "singleton_class"],
            function_types=["method", "singleton_method"],
            method_types=["method", "singleton_method"],
            import_types=["call"],  # require, load statements
            interface_types=["module"],  # Ruby uses modules for interfaces
            definition_query="""(class name: (constant) @class.name) @class.def
(module name: (constant) @interface.name) @interface.def
(method name: (identifier) @method.name) @method.def
(singleton_method name: (identifier) @method.name) @method.def
(call method: (identifier) @import.name) @import.def""",
            test_annotations=["describe", "context", "it", "before", "after"],
            test_name_patterns=["test_", "spec_", "_test", "_spec"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract Ruby method signature."""
        try:
            method_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract first line which contains the method signature
            first_line = method_text.split('\n')[0].strip()
            
            # Remove 'end' if it's a one-liner
            if first_line.endswith(' end'):
                first_line = first_line[:-4].strip()
                
            return first_line
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting Ruby annotations (limited)."""
        return "(comment) @annotation"  # Ruby doesn't have formal annotations, use comments


class GoConfig(LanguageConfig):
    """Go language configuration."""
    
    def __init__(self):
        super().__init__(
            name="go",
            module=ts_go,
            extensions=[".go"],
            class_types=["type_declaration"],  # Go uses structs instead of classes
            function_types=["function_declaration", "method_declaration"],
            method_types=["method_declaration"],
            import_types=["import_declaration", "import_spec"],
            interface_types=["interface_type"],
            definition_query="""(type_declaration (type_spec (type_identifier) @class.name (struct_type))) @class.def
(type_declaration (type_spec (type_identifier) @interface.name (interface_type))) @interface.def
(function_declaration (identifier) @func.name) @func.def
(method_declaration (field_identifier) @method.name) @method.def
(import_declaration) @import.def
(import_spec) @import.def""",
            test_annotations=[],  # Go doesn't use annotations
            test_name_patterns=["Test", "Benchmark", "Example"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract Go function/method signature with receiver."""
        try:
            method_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract signature (everything before the opening brace)
            lines = method_text.split('\n')
            signature_lines = []
            
            for line in lines:
                signature_lines.append(line.strip())
                if '{' in line:
                    # Split at the brace
                    parts = line.split('{')
                    signature_lines[-1] = parts[0].strip()
                    break
            
            signature = ' '.join(signature_lines)
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting Go annotations (build tags, comments)."""
        return "(comment) @annotation"  # Go uses build tags and comments for annotations


class CConfig(LanguageConfig):
    """C language configuration."""
    
    def __init__(self):
        super().__init__(
            name="c",
            module=ts_c,
            extensions=[".c", ".h"],
            class_types=["struct_specifier", "union_specifier"],
            function_types=["function_definition", "function_declarator"],
            method_types=["function_definition"],  # C doesn't have methods, just functions
            import_types=["preproc_include"],
            interface_types=["struct_specifier"],  # C uses structs for interfaces
            definition_query="""(struct_specifier) @struct.def
(function_definition) @func.def
(preproc_include) @import.def""",
            test_annotations=[],  # C doesn't have annotations
            test_name_patterns=["test_", "Test", "check_"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract C function signature."""
        try:
            func_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract signature (everything before the opening brace)
            lines = func_text.split('\n')
            signature_lines = []
            
            for line in lines:
                signature_lines.append(line.strip())
                if '{' in line:
                    parts = line.split('{')
                    signature_lines[-1] = parts[0].strip()
                    break
            
            signature = ' '.join(signature_lines)
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting C annotations (limited)."""
        return "(comment) @annotation"  # C uses preprocessor directives and comments


class CSharpConfig(LanguageConfig):
    """C# language configuration."""
    
    def __init__(self):
        super().__init__(
            name="csharp",
            module=ts_csharp,
            extensions=[".cs"],
            class_types=["class_declaration", "record_declaration", "struct_declaration"],
            function_types=["method_declaration", "constructor_declaration", "destructor_declaration"],
            method_types=["method_declaration"],
            import_types=["using_directive"],
            interface_types=["interface_declaration"],
            definition_query="""(class_declaration name: (identifier) @class.name) @class.def
(interface_declaration name: (identifier) @interface.name) @interface.def
(method_declaration name: (identifier) @method.name) @method.def
(using_directive) @import.def""",
            test_annotations=["[Test]", "[Fact]", "[Theory]", "[TestMethod]", "[TestCase]"],
            test_name_patterns=["Test", "_Test", "Should", "_Should"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract C# method signature with modifiers."""
        try:
            method_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract signature (everything before the opening brace or semicolon)
            lines = method_text.split('\n')
            signature_lines = []
            
            for line in lines:
                signature_lines.append(line.strip())
                if '{' in line or ';' in line:
                    if '{' in line:
                        parts = line.split('{')
                        signature_lines[-1] = parts[0].strip()
                    elif ';' in line:
                        parts = line.split(';')
                        signature_lines[-1] = parts[0].strip()
                    break
            
            signature = ' '.join(signature_lines)
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting C# attributes."""
        return "(attribute) @annotation\n(attribute_list) @annotation"


class KotlinConfig(LanguageConfig):
    """Kotlin language configuration."""
    
    def __init__(self):
        super().__init__(
            name="kotlin",
            module=ts_kotlin,
            extensions=[".kt", ".kts"],
            class_types=["class_declaration", "object_declaration"],
            function_types=["function_declaration"],
            method_types=["function_declaration"],  # Methods are functions in Kotlin
            import_types=["import"],
            interface_types=["class_declaration"],  # Interfaces use class_declaration with interface keyword
            definition_query="""(class_declaration name: (identifier) @class.name) @class.def
(object_declaration name: (identifier) @obj.name) @obj.def
(function_declaration name: (identifier) @func.name) @func.def
(import) @import.def""",
            test_annotations=["@Test", "@ParameterizedTest", "@RepeatedTest", "@BeforeEach", "@AfterEach"],
            test_name_patterns=["test", "Test", "should"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract Kotlin function signature with modifiers and return type."""
        try:
            func_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract signature (everything before the opening brace or equals sign)
            lines = func_text.split('\n')
            signature_lines = []
            
            for line in lines:
                signature_lines.append(line.strip())
                if '{' in line or '=' in line:
                    if '{' in line:
                        parts = line.split('{')
                        signature_lines[-1] = parts[0].strip()
                    elif '=' in line and 'fun' in ' '.join(signature_lines):
                        parts = line.split('=')
                        signature_lines[-1] = parts[0].strip()
                    break
            
            signature = ' '.join(signature_lines)
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting Kotlin annotations."""
        return "(annotation) @annotation\n(file_annotation) @annotation"


class SwiftConfig(LanguageConfig):
    """Swift language configuration."""
    
    def __init__(self):
        super().__init__(
            name="swift",
            module=ts_swift,
            extensions=[".swift"],
            class_types=["class_declaration", "struct_declaration"],
            function_types=["function_declaration"],
            method_types=["function_declaration"],  # Methods are functions in Swift
            import_types=["import_declaration"],
            interface_types=["protocol_declaration"],
            definition_query="""(class_declaration (type_identifier) @class.name) @class.def
(protocol_declaration (type_identifier) @interface.name) @interface.def
(function_declaration (simple_identifier) @func.name) @func.def
(import_declaration) @import.def""",
            test_annotations=["@Test"],  # Swift Testing framework
            test_name_patterns=["test", "Test"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract Swift function signature."""
        try:
            func_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract signature (everything before the opening brace)
            lines = func_text.split('\n')
            signature_lines = []
            
            for line in lines:
                signature_lines.append(line.strip())
                if '{' in line:
                    parts = line.split('{')
                    signature_lines[-1] = parts[0].strip()
                    break
            
            signature = ' '.join(signature_lines)
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting Swift attributes."""
        return "(attribute) @annotation\n(availability_attribute) @annotation"


class ObjectiveCConfig(LanguageConfig):
    """Objective-C language configuration."""
    
    def __init__(self):
        super().__init__(
            name="objc",
            module=ts_objc,
            extensions=[".m", ".mm", ".h"],
            class_types=["class_interface", "class_implementation"],
            function_types=["method_declaration", "method_definition", "function_definition"],
            method_types=["method_declaration", "method_definition"],
            import_types=["preproc_include", "preproc_import"],
            interface_types=["class_interface", "protocol_declaration"],
            definition_query="""(class_interface) @class.def
(class_implementation) @class.def
(method_declaration) @method.def
(method_definition) @method.def
(preproc_include) @import.def""",
            test_annotations=["@Test"],  # XCTest framework
            test_name_patterns=["test", "Test"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract Objective-C method signature."""
        try:
            method_text = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract signature (everything before the opening brace)
            lines = method_text.split('\n')
            signature_lines = []
            
            for line in lines:
                signature_lines.append(line.strip())
                if '{' in line:
                    parts = line.split('{')
                    signature_lines[-1] = parts[0].strip()
                    break
            
            signature = ' '.join(signature_lines)
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting Objective-C attributes."""
        return "(attribute) @annotation\n(property_attribute) @annotation"


# Language configuration registry
LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "python": PythonConfig(),
    "java": JavaConfig(),
    "ruby": RubyConfig(),
    "go": GoConfig(),
    "c": CConfig(),
    "csharp": CSharpConfig(),
    "kotlin": KotlinConfig(),
    "swift": SwiftConfig(),
    "objc": ObjectiveCConfig(),
}

# Extension to language mapping
EXTENSION_TO_LANGUAGE: Dict[str, str] = {}
for lang_name, config in LANGUAGE_CONFIGS.items():
    for ext in config.extensions:
        EXTENSION_TO_LANGUAGE[ext] = lang_name


def get_language_config(language: str) -> Optional[LanguageConfig]:
    """Get language configuration by name."""
    return LANGUAGE_CONFIGS.get(language.lower())


def get_language_from_extension(file_path: str) -> Optional[str]:
    """Get language name from file extension."""
    for ext in EXTENSION_TO_LANGUAGE:
        if file_path.endswith(ext):
            return EXTENSION_TO_LANGUAGE[ext]
    return None


def get_supported_languages() -> List[str]:
    """Get list of all supported language names."""
    return list(LANGUAGE_CONFIGS.keys())


def is_test_file(file_path: str, content: str = "") -> bool:
    """Detect if a file is likely a test file based on patterns."""
    lang = get_language_from_extension(file_path)
    if not lang:
        return False
    
    config = get_language_config(lang)
    if not config:
        return False
    
    # Check file name patterns
    file_name = file_path.lower()
    test_indicators = [
        "test", "spec", "_test", "_spec", 
        "tests", "specs", "testing"
    ]
    
    for indicator in test_indicators:
        if indicator in file_name:
            return True
    
    # Check content for test patterns if provided
    if content:
        content_lower = content.lower()
        
        # Check for test annotations
        for annotation in config.test_annotations:
            # Remove regex patterns for simple string matching
            simple_annotation = annotation.replace(".*", "").replace("@", "").lower()
            if simple_annotation in content_lower:
                return True
        
        # Check for test name patterns in function/method names
        for pattern in config.test_name_patterns:
            if pattern.lower() in content_lower:
                return True
    
    return False