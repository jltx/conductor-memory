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
    import tree_sitter_typescript as ts_typescript
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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting method body implementation details.
        
        Captures:
        - Method/function calls (internal and external)
        - Subscript/index access (e.g., df[key], series.iloc[idx])
        - Attribute access (e.g., self.attr, obj.field)
        - Assignments (writes to attributes)
        - Structural signals: loops (for/while), conditionals (if), try-except
        
        Returns:
            Tree-sitter query string for implementation pattern extraction.
            Override in subclasses for language-specific patterns.
        """
        # Default implementation - most languages don't have queries defined yet
        # Each language subclass should override with appropriate patterns
        return ""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting Python implementation details.
        
        Captures method calls, subscripts, attributes, assignments, and structural patterns
        for verification queries like "does method X use pattern Y?".
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method calls on objects: obj.method(), self.method(), receiver.method()
(call
  function: (attribute
    object: (_) @call.receiver
    attribute: (identifier) @call.method_name)) @call.method

; Simple function calls: func(), module_func()
(call
  function: (identifier) @call.function_name) @call.function

; ============================================
; SUBSCRIPT/INDEX ACCESS
; ============================================

; Simple subscript: df[key], dict[key], list[idx]
(subscript
  value: (identifier) @subscript.target_simple
  subscript: (_) @subscript.index) @subscript.simple

; Attribute subscript: self.attr[key], df.iloc[idx], series.loc[mask]
(subscript
  value: (attribute
    object: (_) @subscript.attr_object
    attribute: (identifier) @subscript.attr_name)
  subscript: (_) @subscript.attr_index) @subscript.attribute

; ============================================
; ATTRIBUTE ACCESS (reads)
; ============================================

; Attribute access: self.attr, obj.field, module.constant
(attribute
  object: (_) @access.object
  attribute: (identifier) @access.attr_name) @access.attribute

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Attribute assignment: self.attr = value, obj.field = x
(assignment
  left: (attribute
    object: (_) @write.target_object
    attribute: (identifier) @write.attr_name)
  right: (_) @write.value) @write.attribute

; Simple assignment: x = value
(assignment
  left: (identifier) @write.simple_target
  right: (_) @write.simple_value) @write.simple

; Augmented assignment: self.attr += value, x -= 1
(augmented_assignment
  left: (attribute
    object: (_) @write.aug_object
    attribute: (identifier) @write.aug_attr)
  right: (_) @write.aug_value) @write.augmented

; ============================================
; STRUCTURAL SIGNALS
; ============================================

; For loops
(for_statement) @structure.for_loop

; While loops  
(while_statement) @structure.while_loop

; If statements (conditionals)
(if_statement) @structure.conditional

; Try-except blocks
(try_statement) @structure.try_except

; With statements (context managers)
(with_statement) @structure.with_block

; List comprehensions
(list_comprehension) @structure.list_comp

; Dict comprehensions
(dictionary_comprehension) @structure.dict_comp

; Generator expressions
(generator_expression) @structure.generator

; Lambda expressions
(lambda) @structure.lambda

; Await expressions
(await) @structure.await

; Yield expressions
(yield) @structure.yield

; Raise statements
(raise_statement) @structure.raise

; Assert statements  
(assert_statement) @structure.assert

; Return statements
(return_statement) @structure.return
"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting Java implementation details.
        
        Captures method calls, field access, array access, assignments, and structural patterns
        for verification queries like "does method X use pattern Y?".
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method invocation on object: obj.method(), this.method(), receiver.method()
(method_invocation
  object: (_) @call.receiver
  name: (identifier) @call.method_name
  arguments: (argument_list)) @call.method

; Static method call: ClassName.staticMethod()
(method_invocation
  object: (identifier) @call.static_class
  name: (identifier) @call.static_method_name
  arguments: (argument_list)) @call.static

; Simple method call (no receiver): method()
(method_invocation
  name: (identifier) @call.function_name
  arguments: (argument_list)
  !object) @call.function

; Method reference: Class::method (Java 8+) - captures class and method names
(method_reference
  .
  (identifier) @call.ref_receiver
  (identifier) @call.ref_method) @call.method_ref

; Method reference on this: this::method
(method_reference
  (this) @call.ref_this
  (identifier) @call.ref_method_on_this) @call.method_ref_this

; Method reference on super: super::method
(method_reference
  (super) @call.ref_super
  (identifier) @call.ref_method_on_super) @call.method_ref_super

; ============================================
; FIELD/ATTRIBUTE ACCESS
; ============================================

; Field access: this.field, object.field
(field_access
  object: (_) @access.object
  field: (identifier) @access.field_name) @access.field

; ============================================
; ARRAY/SUBSCRIPT ACCESS  
; ============================================

; Array access: array[index], list.get(i) style captured under method calls
(array_access
  array: (identifier) @subscript.target_simple
  index: (_) @subscript.index) @subscript.simple

; Array access on field: this.array[index], obj.array[index]
(array_access
  array: (field_access
    object: (_) @subscript.field_object
    field: (identifier) @subscript.field_name)
  index: (_) @subscript.field_index) @subscript.field

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Field assignment: this.field = value, obj.field = value
(assignment_expression
  left: (field_access
    object: (_) @write.target_object
    field: (identifier) @write.field_name)
  right: (_) @write.value) @write.field

; Simple variable assignment: x = value
(assignment_expression
  left: (identifier) @write.simple_target
  right: (_) @write.simple_value) @write.simple

; Array element assignment: array[i] = value
(assignment_expression
  left: (array_access
    array: (_) @write.array_target
    index: (_) @write.array_index)
  right: (_) @write.array_value) @write.array

; ============================================
; STRUCTURAL SIGNALS
; ============================================

; Standard for loop: for (int i = 0; i < n; i++)
(for_statement) @structure.for_loop

; Enhanced for loop (foreach): for (Type item : collection)
(enhanced_for_statement) @structure.enhanced_for

; While loop
(while_statement) @structure.while_loop

; Do-while loop
(do_statement) @structure.do_while

; If statement (conditional)
(if_statement) @structure.conditional

; Switch statement
(switch_expression) @structure.switch

; Try-catch block
(try_statement) @structure.try_catch

; Try-with-resources
(try_with_resources_statement) @structure.try_with_resources

; Synchronized block
(synchronized_statement) @structure.synchronized

; Throw statement
(throw_statement) @structure.throw

; Return statement
(return_statement) @structure.return

; Lambda expression (Java 8+)
(lambda_expression) @structure.lambda

; Object creation: new ClassName()
(object_creation_expression
  type: (_) @creation.type
  arguments: (argument_list)) @structure.new_instance

; Assert statement
(assert_statement) @structure.assert
"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting Ruby implementation details.
        
        Captures method calls, instance variable access, subscripts, assignments, and structural patterns
        for verification queries like "does method X use pattern Y?".
        
        Ruby-specific considerations:
        - Method calls use `call` node with `receiver`, `method`, and optional `operator` (`.`, `&.`, `::`)
        - Instance variables are `@name` via `instance_variable` node
        - Class variables are `@@name` via `class_variable` node
        - Global variables are `$name` via `global_variable` node
        - Index access uses `element_reference` (not subscript)
        - Ruby has `unless` (negated if) and `until` (negated while)
        - Blocks come in two forms: `do...end` (`do_block`) and `{...}` (`block`)
        - Ruby's `case/when` is pattern matching (not switch-like)
        - `begin/rescue/ensure` is Ruby's try-catch-finally
        - Modifier forms exist: `return x if condition`, `x while condition`
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method call with receiver: obj.method, self.method, receiver.method()
; Includes safe navigation operator: obj&.method
(call
  receiver: (_) @call.receiver
  method: (identifier) @call.method_name) @call.method

; Method call with constant receiver: Module::method, Class.new
(call
  receiver: (constant) @call.const_receiver
  method: (identifier) @call.const_method_name) @call.const_method

; Scope resolution call: Module::Class, Namespace::Constant
(call
  receiver: (scope_resolution) @call.scope_receiver
  method: (identifier) @call.scope_method_name) @call.scope_method

; Safe navigation operator: obj&.method (captured by operator field)
(call
  receiver: (_) @call.safe_receiver
  operator: "&."
  method: (identifier) @call.safe_method_name) @call.safe_method

; Simple method call (no receiver): method(), puts, require
(call
  method: (identifier) @call.function_name
  !receiver) @call.function

; Method call with super keyword
(call
  receiver: (super) @call.super_receiver
  method: (identifier) @call.super_method_name) @call.super_method

; ============================================
; INSTANCE/CLASS/GLOBAL VARIABLE ACCESS (reads)
; ============================================

; Instance variable access: @name, @variable
(instance_variable) @access.instance_var

; Class variable access: @@name
(class_variable) @access.class_var

; Global variable access: $name, $stdout
(global_variable) @access.global_var

; Self keyword (explicit receiver)
(self) @access.self

; ============================================
; INDEX/SUBSCRIPT ACCESS (element_reference)
; ============================================

; Simple index access: array[index], hash[key]
(element_reference
  object: (identifier) @subscript.target_simple) @subscript.simple

; Index access on instance variable: @array[index], @hash[key]
(element_reference
  object: (instance_variable) @subscript.ivar_target) @subscript.instance_var

; Index access on method result: obj.method[index]
(element_reference
  object: (call
    receiver: (_) @subscript.call_receiver
    method: (identifier) @subscript.call_method)) @subscript.method_result

; Chained index access: hash[key1][key2]
(element_reference
  object: (element_reference) @subscript.chained_target) @subscript.chained

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Instance variable assignment: @var = value
(assignment
  left: (instance_variable) @write.ivar_target
  right: (_) @write.ivar_value) @write.instance_var

; Class variable assignment: @@var = value
(assignment
  left: (class_variable) @write.cvar_target
  right: (_) @write.cvar_value) @write.class_var

; Global variable assignment: $var = value
(assignment
  left: (global_variable) @write.gvar_target
  right: (_) @write.gvar_value) @write.global_var

; Simple local variable assignment: x = value
(assignment
  left: (identifier) @write.simple_target
  right: (_) @write.simple_value) @write.simple

; Element assignment: array[i] = value, hash[key] = value
(assignment
  left: (element_reference
    object: (_) @write.elem_target) @write.elem_ref
  right: (_) @write.elem_value) @write.element

; Attribute writer assignment via setter: self.attr = value, obj.prop = value
(assignment
  left: (call
    receiver: (_) @write.attr_receiver
    method: (identifier) @write.attr_method)
  right: (_) @write.attr_value) @write.attribute

; Operator assignment on instance variable: @count += 1, @value ||= default
(operator_assignment
  left: (instance_variable) @write.op_ivar_target
  right: (_) @write.op_ivar_value) @write.op_instance_var

; Operator assignment on local variable: x += 1, y ||= []
(operator_assignment
  left: (identifier) @write.op_target
  right: (_) @write.op_value) @write.operator

; Operator assignment on element: array[i] += 1
(operator_assignment
  left: (element_reference) @write.op_elem_target
  right: (_) @write.op_elem_value) @write.op_element

; Multiple assignment (parallel): a, b = 1, 2
(assignment
  left: (left_assignment_list) @write.multi_targets
  right: (_) @write.multi_values) @write.multiple

; ============================================
; STRUCTURAL SIGNALS - LOOPS
; ============================================

; For loop: for item in collection do ... end
(for) @structure.for_loop

; While loop: while condition do ... end
(while) @structure.while_loop

; Until loop (negated while): until condition do ... end
(until) @structure.until_loop

; While modifier: action while condition
(while_modifier) @structure.while_modifier

; Until modifier: action until condition
(until_modifier) @structure.until_modifier

; ============================================
; STRUCTURAL SIGNALS - CONDITIONALS
; ============================================

; If statement: if condition then ... elsif ... else ... end
(if) @structure.conditional

; Unless statement (negated if): unless condition then ... end
(unless) @structure.unless

; If modifier: action if condition
(if_modifier) @structure.if_modifier

; Unless modifier: action unless condition
(unless_modifier) @structure.unless_modifier

; Ternary conditional: condition ? true_val : false_val
(conditional) @structure.ternary

; Case/when statement (pattern matching)
(case) @structure.case

; Case match (Ruby 3+ pattern matching): case expr in pattern
(case_match) @structure.case_match

; ============================================
; STRUCTURAL SIGNALS - EXCEPTION HANDLING
; ============================================

; Begin/rescue/ensure block (try-catch-finally)
(begin) @structure.begin

; Rescue modifier: expr rescue default
(rescue_modifier) @structure.rescue_modifier

; ============================================
; STRUCTURAL SIGNALS - BLOCKS & ITERATORS
; ============================================

; Do block: method do |args| ... end
(do_block) @structure.do_block

; Brace block: method { |args| ... }
(block) @structure.block

; Lambda: -> { }, ->(args) { }
(lambda) @structure.lambda

; ============================================
; STRUCTURAL SIGNALS - CONTROL FLOW
; ============================================

; Return statement: return value
(return) @structure.return

; Yield (to block): yield value
(yield) @structure.yield

; Break (exit loop/block): break value
(break) @structure.break

; Next (continue to next iteration): next value  
(next) @structure.next

; Redo (restart current iteration)
(redo) @structure.redo

; Retry (restart begin block)
(retry) @structure.retry

; ============================================
; STRUCTURAL SIGNALS - DEFINITIONS
; ============================================

; Module definition
(module) @structure.module

; Class definition
(class) @structure.class

; Singleton class: class << self
(singleton_class) @structure.singleton_class

; Method definition
(method) @structure.method

; Singleton method: def self.method, def obj.method
(singleton_method) @structure.singleton_method
"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting Go implementation details.
        
        Captures method calls, field access, index access, assignments, and structural patterns
        for verification queries like "does function X use pattern Y?".
        
        Go-specific considerations:
        - selector_expression is used for both field access AND method calls (before call_expression)
        - Go only has 'for' loops (no while/do-while) - range is a clause within for
        - Goroutines (go statements) and defer are Go-specific concurrency patterns
        - Channel operations (send/receive) are first-class constructs
        - Short variable declarations (:=) are distinct from assignments
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method call on receiver: receiver.Method(args)
; In Go, call_expression wraps selector_expression for method calls
(call_expression
  function: (selector_expression
    operand: (_) @call.receiver
    field: (field_identifier) @call.method_name)
  arguments: (argument_list)) @call.method

; Package-qualified function call: pkg.Function(args)
(call_expression
  function: (selector_expression
    operand: (identifier) @call.package
    field: (field_identifier) @call.qualified_func)
  arguments: (argument_list)) @call.qualified

; Simple function call: Function(args)
(call_expression
  function: (identifier) @call.function_name
  arguments: (argument_list)) @call.function

; Type conversion: Type(value) - looks like call but is conversion
(type_conversion_expression
  type: (_) @call.conversion_type
  operand: (_) @call.conversion_operand) @call.type_conversion

; ============================================
; FIELD/ATTRIBUTE ACCESS (reads)
; ============================================

; Field access: struct.field, pointer.field, receiver.field
; Note: This also matches method access before call, which is fine for analysis
(selector_expression
  operand: (_) @access.object
  field: (field_identifier) @access.field_name) @access.field

; ============================================
; INDEX/SUBSCRIPT ACCESS
; ============================================

; Simple index access: slice[idx], array[idx], map[key]
(index_expression
  operand: (identifier) @subscript.target_simple
  index: (_) @subscript.index) @subscript.simple

; Chained index access: obj.field[idx], pkg.Var[key]
(index_expression
  operand: (selector_expression
    operand: (_) @subscript.selector_object
    field: (field_identifier) @subscript.selector_field)
  index: (_) @subscript.selector_index) @subscript.selector

; Slice expression: slice[start:end], slice[start:end:cap]
(slice_expression
  operand: (identifier) @subscript.slice_target) @subscript.slice

; Slice expression on field: obj.slice[start:end]
(slice_expression
  operand: (selector_expression
    operand: (_) @subscript.slice_object
    field: (field_identifier) @subscript.slice_field)) @subscript.slice_selector

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Field assignment: struct.field = value
(assignment_statement
  left: (expression_list
    (selector_expression
      operand: (_) @write.target_object
      field: (field_identifier) @write.field_name))
  right: (expression_list)) @write.field

; Simple variable assignment: x = value
(assignment_statement
  left: (expression_list
    (identifier) @write.simple_target)
  right: (expression_list)) @write.simple

; Index assignment: slice[i] = value, map[key] = value
(assignment_statement
  left: (expression_list
    (index_expression
      operand: (_) @write.index_target
      index: (_) @write.index_key))
  right: (expression_list)) @write.index

; Short variable declaration: x := value (Go-specific)
(short_var_declaration
  left: (expression_list) @write.short_decl_targets
  right: (expression_list) @write.short_decl_values) @write.short_decl

; Increment/decrement: i++, i--
(inc_statement) @write.increment
(dec_statement) @write.decrement

; ============================================
; STRUCTURAL SIGNALS
; ============================================

; For loop (Go's only loop construct)
(for_statement) @structure.for_loop

; Range clause within for (for key, value := range collection)
(range_clause
  left: (expression_list) @structure.range_vars
  right: (_) @structure.range_target) @structure.range

; If statement (conditional)
(if_statement) @structure.conditional

; Expression switch (switch value { case ... })
(expression_switch_statement) @structure.switch

; Type switch (switch x.(type) { case ... })
(type_switch_statement) @structure.type_switch

; Select statement (channel multiplexing)
(select_statement) @structure.select

; Defer statement (deferred execution)
(defer_statement) @structure.defer

; Go statement (goroutine launch)
(go_statement) @structure.goroutine

; Return statement
(return_statement) @structure.return

; Send statement (channel send: ch <- value)
(send_statement
  channel: (_) @structure.send_channel
  value: (_) @structure.send_value) @structure.channel_send

; Receive expression (channel receive: <-ch)
; Note: receive is often in assignment context
(unary_expression
  operator: "<-"
  operand: (_) @structure.receive_channel) @structure.channel_receive

; Function literal (anonymous function/closure)
(func_literal) @structure.func_literal

; Composite literal (struct/slice/map initialization)
(composite_literal
  type: (_) @structure.literal_type
  body: (literal_value)) @structure.composite_literal

; Type assertion: x.(Type)
(type_assertion_expression
  operand: (_) @structure.assertion_operand
  type: (_) @structure.assertion_type) @structure.type_assertion

; Labeled statement (for break/continue labels)
(labeled_statement
  label: (label_name) @structure.label) @structure.labeled

; Panic via function call (convention, not syntax)
; Captured via call_expression matching "panic"
"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting C (and basic Objective-C) implementation details.
        
        Captures function calls, field access, subscripts, assignments, and structural patterns
        for verification queries like "does function X use pattern Y?".
        
        C-specific considerations:
        - Uses `call_expression` for all function calls
        - `field_expression` handles both dot (struct.field) and arrow (ptr->field) access
        - The operator field distinguishes `.` from `->`
        - No methods per se, but function pointers in structs act similarly
        - `goto_statement` and `labeled_statement` for control flow
        - `do_statement` is do-while (only loop with condition at end)
        
        Objective-C extensions (when using tree-sitter-objc separately):
        - `message_expression` for [obj message:arg] syntax
        - This query includes patterns that work for both C and ObjC file parsing
        """
        return """
; ============================================
; FUNCTION CALLS
; ============================================

; Simple function call: func(), printf(), malloc()
(call_expression
  function: (identifier) @call.function_name
  arguments: (argument_list)) @call.function

; Method-style call via struct: obj.method() - function pointer in struct
(call_expression
  function: (field_expression
    argument: (_) @call.receiver
    operator: "."
    field: (field_identifier) @call.method_name)
  arguments: (argument_list)) @call.method

; Method-style call via pointer: ptr->method() - function pointer via arrow
(call_expression
  function: (field_expression
    argument: (_) @call.receiver_ptr
    operator: "->"
    field: (field_identifier) @call.method_ptr_name)
  arguments: (argument_list)) @call.method_ptr

; Nested/chained call: a.b.c() or a->b->c()
(call_expression
  function: (field_expression
    argument: (field_expression) @call.chained_receiver
    field: (field_identifier) @call.chained_method)
  arguments: (argument_list)) @call.chained

; ============================================
; FIELD/MEMBER ACCESS (reads)
; ============================================

; Dot access: struct.field, obj.member
(field_expression
  argument: (_) @access.object
  operator: "."
  field: (field_identifier) @access.field_name) @access.field

; Arrow access: ptr->field, node->next
(field_expression
  argument: (_) @access.ptr_object
  operator: "->"
  field: (field_identifier) @access.ptr_field_name) @access.ptr_field

; ============================================
; SUBSCRIPT/INDEX ACCESS
; ============================================

; Simple subscript: array[index], buffer[i]
(subscript_expression
  argument: (identifier) @subscript.target_simple
  index: (_) @subscript.index) @subscript.simple

; Subscript on field: struct.array[i], ptr->buffer[j]
(subscript_expression
  argument: (field_expression
    argument: (_) @subscript.field_object
    field: (field_identifier) @subscript.field_name)
  index: (_) @subscript.field_index) @subscript.field

; Chained subscript: array[i][j] (2D arrays)
(subscript_expression
  argument: (subscript_expression) @subscript.chained_target
  index: (_) @subscript.chained_index) @subscript.chained

; Pointer arithmetic subscript: *(ptr + i) style is parenthesized_expression
; but array[i] style is captured above

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Field assignment via dot: struct.field = value
(assignment_expression
  left: (field_expression
    argument: (_) @write.target_object
    operator: "."
    field: (field_identifier) @write.field_name)
  right: (_) @write.value) @write.field

; Field assignment via arrow: ptr->field = value
(assignment_expression
  left: (field_expression
    argument: (_) @write.ptr_target
    operator: "->"
    field: (field_identifier) @write.ptr_field_name)
  right: (_) @write.ptr_value) @write.ptr_field

; Simple variable assignment: x = value
(assignment_expression
  left: (identifier) @write.simple_target
  right: (_) @write.simple_value) @write.simple

; Subscript assignment: array[i] = value
(assignment_expression
  left: (subscript_expression
    argument: (_) @write.array_target
    index: (_) @write.array_index)
  right: (_) @write.array_value) @write.subscript

; Pointer dereference assignment: *ptr = value
(assignment_expression
  left: (pointer_expression
    argument: (_) @write.deref_target)
  right: (_) @write.deref_value) @write.pointer_deref

; Compound/augmented assignment: x += 1, count -= n
(assignment_expression
  left: (identifier) @write.compound_target
  operator: (_) @write.compound_operator
  right: (_) @write.compound_value) @write.compound

; ============================================
; STRUCTURAL SIGNALS - LOOPS
; ============================================

; For loop: for (init; cond; update) { }
(for_statement) @structure.for_loop

; While loop: while (condition) { }
(while_statement) @structure.while_loop

; Do-while loop: do { } while (condition);
(do_statement) @structure.do_while

; ============================================
; STRUCTURAL SIGNALS - CONDITIONALS
; ============================================

; If statement: if (condition) { } else { }
(if_statement) @structure.conditional

; Switch statement: switch (value) { case ...: }
(switch_statement) @structure.switch

; Case label within switch
(case_statement) @structure.case

; Ternary/conditional expression: condition ? a : b
(conditional_expression) @structure.ternary

; ============================================
; STRUCTURAL SIGNALS - CONTROL FLOW
; ============================================

; Goto statement: goto label;
(goto_statement) @structure.goto

; Labeled statement: label: statement
(labeled_statement
  label: (statement_identifier) @structure.label_name) @structure.label

; Return statement: return value;
(return_statement) @structure.return

; Break statement: break;
(break_statement) @structure.break

; Continue statement: continue;
(continue_statement) @structure.continue

; ============================================
; STRUCTURAL SIGNALS - DECLARATIONS
; ============================================

; Local variable declaration with initialization
(declaration
  declarator: (init_declarator
    declarator: (_) @write.decl_name
    value: (_) @write.decl_value)) @write.declaration

; Struct specifier (struct definition)
(struct_specifier) @structure.struct

; Union specifier
(union_specifier) @structure.union

; Enum specifier
(enum_specifier) @structure.enum

; ============================================
; STRUCTURAL SIGNALS - EXPRESSIONS
; ============================================

; Sizeof expression: sizeof(type) or sizeof(expr)
(sizeof_expression) @structure.sizeof

; Cast expression: (type)value
(cast_expression) @structure.cast

; Pointer dereference: *ptr
(pointer_expression
  operator: "*") @structure.pointer_deref

; Address-of: &variable
(pointer_expression
  operator: "&") @structure.address_of

; Increment/decrement: i++, ++i, i--, --i
(update_expression) @structure.update

; Comma expression: (expr1, expr2)
(comma_expression) @structure.comma

"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting C# implementation details.
        
        Captures method calls, property access, index access, assignments, and structural patterns
        for verification queries like "does method X use pattern Y?".
        
        C#-specific considerations:
        - Uses `invocation_expression` for method calls (wraps `member_access_expression`)
        - Uses `member_access_expression` for property/field access (object.Property)
        - Uses `element_access_expression` for indexer access (array[i], dict[key])
        - Uses `assignment_expression` for all assignments
        - Has `this` node for explicit `this` keyword
        - LINQ expressions use `query_expression` with `from_clause`, `where_clause`, `select_clause`
        - `using_statement` for resource management (like Python's `with`)
        - `await_expression` for async patterns
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method invocation on object: obj.Method(), this.Method()
; invocation_expression wraps member_access_expression for the method name
(invocation_expression
  function: (member_access_expression
    expression: (_) @call.receiver
    name: (identifier) @call.method_name)
  arguments: (argument_list)) @call.method

; Static method call: ClassName.StaticMethod()
(invocation_expression
  function: (member_access_expression
    expression: (identifier) @call.static_class
    name: (identifier) @call.static_method_name)
  arguments: (argument_list)) @call.static

; Simple invocation (local method): Method()
(invocation_expression
  function: (identifier) @call.function_name
  arguments: (argument_list)) @call.function

; ============================================
; PROPERTY/FIELD ACCESS (reads)
; ============================================

; Member access: this.Property, obj.Field, receiver.Member
; The wildcard (_) captures 'this' keyword, identifiers, and other expressions
(member_access_expression
  expression: (_) @access.object
  name: (identifier) @access.property_name) @access.property

; ============================================
; INDEX/SUBSCRIPT ACCESS
; ============================================

; Element access on identifier: array[index], dict[key]
(element_access_expression
  expression: (identifier) @subscript.target_simple
  subscript: (bracketed_argument_list
    (argument (_) @subscript.index))) @subscript.simple

; Element access on member: this._cache[key], obj.List[i]
(element_access_expression
  expression: (member_access_expression
    expression: (_) @subscript.member_object
    name: (identifier) @subscript.member_name)
  subscript: (bracketed_argument_list
    (argument (_) @subscript.member_index))) @subscript.member

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Property/field assignment: this.Property = value, obj.Field = value
(assignment_expression
  left: (member_access_expression
    expression: (_) @write.target_object
    name: (identifier) @write.property_name)
  right: (_) @write.value) @write.property

; Simple variable assignment: x = value
(assignment_expression
  left: (identifier) @write.simple_target
  right: (_) @write.simple_value) @write.simple

; Element assignment: array[i] = value, dict[key] = value
(assignment_expression
  left: (element_access_expression
    expression: (_) @write.element_target
    subscript: (bracketed_argument_list
      (argument (_) @write.element_index)))
  right: (_) @write.element_value) @write.element

; Compound assignment: x += 1, count -= 1
(assignment_expression
  left: (identifier) @write.compound_target
  operator: (_) @write.compound_operator
  right: (_) @write.compound_value) @write.compound

; ============================================
; STRUCTURAL SIGNALS
; ============================================

; For loop: for (int i = 0; i < n; i++)
(for_statement) @structure.for_loop

; Foreach loop: foreach (var item in collection)
(foreach_statement) @structure.foreach_loop

; While loop
(while_statement) @structure.while_loop

; Do-while loop
(do_statement) @structure.do_while

; If statement (conditional)
(if_statement) @structure.conditional

; Switch statement: switch (value) { case ... }
(switch_statement) @structure.switch

; Switch expression (C# 8.0+): value switch { pattern => result }
(switch_expression) @structure.switch_expression

; Conditional/ternary expression: condition ? a : b
(conditional_expression) @structure.ternary

; Try-catch block
(try_statement) @structure.try_catch

; Catch clause
(catch_clause) @structure.catch

; Finally clause
(finally_clause) @structure.finally

; Using statement (resource management): using (var x = ...) { }
(using_statement) @structure.using

; Note: C# 8.0 'using var' declarations are local_declaration_statement with 'using' keyword
; Not a separate node type, so we don't capture them specifically here

; Throw statement
(throw_statement) @structure.throw

; Return statement
(return_statement) @structure.return

; Await expression (async/await)
(await_expression) @structure.await

; Lambda expression: (x) => expression, (x, y) => { statements }
(lambda_expression) @structure.lambda

; Object creation: new ClassName()
(object_creation_expression
  type: (_) @creation.type
  arguments: (argument_list)) @structure.new_instance

; Anonymous object: new { Property = value }
(anonymous_object_creation_expression) @structure.anonymous_object

; Array creation: new int[] { 1, 2, 3 }
(array_creation_expression) @structure.array_creation

; LINQ query expression: from x in collection where ... select ...
(query_expression) @structure.linq_query

; LINQ from clause
(from_clause) @structure.linq_from

; LINQ where clause
(where_clause) @structure.linq_where

; LINQ select clause
(select_clause) @structure.linq_select

; LINQ orderby clause (note: underscore in node name)
(order_by_clause) @structure.linq_orderby

; LINQ group clause
(group_clause) @structure.linq_group

; LINQ join clause
(join_clause) @structure.linq_join

; Lock statement (thread synchronization)
(lock_statement) @structure.lock

; Yield return statement (iterators)
(yield_statement) @structure.yield

; Checked/unchecked blocks
(checked_statement) @structure.checked
(checked_expression) @structure.checked_expr

; Null-conditional access: obj?.Property
(conditional_access_expression) @structure.null_conditional

; Null-coalescing: value ?? default
(binary_expression
  operator: "??") @structure.null_coalescing

; Pattern matching: x is Type t
(is_pattern_expression) @structure.pattern_match

; Pattern matching in switch
(switch_expression_arm) @structure.switch_arm
"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting Kotlin implementation details.
        
        Captures method calls, property access, index access, assignments, and structural patterns
        for verification queries like "does method X use pattern Y?".
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method calls via navigation: obj.method(), this.method()
(call_expression
  (navigation_expression
    (_) @call.receiver
    (identifier) @call.method_name)) @call.method

; Safe call: obj?.method()
(call_expression
  (navigation_expression
    (_) @call.safe_receiver
    (identifier) @call.safe_method_name)) @call.safe_method

; Simple function calls: func(), process()
(call_expression
  (identifier) @call.function_name) @call.function

; ============================================
; PROPERTY/ATTRIBUTE ACCESS
; ============================================

; Navigation expression (property access): this.property, obj.field
(navigation_expression
  (_) @access.object
  (identifier) @access.attr_name) @access.attribute

; ============================================
; INDEX/SUBSCRIPT ACCESS
; ============================================

; Simple index access: list[index], map[key]
(index_expression
  (identifier) @subscript.target_simple
  [(identifier) (number_literal) (_)] @subscript.index) @subscript.simple

; Chained index access: obj.list[index]
(index_expression
  (navigation_expression
    (_) @subscript.nav_object
    (identifier) @subscript.nav_attr)
  (_) @subscript.nav_index) @subscript.navigation

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Property assignment via navigation: this.property = value, obj.field = value
(assignment
  (navigation_expression
    (_) @write.target_object
    (identifier) @write.attr_name)
  (_) @write.value) @write.attribute

; Simple variable assignment: x = value
(assignment
  (identifier) @write.simple_target
  (_) @write.simple_value) @write.simple

; Index assignment: list[i] = value, map[key] = value
(assignment
  (index_expression
    (_) @write.index_target
    (_) @write.index_key)
  (_) @write.index_value) @write.index

; Augmented assignment: x += 1, y -= 2
(assignment
  (identifier) @write.aug_target
  (_) @write.aug_value) @write.augmented

; ============================================
; STRUCTURAL SIGNALS
; ============================================

; For loop: for (item in collection)
(for_statement) @structure.for_loop

; While loop
(while_statement) @structure.while_loop

; Do-while loop
(do_while_statement) @structure.do_while

; When expression (Kotlin's switch/pattern matching)
(when_expression) @structure.when

; If expression (Kotlin if is an expression)
(if_expression) @structure.conditional

; Try-catch block
(try_expression) @structure.try_catch

; Catch block
(catch_block) @structure.catch

; Finally block
(finally_block) @structure.finally

; Lambda literal
(lambda_literal) @structure.lambda

; Annotated lambda (lambdas with trailing lambda syntax)
(annotated_lambda) @structure.annotated_lambda

; Return expression
(return_expression) @structure.return

; Throw expression
(throw_expression) @structure.throw

; Object creation: ClassName()
(call_expression
  (identifier) @creation.type) @structure.new_instance

; Object declaration (singleton pattern)
(object_declaration) @structure.object

; Anonymous function
(anonymous_function) @structure.anonymous_func

; Spread operator: *array
(spread_expression) @structure.spread
"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting Swift implementation details.
        
        Captures method calls, property access, subscript access, assignments, and structural patterns
        for verification queries like "does method X use pattern Y?".
        
        Swift-specific considerations:
        - Uses `navigation_expression` for property/method access (obj.property, self.method)
        - `call_expression` wraps both function calls AND subscript access
        - `call_suffix` contains either `value_arguments` for calls or brackets for subscripts
        - `self_expression` represents the `self` keyword
        - `guard_statement` is Swift's early-exit pattern
        - `repeat_while_statement` is Swift's do-while equivalent
        - `do_statement` with `catch_block` for error handling
        - `await_expression` for async/await
        - `try_expression` for error propagation
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method call with navigation: obj.method(), self.method(), receiver.method()
; call_expression wraps navigation_expression when calling methods
(call_expression
  (navigation_expression
    (_) @call.receiver
    (navigation_suffix
      (simple_identifier) @call.method_name))
  (call_suffix
    (value_arguments))) @call.method

; Simple function call: func(), print(), doSomething()
(call_expression
  (simple_identifier) @call.function_name
  (call_suffix
    (value_arguments))) @call.function

; ============================================
; SUBSCRIPT/INDEX ACCESS
; ============================================

; Note: In Swift's tree-sitter grammar, subscript access and function calls 
; both use call_expression with call_suffix. The difference is:
; - Function calls: call_suffix contains value_arguments starting with '('
; - Subscripts: call_suffix contains value_arguments starting with '['
; 
; Since tree-sitter queries can't reliably filter by text content,
; we capture all call_expressions. The processing code (_process_subscript_capture)
; filters these by checking for '[' in the node text.

; Simple call/subscript: array[index] or func()
; Processing code will filter based on presence of '[' 
(call_expression
  (simple_identifier) @subscript.target_simple
  (call_suffix)) @subscript.simple

; Chained call/subscript: obj.array[index] or obj.method()
; Processing code will filter based on presence of '['
(call_expression
  (navigation_expression
    (_) @subscript.attr_object
    (navigation_suffix
      (simple_identifier) @subscript.attr_name))
  (call_suffix)) @subscript.attribute

; ============================================
; PROPERTY/ATTRIBUTE ACCESS (reads)
; ============================================

; Property access: self.property, obj.field, module.constant
; Uses capture name expected by _process_access_capture: access.attribute
(navigation_expression
  (_) @access.object
  (navigation_suffix
    (simple_identifier) @access.attr_name)) @access.attribute

; Self expression alone (accessing self)
(self_expression) @access.self

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Property assignment via navigation: self.property = value, obj.field = value
; Uses capture names expected by _process_write_capture: write.attribute, write.attr_name
(assignment
  (directly_assignable_expression
    (navigation_expression
      (_) @write.target_object
      (navigation_suffix
        (simple_identifier) @write.attr_name)))
  (_) @write.value) @write.attribute

; Subscript assignment: array[i] = value, dict[key] = value
(assignment
  (directly_assignable_expression
    (call_expression
      (_) @write.subscript_target
      (call_suffix)))
  (_) @write.subscript_value) @write.subscript

; Simple variable assignment: x = value
(assignment
  (directly_assignable_expression
    (simple_identifier) @write.simple_target)
  (_) @write.simple_value) @write.simple

; ============================================
; VARIABLE DECLARATIONS
; ============================================

; Property declaration (both let and var): let x = value, var x = value
; The value_binding_pattern contains the let/var keyword
(property_declaration
  (value_binding_pattern) @write.binding
  (pattern
    (simple_identifier) @write.decl_name)) @write.declaration

; ============================================
; STRUCTURAL SIGNALS
; ============================================

; For-in loop: for item in collection { }
(for_statement) @structure.for_loop

; While loop
(while_statement) @structure.while_loop

; Repeat-while loop (Swift's do-while): repeat { } while condition
(repeat_while_statement) @structure.repeat_while

; If statement (conditional)
(if_statement) @structure.conditional

; Guard statement (Swift-specific early exit)
(guard_statement) @structure.guard

; Switch statement
(switch_statement) @structure.switch

; Do-catch block (error handling)
; Also emits structure.try_except for compatibility with _process_structure_capture
(do_statement) @structure.do_catch @structure.try_except

; Catch block
(catch_block) @structure.catch

; Try expression
(try_expression) @structure.try

; Await expression (async/await)
(await_expression) @structure.await

; Control transfer statements (return, break, continue, throw, fallthrough)
(control_transfer_statement) @structure.control_transfer

; Lambda/closure literal: { params in body }
(lambda_literal) @structure.closure
"""


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
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting Objective-C implementation details.
        
        Captures message sends, function calls, property access, subscripts, assignments,
        and structural patterns for verification queries like "does method X use pattern Y?".
        
        Objective-C-specific considerations:
        - Uses `message_expression` for [receiver message:arg] syntax (THE defining ObjC pattern)
        - Inherits C patterns: field_expression, subscript_expression, etc.
        - `@try/@catch/@finally` for exception handling
        - `for-in` loops for fast enumeration
        - Property access via dot syntax uses `field_expression` (like C)
        - Uses same `call_expression` as C for C-style function calls
        """
        return """
; ============================================
; OBJECTIVE-C MESSAGE SENDS (primary pattern)
; ============================================

; Message send: [receiver message] or [receiver message:arg1 label:arg2]
; This is THE defining Objective-C pattern
(message_expression) @call.objc_message

; ============================================
; C-STYLE FUNCTION CALLS (inherited from C)
; ============================================

; Simple function call: func(), NSLog(), malloc()
(call_expression
  function: (identifier) @call.function_name
  arguments: (argument_list)) @call.function

; Function call via struct pointer: obj->funcPtr()
(call_expression
  function: (field_expression
    argument: (_) @call.receiver
    field: (field_identifier) @call.method_name)
  arguments: (argument_list)) @call.method

; ============================================
; FIELD/PROPERTY ACCESS (reads)
; ============================================

; Property access via dot: obj.property (Objective-C 2.0 dot syntax)
; Also captures C-style struct.field
(field_expression
  argument: (_) @access.object
  field: (field_identifier) @access.field_name) @access.field

; ============================================
; SUBSCRIPT/INDEX ACCESS
; ============================================

; Simple subscript: array[index], dict[key]
; Objective-C uses this for NSArray/NSDictionary subscripting (Modern ObjC)
(subscript_expression
  argument: (identifier) @subscript.target_simple
  index: (_) @subscript.index) @subscript.simple

; Subscript on property: self.array[i], obj.dict[key]
(subscript_expression
  argument: (field_expression
    argument: (_) @subscript.field_object
    field: (field_identifier) @subscript.field_name)
  index: (_) @subscript.field_index) @subscript.field

; Chained subscript: array[i][j]
(subscript_expression
  argument: (subscript_expression) @subscript.chained_target
  index: (_) @subscript.chained_index) @subscript.chained

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Property assignment via dot: self.property = value, obj.field = value
(assignment_expression
  left: (field_expression
    argument: (_) @write.target_object
    field: (field_identifier) @write.field_name)
  right: (_) @write.value) @write.field

; Simple variable assignment: x = value
(assignment_expression
  left: (identifier) @write.simple_target
  right: (_) @write.simple_value) @write.simple

; Subscript assignment: array[i] = value, dict[key] = value
(assignment_expression
  left: (subscript_expression
    argument: (_) @write.subscript_target
    index: (_) @write.subscript_index)
  right: (_) @write.subscript_value) @write.subscript

; Pointer dereference assignment: *ptr = value
(assignment_expression
  left: (pointer_expression
    argument: (_) @write.deref_target)
  right: (_) @write.deref_value) @write.pointer_deref

; ============================================
; STRUCTURAL SIGNALS - LOOPS
; ============================================

; For loop: for (init; cond; update) { }
; Also captures for-in: for (id item in collection) { } (fast enumeration)
(for_statement) @structure.for_loop

; While loop
(while_statement) @structure.while_loop

; Do-while loop
(do_statement) @structure.do_while

; ============================================
; STRUCTURAL SIGNALS - CONDITIONALS
; ============================================

; If statement
(if_statement) @structure.conditional

; Switch statement
(switch_statement) @structure.switch

; Case label
(case_statement) @structure.case

; Ternary expression
(conditional_expression) @structure.ternary

; ============================================
; STRUCTURAL SIGNALS - EXCEPTION HANDLING
; ============================================

; @try/@catch/@finally block
(try_statement) @structure.try_catch

; @catch clause
(catch_clause) @structure.catch

; @finally clause
(finally_clause) @structure.finally

; @throw statement
(throw_statement) @structure.throw

; ============================================
; STRUCTURAL SIGNALS - CONTROL FLOW
; ============================================

; Return statement
(return_statement) @structure.return

; Break statement
(break_statement) @structure.break

; Continue statement
(continue_statement) @structure.continue

; Goto statement (inherited from C)
(goto_statement) @structure.goto

; Labeled statement
(labeled_statement) @structure.label

; ============================================
; STRUCTURAL SIGNALS - ObjC-SPECIFIC
; ============================================

; @synchronized block
(synchronized_statement) @structure.synchronized

; Note: @autoreleasepool is represented as a compound_statement in the grammar,
; not a separate node type, so we cannot capture it distinctly here.

; Class implementation: @implementation ClassName ... @end
; Also used for category implementations with (CategoryName)
(class_implementation) @structure.class_impl

; Class interface: @interface ClassName ... @end
; Also used for category interfaces with (CategoryName)
(class_interface) @structure.class_interface

; Protocol declaration: @protocol ProtocolName ... @end
(protocol_declaration) @structure.protocol

; Note: category_implementation and category_interface are not separate node types
; in tree-sitter-objc; they use class_implementation/class_interface with
; a parenthesized category name

; ============================================
; STRUCTURAL SIGNALS - DECLARATIONS
; ============================================

; Local variable declaration with initialization
(declaration
  declarator: (init_declarator
    declarator: (_) @write.decl_name
    value: (_) @write.decl_value)) @write.declaration

; Sizeof expression
(sizeof_expression) @structure.sizeof

; Cast expression
(cast_expression) @structure.cast

; Pointer expressions
(pointer_expression) @structure.pointer
"""


class _TypeScriptModuleWrapper:
    """
    Wrapper for tree_sitter_typescript module to provide a consistent interface.
    
    The tree_sitter_typescript module uses language_typescript() instead of language()
    like other tree-sitter language modules. This wrapper provides the expected interface.
    """
    def language(self):
        return ts_typescript.language_typescript()


class TypeScriptConfig(LanguageConfig):
    """TypeScript language configuration."""
    
    def __init__(self):
        # tree_sitter_typescript provides language_typescript() for .ts files
        # and language_tsx() for .tsx files - we use typescript for both here
        # We use a wrapper to provide the expected language() interface
        super().__init__(
            name="typescript",
            module=_TypeScriptModuleWrapper(),
            extensions=[".ts", ".tsx", ".mts", ".cts"],
            class_types=["class_declaration"],
            function_types=["function_declaration", "arrow_function", "method_definition"],
            method_types=["method_definition"],
            import_types=["import_statement", "import_clause"],
            interface_types=["interface_declaration", "type_alias_declaration"],
            definition_query="""(class_declaration name: (type_identifier) @class.name) @class.def
(interface_declaration name: (type_identifier) @interface.name) @interface.def
(type_alias_declaration name: (type_identifier) @type.name) @type.def
(function_declaration name: (identifier) @func.name) @func.def
(method_definition name: (property_identifier) @method.name) @method.def
(import_statement) @import.def""",
            test_annotations=["@Test", "@test"],  # Jest, Vitest, etc.
            test_name_patterns=["test", "Test", "spec", "Spec", "describe", "it", "should"]
        )
    
    def _extract_signature_impl(self, node: Node, code: bytes) -> str:
        """Extract TypeScript function/method signature with type annotations."""
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
                # Handle arrow functions with implicit return
                if '=>' in line and '{' not in line:
                    parts = line.split('=>')
                    signature_lines[-1] = parts[0].strip() + ' =>'
                    break
            
            signature = ' '.join(signature_lines)
            return signature
        except:
            return code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore').strip()
    
    def get_annotation_query(self) -> str:
        """Get tree-sitter query for extracting TypeScript decorators."""
        return "(decorator) @annotation"
    
    def get_implementation_query(self) -> str:
        """
        Get tree-sitter query for extracting TypeScript implementation details.
        
        Captures method calls, property access, subscript access, assignments, and structural patterns
        for verification queries like "does method X use pattern Y?".
        
        TypeScript-specific considerations:
        - Uses `member_expression` for property access (like JavaScript)
        - Uses `property_identifier` instead of Python's `identifier` for property names
        - `for_in_statement` covers both `for...in` and `for...of` loops
        - Type annotations don't affect the AST structure for expressions
        - Supports async/await, arrow functions, and optional chaining
        """
        return """
; ============================================
; METHOD/FUNCTION CALLS
; ============================================

; Method calls on objects: obj.method(), this.method(), receiver.method()
(call_expression
  function: (member_expression
    object: (_) @call.receiver
    property: (property_identifier) @call.method_name)) @call.method

; Simple function calls: func(), importedFunc()
(call_expression
  function: (identifier) @call.function_name) @call.function

; ============================================
; SUBSCRIPT/INDEX ACCESS
; ============================================

; Simple subscript: arr[index], obj[key], map[id]
(subscript_expression
  object: (identifier) @subscript.target_simple
  index: (_) @subscript.index) @subscript.simple

; Subscript on member expression: this.items[i], obj.data[key]
(subscript_expression
  object: (member_expression
    object: (_) @subscript.attr_object
    property: (property_identifier) @subscript.attr_name)
  index: (_) @subscript.attr_index) @subscript.attribute

; ============================================
; PROPERTY/ATTRIBUTE ACCESS (reads)
; ============================================

; Property access: this.property, obj.field, module.export
(member_expression
  object: (_) @access.object
  property: (property_identifier) @access.property_name) @access.property

; ============================================
; ASSIGNMENTS (writes)
; ============================================

; Property assignment: this.prop = value, obj.field = x
(assignment_expression
  left: (member_expression
    object: (_) @write.target_object
    property: (property_identifier) @write.property_name)
  right: (_) @write.value) @write.property

; Simple variable assignment: x = value
(assignment_expression
  left: (identifier) @write.simple_target
  right: (_) @write.simple_value) @write.simple

; Subscript assignment: arr[i] = value, obj[key] = value
(assignment_expression
  left: (subscript_expression
    object: (_) @write.subscript_target
    index: (_) @write.subscript_index)
  right: (_) @write.subscript_value) @write.subscript

; Variable declarations: const x = value, let y = value
(lexical_declaration
  (variable_declarator
    name: (identifier) @write.var_name
    value: (_) @write.var_value)) @write.declaration

; ============================================
; STRUCTURAL SIGNALS
; ============================================

; Traditional for loop: for (let i = 0; i < n; i++)
(for_statement) @structure.for_loop

; For-in and for-of loops (same node type in TypeScript grammar)
; for (const key in obj) and for (const item of array)
(for_in_statement) @structure.for_in_of

; While loop
(while_statement) @structure.while_loop

; If statement (conditional)
(if_statement) @structure.conditional

; Switch statement
(switch_statement) @structure.switch

; Try-catch block
(try_statement) @structure.try_catch

; Throw statement
(throw_statement) @structure.throw

; Return statement
(return_statement) @structure.return

; Await expression
(await_expression) @structure.await

; New expression (object creation): new ClassName()
(new_expression
  constructor: (_) @creation.type) @structure.new_instance

; Arrow function expression
(arrow_function) @structure.arrow_function

; Ternary/conditional expression: condition ? a : b
(ternary_expression) @structure.ternary

; Template literal (template strings with expressions)
(template_string) @structure.template_string

; Spread element: ...array
(spread_element) @structure.spread

; Yield expression (generators)
(yield_expression) @structure.yield

; Class declaration
(class_declaration) @structure.class

; Method definition
(method_definition
  name: (property_identifier) @method.name) @structure.method
"""


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
    "typescript": TypeScriptConfig(),
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