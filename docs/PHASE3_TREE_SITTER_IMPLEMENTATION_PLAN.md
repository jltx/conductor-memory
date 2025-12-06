# Phase 3: Multi-Language AST Chunking Implementation Plan

## Summary of Decisions

| Decision | Choice |
|----------|--------|
| Class split threshold | 100 lines |
| Import handling | One chunk per file |
| Method signatures | Full with types (Option C) |
| Docstrings | Include in chunk |
| Nested classes | Separate chunks with `parent:` tag |
| Test detection | Name patterns + annotations + file path |
| Languages | All 9 at once |

## Implementation Steps

### Step 1: Add Dependencies ✅

**File**: `requirements.txt`

Added tree-sitter + 9 language modules.

### Step 2: Create Parsers Subpackage ✅

**New files**:
```
src/conductor_memory/search/parsers/
├── __init__.py
├── base.py
├── tree_sitter_parser.py
├── language_configs.py
└── domain_detector.py
```

### Step 3: `base.py` - Abstract Interface ✅

Implemented `ContentParser` ABC with `supports()` and `parse()` methods.
Also includes `ParseError` and `UnsupportedLanguageError` exceptions.

### Step 4: `language_configs.py` - Per-Language Configuration ✅

Implemented per-language configuration classes for all 9 languages:
- Python, Java, Kotlin, Go, Ruby, C, C#, Swift, Objective-C
- Each has tree-sitter query strings (fixed for 0.25.x API)
- Method signature extraction with types
- Test annotation detection

### Step 5: `domain_detector.py` - Domain Classification ✅

Implemented with full domain detection:
- test (name patterns, annotations, file path)
- imports
- interface
- private
- accessor
- constant
- class
- function

### Step 6: `tree_sitter_parser.py` - Main Implementation ✅

Implemented full parser with:
- Class chunking (small classes as single chunk, large classes split)
- Method extraction with proper byte-range containment check
- Import combining into single chunk
- Uses `ast_node_type` for domain detection (fixed bug where capture name was used instead)

### Step 7: Modify `chunking.py` ✅

Integrated `TreeSitterParser` into `ChunkingManager.chunk_text()`.

### Step 8: Update `ChunkMetadata` ✅

Added `parent_class` field to `ChunkMetadata`.

### Step 9: Update Tag Generation in `memory_service.py` ✅

Added `parent:ClassName` tag support.

### Step 10: Unit Tests - PENDING

Need to write formal unit tests. Currently tested via `test_tree_sitter_queries.py` which validates all 9 language queries work.

### Step 11: Integration Test - PENDING

Need to test with actual codebase (truthsocial-android).

## Current Status

**Completed:**
- ✅ All 9 language parsers working (validated with test script)
- ✅ Domain detection working (class, function, test, accessor, imports)
- ✅ Method containment check (methods inside classes not duplicated)
- ✅ tree-sitter 0.25.x API compatibility (QueryCursor constructor fix)

**Bugs Fixed:**
- Query syntax errors (leading whitespace)
- Wrong Kotlin node types (`simple_identifier` → `identifier`, `import_header` → `import`)
- Domain detection used capture name instead of AST node type

**Remaining:**
- Unit tests
- Integration test with real codebase
- Large class splitting (>100 lines) not yet exercised

## Expected Outcomes

After implementation:

| Query | Before | After |
|-------|--------|-------|
| "FeedsViewModel" | Returns full 500-line class | Returns class_summary + individual methods |
| Search with `exclude_tags: ["domain:test"]` | No effect (no domain tags on Java/Kotlin) | Properly excludes test code |
| Search with `domain_boosts: {"class": 1.5}` | No effect | Boosts class and class_summary results |
| "authentication" in truthsocial | Might return build/generated code | Only returns actual source code |

## Estimated Effort

| Step | Time |
|------|------|
| Dependencies | 5 min |
| base.py | 15 min |
| language_configs.py (9 languages) | 2 hr |
| domain_detector.py | 45 min |
| tree_sitter_parser.py | 2.5 hr |
| Integrate into chunking.py | 30 min |
| Update ChunkMetadata & tags | 15 min |
| Unit tests | 1.5 hr |
| Integration testing | 1 hr |
| Bug fixes & tuning | 1 hr |
| **Total** | **~10 hours** |

## Implementation Notes

- **Class split threshold**: 100 lines
- **Import handling**: One chunk per file containing all imports
- **Method signatures**: Full with types (e.g., `def method(self, arg1: Type1) -> ReturnType`)
- **Docstrings**: Include in chunk (not just summary)
- **Nested classes**: Separate chunks with `parent:OuterClass` tag
- **Test detection**: Name patterns + annotations + file path checks
- **Languages**: Python, Ruby, Go, C, C#, Kotlin, Swift, Objective-C, Java

## Dependencies Added

```
tree-sitter>=0.25.0
tree-sitter-python>=0.23.0
tree-sitter-java>=0.23.0
tree-sitter-ruby>=0.23.0
tree-sitter-go>=0.25.0
tree-sitter-c>=0.23.0
tree-sitter-c-sharp>=0.23.0
tree-sitter-kotlin>=1.0.0
tree-sitter-swift>=0.0.1
tree-sitter-objc>=3.0.0
```

Total size: ~30MB (pre-compiled wheels)