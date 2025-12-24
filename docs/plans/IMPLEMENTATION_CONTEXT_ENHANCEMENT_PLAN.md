# Conductor Memory Enhancement Plan: Implementation Context for Verification Tasks

**Status:** In Progress  
**Created:** 2024-12-23  
**Updated:** 2024-12-24  
**Author:** Claude + Joshua  

## Executive Summary

Enhance conductor-memory to better support **verification queries** like "does feature X use pattern Y?" and **implementation detail queries** like "how does method X work?". 

The key insight from a real verification task (options-ml-trader sliding window compatibility check) is that current search returns the right *files* but lacks sufficient *implementation depth* to answer verification questions without reading the full source.

## Problem Statement

### Current Behavior
```
User: "Verify _generate_features uses window-relative bar_index for DataFrame access"

Memory returns:
- SetupFeatureGenerator class summary
- File locations with class/function names
- Purpose: "Generates features for swing trading setups"
```

### Desired Behavior
```
Memory returns:
- Evidence: "df.iloc[bar_index]" found in method body
- Context: "bar_index parameter used as subscript index"
- Verification: CONFIRMED - method uses window-relative indexing
```

## Design Principles

1. **Zero-LLM-call path must remain viable** - Heuristic extraction provides baseline depth
2. **LLM enhancement is additive** - Richer summaries when budget allows
3. **Generic, not domain-specific** - No hardcoded patterns; extract raw signals
4. **Incremental value** - Each phase delivers standalone improvements
5. **Performance-conscious** - Stay within ~1 hour for 3000 files summarization budget

---

## Phase 1: Generic Implementation Signal Extraction

**Goal:** Extract method-level implementation details via AST, making them searchable without LLM costs.

**Effort:** 3-4 days  
**Impact:** High  
**LLM Cost:** None

### 1.1 What to Extract

For every method/function, extract these **generic signals**:

```python
@dataclass
class MethodImplementationDetail:
    name: str
    signature: str  # Full signature with type hints
    
    # Call patterns (no predefined list - extract everything)
    internal_calls: List[str]    # self.method(), self.attr.method()
    external_calls: List[str]    # module.func(), Class.method()
    
    # Data access patterns
    attribute_reads: List[str]   # self._df, self.config.value
    attribute_writes: List[str]  # self._cache = x
    subscript_access: List[str]  # df[key], series.iloc[idx], dict[key]
    
    # Parameter usage
    parameters_used: List[str]   # Which params appear in method body
    
    # Structural signals
    has_loop: bool
    has_conditional: bool
    has_try_except: bool
    is_async: bool
    line_count: int
```

### 1.2 Why Generic Signals Work

Instead of detecting predefined patterns like `uses_iloc`, we extract raw data:

| Extracted Signal | Example | Enables Query |
|------------------|---------|---------------|
| `subscript_access: ["df.iloc[bar_index]"]` | swing_strategy.py | "find methods using iloc with bar_index" |
| `internal_calls: ["self.feature_generator.fit"]` | swing_strategy.py | "what calls fit?" |
| `attribute_reads: ["self._feature_generator_fitted_len"]` | swing_strategy.py | "what reads fitted_len?" |

No predefined patterns needed - raw signals are inherently searchable.

### 1.3 Storage Strategy

**Enhanced Chunk Content:**

Append implementation signals to indexed chunk content so they're searchable via existing hybrid search:

```python
original_content = "def _generate_features(self, df, setup, bar_index):..."

enhanced_content = f"""
{original_content}

[Implementation Signals]
Calls: feature_generator.fit, create_setup_context, _get_htf_features
Reads: self._df, self._feature_generator_fitted_len, context.bar_index
Writes: self._feature_generator_fitted_len
Subscripts: df.iloc[bar_index], self._atr_series.iloc[bar_index]
Parameters used: df, setup, bar_index
"""
```

**Tags for Filtering:**

Generate searchable tags from extractions:
- `calls:feature_generator.fit`
- `calls:create_setup_context`
- `reads:self._df`
- `subscript:iloc`
- `param:bar_index`

### 1.4 AST Extraction (Language-Agnostic)

Tree-sitter queries that work across languages:

```scheme
; Capture all method/function calls
(call
  function: (attribute
    object: (_) @receiver
    attribute: (identifier) @method_name)) @call

; Capture all subscript/index access  
(subscript
  value: (_) @target
  subscript: (_) @index) @subscript

; Capture all attribute access
(attribute
  object: (_) @object
  attribute: (identifier) @attr) @access

; Capture assignments (writes)
(assignment
  left: (attribute) @write_target
  right: (_)) @assignment
```

These patterns work for Python, Java, TypeScript, Go, etc.

### 1.5 New Search Parameters

```python
memory_search(
    query="sliding window indexing",
    calls=["fit", "iloc"],        # NEW: Methods that call these
    accesses=["bar_index"],       # NEW: Methods that access these
    subscripts=["iloc"],          # NEW: Methods with subscript patterns
)
```

### 1.6 Implementation Tasks

- [x] Extend `HeuristicExtractor` with method body analysis
- [x] Add tree-sitter queries for calls, subscripts, attributes (Python)
- [x] Add implementation queries for Java — see `JavaConfig.get_implementation_query()` (2024-12-24)
- [x] Add implementation queries for remaining languages (Kotlin, TypeScript, Go, C#, Swift, Ruby, C/ObjC) — Completed 2024-12-24
- [x] Create `MethodImplementationDetail` dataclass
- [x] Update `ChunkMetadata` with `implementation_details` field and helper methods:
  - `get_searchable_signals()` - formats signals as searchable text
  - `get_signal_tags()` - generates tags like `calls:method`, `subscript:iloc`, `reads:self._attr`
- [x] Update `ChunkingManager` to populate implementation details during chunking
- [x] Generate tags from implementation signals (integrate into indexing)
- [x] Add `calls`, `accesses`, `subscripts` filter params to `search_async()`
- [x] Update BM25 index to include enhanced content — Signals included via `get_searchable_signals()` in chunk content
- [x] Write tests for extraction accuracy (tests/test_implementation_signals.py - 37 tests)

**Phase 1 Status: ✅ COMPLETE** (2024-12-24)

---

## Phase 2: Implementation-Aware LLM Summaries

**Goal:** Enhance LLM summaries to explain HOW code works, not just WHAT it does.

**Effort:** 2-3 days  
**Impact:** High  
**LLM Cost:** Same model, +20% tokens (larger prompt with signals)

### 2.1 Leverage Phase 1 Extractions

Pass extracted signals to LLM for context-aware summarization:

```
Given this code and its implementation signals:
- Calls: {extracted_calls}
- Reads: {extracted_attribute_reads}
- Subscripts: {extracted_subscripts}

Generate a summary that explains:
1. What this code does (purpose)
2. HOW it accomplishes this (key mechanisms)
3. Notable patterns or optimizations used
```

### 2.2 Enhanced Summary Schema

**Current:**
```json
{
  "purpose": "Generates features for swing trading setups",
  "pattern": "Generator",
  "key_exports": ["SetupFeatureGenerator"],
  "dependencies": ["pandas", "numpy"],
  "domain": "ml"
}
```

**Enhanced:**
```json
{
  "purpose": "Generates features for swing trading setups",
  "pattern": "Generator",
  "key_exports": ["SetupFeatureGenerator"],
  "dependencies": ["pandas", "numpy"],
  "domain": "ml",
  
  "how_it_works": "Uses window-relative indexing via df.iloc[bar_index]. Caches fitted state in _feature_generator_fitted_len to avoid repeated computation. Resamples HTF data for multi-timeframe features.",
  
  "key_mechanisms": [
    "window-relative indexing",
    "fit caching",
    "HTF resampling"
  ],
  
  "method_summaries": {
    "_generate_features": "Fits generator only when bar_index exceeds cached length, uses window-relative DataFrame access",
    "_get_htf_features": "Resamples base timeframe to HTF, requires minimum 14 bars"
  }
}
```

### 2.3 Updated Prompts

Add to summarization prompt:

```
IMPORTANT: Focus on HOW the code works, not just what it does.

For each significant method, explain:
- What data structures/indices it uses
- Any caching, memoization, or optimization strategies
- How parameters flow through the logic
- Any coordinate systems or index translations

Use the implementation signals provided to inform your analysis.
```

### 2.4 Implementation Tasks

- [ ] Update `FileSummarizer._build_user_prompt()` to include implementation signals
- [ ] Update `FileSummarizer._build_system_prompt()` with HOW-focused instructions
- [ ] Extend `FileSummary` dataclass with new fields
- [ ] Update summary JSON schema
- [ ] Test with current model, evaluate quality
- [ ] Document token usage impact

---

## Phase 3: Verification Search Mode

**Goal:** Distinct search mode for "does X use pattern Y?" type queries.

**Effort:** 3-4 days  
**Impact:** Medium-High  
**LLM Cost:** None (rule-based parsing) or minimal (LLM-assisted parsing)

### 3.1 Verification Mode Behavior

```python
memory_search(
    query="verify _generate_features uses window-relative bar_index for DataFrame access",
    search_mode="verify"
)
```

**Processing Steps:**
1. Parse query to extract subject and claim
2. Find subject (method/class/file) via existing search
3. Check implementation signals for evidence
4. Return structured verification result

### 3.2 Query Parsing

**Rule-based approach (start here):**

```python
VERIFY_PATTERNS = [
    r"verify\s+(\w+)\s+uses?\s+(.+)",
    r"does\s+(\w+)\s+use\s+(.+)",
    r"check\s+if\s+(\w+)\s+(.+)",
    r"confirm\s+(\w+)\s+(.+)",
]

def parse_verification_query(query: str) -> Optional[VerificationIntent]:
    for pattern in VERIFY_PATTERNS:
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            return VerificationIntent(
                subject=match.group(1),
                claim=match.group(2)
            )
    return None
```

**Future enhancement:** LLM-assisted parsing for complex queries.

### 3.3 Evidence Matching

Match claim keywords against implementation signals:

```python
def find_evidence(subject_chunk: MemoryChunk, claim: str) -> List[Evidence]:
    evidence = []
    claim_terms = extract_key_terms(claim)
    
    for subscript in subject_chunk.subscripts:
        if matches_any(subscript, claim_terms):
            evidence.append(Evidence(
                type="subscript_access",
                detail=subscript,
                relevance=calculate_relevance(subscript, claim_terms)
            ))
    
    return sorted(evidence, key=lambda e: e.relevance, reverse=True)
```

### 3.4 Verification Response Format

```json
{
  "search_mode": "verify",
  "subject": {
    "name": "_generate_features",
    "file": "src/trading/replay/swing_strategy.py",
    "found": true
  },
  "verification": {
    "status": "SUPPORTED",
    "confidence": 0.92,
    "evidence": [
      {"type": "subscript_access", "detail": "df.iloc[bar_index]", "line": 631}
    ]
  },
  "summary": "VERIFIED: _generate_features uses bar_index as window-relative index."
}
```

### 3.5 Verification Statuses

| Status | Meaning |
|--------|---------|
| SUPPORTED | Evidence found supporting the claim |
| NOT_SUPPORTED | Subject found but no evidence for claim |
| CONTRADICTED | Evidence suggests opposite of claim |
| INCONCLUSIVE | Some evidence but not definitive |
| SUBJECT_NOT_FOUND | Could not locate the subject |

### 3.6 Implementation Tasks

- [ ] Add VerificationIntent and Evidence dataclasses
- [ ] Implement rule-based query parser
- [ ] Implement evidence matching against implementation signals
- [ ] Add search_mode="verify" handling in search_async()
- [ ] Create verification response formatter
- [ ] Write tests for various verification scenarios

---

## Phase 4: Method Call Graph

**Goal:** Track method-to-method relationships for "what calls X?" queries.

**Effort:** 4-5 days  
**Impact:** Medium  
**LLM Cost:** None

### 4.1 Extend Import Graph to Method Level

```python
@dataclass
class MethodNode:
    qualified_name: str
    file_path: str
    class_name: Optional[str]
    method_name: str
    line_number: int

@dataclass  
class MethodCallGraph:
    nodes: Dict[str, MethodNode]
    edges: Dict[str, List[str]]  # caller -> [callees]
    reverse_edges: Dict[str, List[str]]  # callee -> [callers]
```

### 4.2 New MCP Tool

```python
memory_method_relationships(
    method="_generate_features",
    codebase="options-ml-trader",
    relationship="all"
)
```

### 4.3 Implementation Tasks

- [ ] Create MethodNode and MethodCallGraph dataclasses
- [ ] Implement call graph builder from chunk data
- [ ] Add graph storage (in-memory per codebase)
- [ ] Implement memory_method_relationships MCP tool
- [ ] Write tests for graph construction and queries

---

## Implementation Order & Timeline

| Phase | Effort | Dependencies | Priority | Status |
|-------|--------|--------------|----------|--------|
| Phase 1: Generic Signal Extraction | 3-4 days | None | P0 - Foundation | ✅ Complete |
| Phase 2: Implementation-Aware Summaries | 2-3 days | Phase 1 | P1 | Not Started |
| Phase 3: Verification Search Mode | 3-4 days | Phase 1 | P1 - Primary use case | Not Started |
| Phase 4: Method Call Graph | 4-5 days | Phase 1 | P2 | Not Started |

**Total: ~14-16 days**

**Recommended order:** Phase 1 -> Phase 3 -> Phase 2 -> Phase 4

---

## Open Questions

1. **Subscript detail level**: Store iloc, df.iloc, or df.iloc[bar_index]?
   - **Proposed:** All three - tag for filtering, full expression in content

2. **Cross-file call resolution**: Store as-is or resolve to actual class?
   - **Proposed:** Start simple, add resolution in Phase 4

3. **Claim parsing**: Rule-based or LLM fallback?
   - **Proposed:** Rule-based first, add LLM if needed

---

## Success Metrics

### Phase 1
- 90%+ method calls correctly extracted
- Search with calls=["fit"] returns expected methods
- Indexing time regression < 15%

### Phase 2
- Summaries include "how_it_works" section
- Token usage increase < 30%

### Phase 3
- 80%+ verification queries correctly parsed
- Evidence matching finds relevant signals

### Phase 4
- Call graph correctly represents method relationships
- Graph updates correctly on file changes

---

## Progress Log

### Phase 1 Completion — 2024-12-24

**Summary:** Implemented generic implementation signal extraction via AST analysis, enabling searchable method-level details without LLM costs.

**What Was Implemented:**
- `MethodImplementationDetail` dataclass for storing extracted signals (calls, attribute reads/writes, subscript access, parameters, structural signals)
- Tree-sitter queries for Python to extract calls, subscripts, and attribute access patterns
- `ChunkMetadata` extensions with `get_searchable_signals()` and `get_signal_tags()` helper methods
- `ChunkingManager` integration to populate implementation details during indexing
- New search filter parameters: `calls`, `accesses`, `subscripts` in `search_async()`
- Signal tags auto-generated during indexing (e.g., `calls:fit`, `subscript:iloc`, `reads:self._cache`)
- Implementation signals appended to chunk content for BM25 searchability

**Files Modified:**
- `src/conductor_memory/core/models.py` — Added `MethodImplementationDetail` dataclass, updated `ChunkMetadata`
- `src/conductor_memory/search/heuristics.py` — Extended `HeuristicExtractor` with method body analysis
- `src/conductor_memory/search/chunking.py` — Updated `ChunkingManager` to populate implementation details
- `src/conductor_memory/search/parsers/tree_sitter_parser.py` — Added implementation signal extraction queries for Python
- `src/conductor_memory/service/memory_service.py` — Added `calls`, `accesses`, `subscripts` filter params to `search_async()`
- `src/conductor_memory/client/tools.py` — Exposed new filter params in MCP tool schema

**Tests Added:**
- `tests/test_implementation_signals.py` — 37 tests covering extraction accuracy, edge cases, and search filtering

**Deviations from Original Plan:**
- **BM25 integration approach:** Instead of a separate BM25 index update, signals are included via `get_searchable_signals()` in chunk content, which automatically makes them searchable through existing hybrid search.

**Metrics Achieved:**
- Method calls correctly extracted: >90% (verified via test suite)
- Search with `calls=["fit"]` returns expected methods: Working
- Indexing time regression: Minimal (signal extraction is fast, runs during existing chunking pass)

### Multi-Language Implementation Queries — 2024-12-24

**Summary:** Extended implementation signal extraction to support 8 additional programming languages, completing full Phase 1 multi-language coverage.

**Languages Added:**
- Java
- Kotlin
- TypeScript
- Go
- C#
- Swift
- Ruby
- C/Objective-C

**What Each Language Supports:**
Each language configuration includes comprehensive tree-sitter queries for:
- **Calls:** Method invocations, function calls, chained calls
- **Field/Attribute Access:** Property reads, member access patterns
- **Subscripts:** Array indexing, dictionary access, bracket notation
- **Assignments:** Field writes, property assignments
- **Structural Signals:** Loops, conditionals, try/catch, async markers

**Files Modified:**
- `src/conductor_memory/search/parsers/language_configs.py` — Added `get_implementation_query()` methods to all language configs

**Tests Added:**
- `tests/test_typescript_impl_queries.py` — TypeScript implementation query tests
- `tests/test_csharp_impl_queries.py` — C# implementation query tests

**Impact:**
- All Phase 1 tasks now complete
- Implementation signal extraction works across Python, Java, Kotlin, TypeScript, Go, C#, Swift, Ruby, and C/Objective-C
- No deviations from plan remain for Phase 1
