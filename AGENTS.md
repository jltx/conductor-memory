# AGENTS.md - Memory-Augmented Code Assistant Guidelines

This file defines how AI coding assistants should leverage the conductor-memory MCP server for efficient codebase exploration and context retrieval.

## Memory MCP Tools Available

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `memory_search` | Advanced semantic/keyword search with filtering | Heuristic filtering, summary integration, relevance boosting, **verification mode** |
| `memory_store` | Store conversation context for later retrieval | Tagging, pinning, source tracking |
| `memory_store_decision` | Store architectural decisions (auto-pinned) | Structured decision format, permanent storage |
| `memory_store_lesson` | Store debugging insights/lessons learned (auto-pinned) | Problem-solution format, searchable |
| `memory_status` | Check indexing status and codebase info | Multi-codebase status, file counts |
| `memory_summarization_status` | Check LLM summarization progress | Time estimates, completion tracking |
| `memory_reindex_codebase` | Force reindexing of specific codebase | Refresh metadata, update heuristics |
| `memory_delete` | Remove outdated memories by ID | Cleanup, maintenance |
| `memory_prune` | Clean up old unpinned memories | Bulk cleanup by age |
| `memory_method_relationships` | **NEW:** Query method call relationships | Find callers/callees, call chains, transitive dependencies |
| `memory_file_centrality` | Get files sorted by importance | PageRank-based centrality for prioritization |
| `memory_file_dependencies` | Get imports/imported_by for a file | Dependency analysis |
| `memory_import_graph_stats` | Get import graph statistics | File counts, edges, centrality info |
| `memory_invalidate_summaries` | **NEW:** Clear all summaries for re-summarization | Schema migration, force regeneration |

## Search Strategy: Tiered Approach

Use a tiered strategy based on query complexity and user needs:

### Tier 1: Memory Search (Default - Use First)

**When:** Quick lookups, finding specific code, augmenting context

```
User: "Where is the login logic?"
Action: memory_search("login authentication user credentials")
```

**Characteristics:**
- Response time: 20-50ms
- Token cost: Low (~500-1500 tokens)
- Returns: Code snippets with file locations
- Best for: Known concepts, specific lookups, context augmentation

**Advanced Query Features:**
- **Heuristic Filtering**: `min_class_count=1`, `languages=["python"]`, `include_tags=["domain:service"]`
- **Summary Integration**: `include_summaries=True` for LLM-generated file summaries
- **Relevance Boosting**: `boost_summarized=True` for 15% boost to summarized files
- **Multi-codebase**: `codebase="backend-api"` for targeted searches
- **Smart Mode Detection**: Auto-detects semantic vs keyword search based on query

**Query Tips:**
- Use multiple related terms: `"feed timeline posts list"` not just `"feed"`
- Include technical terms: `"repository pattern data layer"` 
- Filter by complexity: `min_class_count=2` for substantial files
- Try variations if first query returns nothing

### Tier 2: Explore Agent (Escalate When Needed)

**When:** Memory search returns insufficient results, need synthesized understanding, unfamiliar territory

```
User: "How does the entire notification system work?"
Action: Task agent with explore type for comprehensive analysis
```

**Characteristics:**
- Response time: 30-60 seconds
- Token cost: High (~3000-5000 tokens)
- Returns: Synthesized summaries, architecture diagrams, design pattern explanations
- Best for: Learning new areas, documentation, complex feature understanding

### Decision Tree

```
User asks about code/feature
           │
           ▼
   ┌───────────────────┐
   │ Try memory_search │
   │ with good query   │
   └─────────┬─────────┘
             │
             ▼
    ┌────────────────┐
    │ Results found? │
    └───────┬────────┘
            │
     ┌──────┴──────┐
     │             │
    Yes            No
     │             │
     ▼             ▼
┌─────────┐  ┌──────────────┐
│ Return  │  │ Try different│
│ results │  │ query terms  │
└─────────┘  └──────┬───────┘
                    │
                    ▼
           ┌────────────────┐
           │ Still nothing? │
           └───────┬────────┘
                   │
            ┌──────┴──────┐
            │             │
           Yes            No
            │             │
            ▼             ▼
    ┌───────────────┐  ┌─────────┐
    │ Use explore   │  │ Return  │
    │ task agent    │  │ results │
    └───────────────┘  └─────────┘
```

## Storing Knowledge

### When to Store Decisions

Store architectural decisions when:
- Choosing between alternative approaches
- Establishing patterns for consistency
- Making tradeoffs with long-term implications
- Deprecating old approaches

**Format:**
```
DECISION: [One-line summary]
CONTEXT: [What prompted this decision]
ALTERNATIVES: [Other options considered]
RATIONALE: [Why this choice was made]
CONSEQUENCES: [Implications and tradeoffs]
```

**Example:**
```
memory_store_decision(
  content="""DECISION: Use Room database for offline caching instead of SharedPreferences
CONTEXT: App needs to cache large amounts of feed data for offline viewing
ALTERNATIVES: SharedPreferences (simple but limited), SQLite directly (complex), DataStore (good for preferences, not relational data)
RATIONALE: Room provides type-safe queries, migration support, and handles relational data well
CONSEQUENCES: Adds compile-time overhead for annotation processing, requires schema migrations for updates""",
  tags=["database", "caching", "architecture"]
)
```

### When to Store Lessons

Store debugging insights when:
- Solving non-obvious bugs
- Discovering undocumented behavior
- Finding workarounds for library issues
- Learning from production incidents

**Format:**
```
LESSON: [One-line summary]
SYMPTOM: [What went wrong / how it manifested]
ROOT CAUSE: [The actual underlying issue]
SOLUTION: [How to fix it]
PREVENTION: [How to avoid in future]
```

**Example:**
```
memory_store_lesson(
  content="""LESSON: ChromaDB collection.get() returns empty list, not exception, for missing IDs
SYMPTOM: Delete operations reported "not found" even for IDs that were just stored
ROOT CAUSE: Exception handler was catching AttributeError from typo, not missing data
SOLUTION: Fixed attribute name from self._hybrid_searchers to self.hybrid_searcher
PREVENTION: Use specific exception types, not bare except; test error paths explicitly""",
  tags=["chromadb", "debugging", "error-handling"]
)
```

### When NOT to Store

- Trivial or temporary information
- User-specific preferences (use config files)
- Sensitive data (credentials, tokens)
- Rapidly changing state

## Multi-Codebase Usage

When working with multiple codebases:

1. **Check status first** to see available codebases:
   ```
   memory_status() → shows indexed codebases and file counts
   ```

2. **Specify codebase** for targeted searches:
   ```
   memory_search(query="...", codebase="backend-api")
   ```

3. **Omit codebase** for cross-codebase searches:
   ```
   memory_search(query="authentication flow")  # searches all
   ```

## Response Formatting

When returning memory search results to users:

**Do:**
- Highlight the most relevant 2-3 results
- Include file paths for easy navigation
- Quote key code snippets directly
- Mention if results seem incomplete (suggest explore agent)

**Don't:**
- Dump all 10 results without curation
- Omit file locations
- Pretend memory search gives complete understanding of complex systems

**Example Response:**
```
Based on memory search, the login logic is in:

1. `app/src/auth/LoginViewModel.kt` (lines 45-89) - handles credential validation
2. `core/data/AuthRepository.kt` (lines 23-67) - manages token storage

Key snippet from LoginViewModel:
​```kotlin
fun login(email: String, password: String) {
    viewModelScope.launch {
        authRepository.authenticate(email, password)
            .onSuccess { token -> saveToken(token) }
            .onError { handleLoginError(it) }
    }
}
​```

Want me to explore the full authentication architecture in more depth?
```

## Performance Characteristics

| Operation | Typical Time | Token Cost | Use Case |
|-----------|--------------|------------|----------|
| memory_search | 20-50ms | ~500-1500 | Quick lookups |
| memory_store | 100-200ms | ~100 | Saving context |
| memory_status | 10-20ms | ~200 | Health checks |
| explore agent | 30-60s | ~3000-5000 | Deep analysis |

## Error Handling

- **"No relevant context found"**: Try different query terms, broaden search
- **"Codebase not found"**: Check memory_status for available codebases
- **"Memory not found" on delete**: Memory may have been pruned or ID is incorrect
- **Slow indexing**: Large codebases take time on first index; check memory_status for progress

## Advanced Features (Latest)

### Heuristic Filtering
Filter search results by code structure and complexity:
```
memory_search(
    query="authentication logic",
    min_class_count=2,           # Files with 2+ classes
    languages=["python"],        # Python files only
    include_tags=["domain:service"]  # Service layer code
)
```

### Implementation Signal Filtering
Filter by method calls, attribute access, and subscript patterns extracted from AST analysis:
```
memory_search(
    query="sliding window indexing",
    calls=["fit", "iloc"],        # Methods that call these functions
    accesses=["bar_index"],       # Methods that access these attributes
    subscripts=["iloc"],          # Methods with subscript patterns (e.g., df.iloc[x])
)
```

Use cases:
- **Find callers**: `calls=["fit"]` → methods that call `fit()`
- **Find data access**: `accesses=["_cache"]` → methods that read `self._cache`
- **Find indexing patterns**: `subscripts=["iloc"]` → methods using `df.iloc[...]`

### Summary Integration
Include LLM-generated file summaries for better context:
```
memory_search(
    query="user management",
    include_summaries=True,      # Include structured summaries
    boost_summarized=True        # 15% boost to summarized files
)
```

Returns enhanced results with:
- `has_summary`: Boolean indicating if file has summary
- `file_summary`: Structured summary with purpose, pattern, domain, exports

### Multi-Codebase Search
Search across specific codebases or all at once:
```
memory_search(query="api endpoints", codebase="backend")  # Specific codebase
memory_search(query="api endpoints")                      # All codebases
```

### Performance Monitoring
Track summarization progress with time estimates:
```
memory_summarization_status()  # Returns timing estimates and queue status
```

### Verification Search Mode (NEW)
Use `search_mode="verify"` for "does X use pattern Y?" queries:
```
memory_search(
    query="verify _generate_features uses window-relative bar_index for DataFrame access",
    search_mode="verify"
)
```

Returns structured verification result:
```json
{
  "search_mode": "verify",
  "subject": {"name": "_generate_features", "file": "src/strategy.py", "found": true},
  "verification": {
    "status": "SUPPORTED",
    "confidence": 0.92,
    "evidence": [{"type": "subscript_access", "detail": "df.iloc[bar_index]", "line": 631}]
  },
  "summary": "VERIFIED: _generate_features uses bar_index as window-relative index."
}
```

Supported query patterns:
- `"verify X uses Y"`, `"does X use Y"`, `"is X using Y"`
- `"does X call Y"`, `"does X access Y"`, `"does X have Y"`
- `"check if X ..."`, `"confirm X ..."`

### Method Call Graph (NEW)
Query method-to-method relationships using `memory_method_relationships`:
```
memory_method_relationships(
    method="_generate_features",
    codebase="options-ml-trader",
    relationship="all"  # "callers", "callees", or "all"
)
```

Returns:
```json
{
  "method": "_generate_features",
  "codebase": "options-ml-trader",
  "callers": [{"name": "SwingStrategy.run", "file": "src/strategy.py", "line": 45}],
  "callees": [{"name": "FeatureGenerator.fit", "file": "src/features.py", "line": 123}],
  "stats": {"caller_count": 1, "callee_count": 1}
}
```

Use cases:
- **Find what calls a method**: `relationship="callers"`
- **Find what a method calls**: `relationship="callees"`
- **Understand call chains**: Combine with transitive queries

## Next Features (Roadmap)

### Conversation Memory Integration
- **Context Persistence**: Automatically store important conversation context
- **Decision Tracking**: Link code changes to architectural decisions
- **Learning Integration**: Connect debugging sessions to code improvements

### Enhanced Search Intelligence  
- ~~**Semantic Code Relationships**: Understand function call graphs and dependencies~~ ✅ IMPLEMENTED (Phase 4)
- **Pattern Recognition**: Identify similar code patterns across codebases
- **Change Impact Analysis**: Predict which files might be affected by changes

### Advanced Summarization
- **Incremental Updates**: Smart re-summarization when files change
- ~~**Cross-File Context**: Summaries that understand file relationships~~ ✅ IMPLEMENTED (Phase 2 - method_summaries)
- **Custom Prompts**: Domain-specific summarization strategies
