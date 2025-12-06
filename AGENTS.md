# AGENTS.md - Memory-Augmented Code Assistant Guidelines

This file defines how AI coding assistants should leverage the conductor-memory MCP server for efficient codebase exploration and context retrieval.

## Memory MCP Tools Available

| Tool | Purpose |
|------|---------|
| `memory_search` | Semantic/keyword search across indexed codebases |
| `memory_store` | Store conversation context for later retrieval |
| `memory_store_decision` | Store architectural decisions (auto-pinned) |
| `memory_store_lesson` | Store debugging insights/lessons learned (auto-pinned) |
| `memory_status` | Check indexing status and codebase info |
| `memory_delete` | Remove outdated memories by ID |
| `memory_prune` | Clean up old unpinned memories |

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

**Query Tips:**
- Use multiple related terms: `"feed timeline posts list"` not just `"feed"`
- Include technical terms: `"repository pattern data layer"` 
- Try variations if first query returns nothing
- Specify codebase parameter for multi-codebase setups

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
