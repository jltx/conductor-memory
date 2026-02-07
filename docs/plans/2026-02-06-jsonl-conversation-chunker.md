# JSONL Conversation Chunker Implementation

**Date:** 2026-02-06
**Status:** Ready for implementation
**Purpose:** Add conversation-aware chunking for Claude Code session files.

## Overview

Add a `ConversationParser` that understands Claude Code JSONL session format, creating semantic chunks at message boundaries with conversation metadata.

## Implementation Steps

### 1. Add ChunkingStrategy enum value

**File:** `src/conductor_memory/search/chunking.py`

```python
class ChunkingStrategy(str, Enum):
    # ... existing ...
    JSONL_CONVERSATION = "jsonl_conversation"
```

### 2. Create ConversationParser

**File:** `src/conductor_memory/search/parsers/conversation_parser.py`

```python
class ConversationParser(ContentParser):
    """Parser for Claude Code JSONL conversation files."""

    def supports(self, file_path: str) -> bool:
        return file_path.endswith('.jsonl')

    def parse(self, content: str, file_path: str, commit_hash: Optional[str] = None) -> List[Tuple[str, ChunkMetadata]]:
        # Parse JSONL line by line
        # Group user message + assistant response into chunks
        # Extract metadata: session_id, project (cwd), timestamp, turn_index
        ...
```

**Chunk structure:**
```
[User]: <user message content>

[Claude]: <assistant response content>
```

**Metadata fields:**
- `session_id` - Extracted from filename
- `project` - Extracted from `cwd` field in JSONL
- `turn_index` - Sequential turn number
- `domain` - Set to "conversation"

**JSONL parsing logic:**
1. Read lines, parse JSON
2. Look for `type: "user"` and `type: "assistant"` messages
3. Skip meta messages (`isMeta: true`), hooks, progress indicators
4. Group consecutive user + assistant into turns
5. Handle tool calls by summarizing: `[Tool: Read file.py]`

### 3. Register parser

**File:** `src/conductor_memory/search/parsers/__init__.py`

```python
from .conversation_parser import ConversationParser

__all__ = ['TreeSitterParser', 'ConversationParser']
```

### 4. Update ChunkingManager

**File:** `src/conductor_memory/search/chunking.py`

In `chunk_text()`, add early check:

```python
def chunk_text(self, text: str, file_path: str, commit_hash: Optional[str] = None):
    # Check for conversation files first
    if file_path.endswith('.jsonl'):
        try:
            from .parsers import ConversationParser
            parser = ConversationParser()
            if parser.supports(file_path):
                return parser.parse(text, file_path, commit_hash)
        except Exception as e:
            logger.warning(f"Conversation parsing failed: {e}, falling back")

    # ... existing logic ...
```

### 5. Add conversation-specific tags

Chunks from conversations should have tags:
- `type:conversation`
- `session:<session_id>`
- `project:<project_name>` (extracted from cwd)
- `turn:<N>`

### 6. Create tests

**File:** `tests/test_conversation_parser.py`

Test cases:
- Basic user/assistant turn parsing
- Multi-turn conversation
- Skip meta messages and hooks
- Handle tool calls in responses
- Extract session_id from filename
- Extract project from cwd field
- Malformed JSONL lines (graceful skip)
- Empty conversation file

### 7. Add example config

**File:** `examples/claude-conversations-config.yaml`

```yaml
codebases:
  - name: claude-conversations
    path: ~/.claude/projects
    extensions: ['.jsonl']
    ignore_patterns: []
    description: Claude Code conversation history
```

## JSONL Message Format Reference

From observed session files:

```json
{"type":"user","message":{"role":"user","content":"..."}}
{"type":"assistant","message":{"role":"assistant","content":"..."}}
{"type":"progress","data":{"type":"hook_progress",...}}  // Skip
{"isMeta":true,...}  // Skip
```

Key fields to extract:
- `type` - "user", "assistant", "progress", etc.
- `message.content` - The actual message text
- `cwd` - Working directory (project path)
- `sessionId` - Session identifier
- `timestamp` - Message timestamp
- `isMeta` - Skip if true

## Testing Commands

```bash
# Run specific tests
pytest tests/test_conversation_parser.py -v

# Test with real session file
python -c "
from conductor_memory.search.parsers import ConversationParser
parser = ConversationParser()
with open('~/.claude/projects/C--Users.../session.jsonl') as f:
    chunks = parser.parse(f.read(), 'session.jsonl')
    for text, meta in chunks[:3]:
        print(f'Turn {meta.start_line}: {text[:100]}...')
"
```

## Success Criteria

1. Conversations are chunked at message boundaries
2. Session/project metadata preserved in chunks
3. Semantic search returns coherent user+assistant pairs
4. Tool calls summarized (not full JSON)
5. Meta/progress messages skipped
6. Graceful handling of malformed lines
