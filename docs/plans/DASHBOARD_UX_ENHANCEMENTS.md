# Dashboard UX Enhancements Plan

**Status:** ✅ Complete
**Created:** 2024-12-25
**Completed:** 2024-12-25
**Author:** Claude + Joshua

## Overview

Enhance the web dashboard to expose new summarization features and improve usability. The dashboard already has strong foundations - this plan focuses on surfacing Phase 2 fields, adding action buttons, and improving the Validate tab.

---

## Current State

### What Already Works Well
- Status tab with summarization progress, time estimates, per-codebase stats
- Browse tab with summary indicators, filters, detail view
- Search tab with "Include Summaries" option
- Validate tab with side-by-side source/summary comparison

### Problems to Fix
1. **"Session" terminology** - Service runs continuously, "session" stats are misleading
2. **"Skipped" is ambiguous** - Includes already-summarized files (not actually skipped)
3. **No action buttons** - Must use MCP tools for invalidate/queue/reindex
4. **Phase 2 fields not visible** - `how_it_works`, `key_mechanisms`, `method_summaries` not shown
5. **Simple file detection not visible** - Can't tell if file was template vs LLM summarized
6. **Validate tab naming** - "Validate" is unclear, really about reviewing summaries

---

## Implementation Plan

### Phase A: Status Tab Improvements

#### Task A1: Fix Summarization Stats Terminology ✅ COMPLETED
**Priority:** High | **Effort:** 30 min | **Completed:** 2024-12-25

**Current Stats:**
- Files completed (this session)
- Files skipped
- Files failed

**New Stats:**
- `Files Summarized` (total in index)
- `Files Pending` (in queue)
- `Simple Files` (template-summarized, no LLM)
- `LLM Summarized` (used LLM)
- `Failed` (only show if > 0)

**Changes:**
- `src/conductor_memory/server/sse.py` - Update stats grid HTML
- `src/conductor_memory/service/memory_service.py` - Add `get_summary_stats()` method returning new breakdown
- API endpoint `/api/summarization` - Return new fields

**Acceptance Criteria:**
- [x] Stats reflect actual index state, not session state
- [x] Simple vs LLM breakdown visible
- [x] "Skipped" field removed entirely
- [x] "Session" terminology removed

---

#### Task A2: Add Action Buttons to Status Tab ✅ COMPLETED
**Priority:** High | **Effort:** 45 min | **Completed:** 2024-12-25

**Add buttons:**
1. **"Queue Summarization"** - Opens modal to select codebase, calls `memory_queue_codebase_summarization`
2. **"Invalidate All Summaries"** - Confirmation dialog, calls `memory_invalidate_summaries`
3. **"Reindex Codebase"** - Opens modal to select codebase, calls `memory_reindex_codebase`

**UI Design:**
```
[Action Buttons Card]
┌─────────────────────────────────────────────────────────┐
│  Actions                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ Queue        │ │ Invalidate   │ │ Reindex      │    │
│  │ Summarization│ │ Summaries    │ │ Codebase     │    │
│  └──────────────┘ └──────────────┘ └──────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Changes:**
- `src/conductor_memory/server/sse.py` - Add buttons HTML, modal dialogs, JS handlers
- New API endpoints or reuse existing MCP tool logic

**Acceptance Criteria:**
- [x] Queue button opens codebase selector, triggers summarization
- [x] Invalidate button shows confirmation, clears all summaries
- [x] Reindex button opens codebase selector, triggers reindex
- [x] Buttons show loading state during operation
- [x] Success/error feedback shown

---

### Phase B: Browse Tab Enhancements

#### Task B1: Show Simple File Indicator ✅ COMPLETED
**Priority:** Medium | **Effort:** 30 min | **Completed:** 2024-12-25

**In file list:**
- Add badge/icon next to files showing summarization type:
  - 🔹 Simple (template) - with reason tooltip (barrel, empty, generated, constants)
  - 📝 LLM Summarized
  - ⏳ Pending (no summary yet)

**In file details:**
- Show "Summarization Method" field: "Simple (barrel_reexport)" or "LLM (qwen2.5-coder:1.5b)"

**Changes:**
- `src/conductor_memory/service/memory_service.py` - Added `simple_file` and `simple_file_reason` to file list and details API responses
- `src/conductor_memory/server/sse.py` - Added CSS for badges, `getSummaryBadge()` and `getSummarizationMethod()` JS helpers, updated file list and detail panel rendering

**Acceptance Criteria:**
- [x] File list shows visual indicator for simple vs LLM
- [x] Tooltip shows reason for simple files
- [x] Detail view shows summarization method

---

#### Task B2: Show Phase 2 Fields in File Details ✅ COMPLETED
**Priority:** Medium | **Effort:** 45 min | **Completed:** 2024-12-25

**Add to summary section in file details:**

```
┌─ Summary ──────────────────────────────────────────────┐
│ Purpose: Handles user authentication and session mgmt  │
│ Pattern: Service    Domain: auth                       │
│                                                        │
│ ▼ How It Works                                         │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Uses bcrypt for password hashing. Sessions are     │ │
│ │ stored in Redis with 24-hour TTL. JWT tokens are   │ │
│ │ signed with RS256 algorithm...                     │ │
│ └────────────────────────────────────────────────────┘ │
│                                                        │
│ Key Mechanisms: [caching] [retry-logic] [rate-limit]   │
│                                                        │
│ ▼ Method Summaries (5)                                 │
│   authenticate() - Validates credentials against DB    │
│   create_session() - Creates new session in Redis      │
│   refresh_token() - Issues new JWT from refresh token  │
│   logout() - Invalidates session and revokes token     │
│   validate_token() - Verifies JWT signature and expiry │
└────────────────────────────────────────────────────────┘
```

**Changes:**
- `src/conductor_memory/service/memory_service.py` - Updated `get_file_details_async()` to use `get_full_summary()` and include Phase 2 fields (`purpose`, `how_it_works`, `key_mechanisms`, `method_summaries`, `exports`)
- `src/conductor_memory/server/sse.py` - Added CSS for collapsible sections, mechanism tags, method list; Added `renderSummarySection()` and `toggleCollapsible()` JavaScript functions

**Acceptance Criteria:**
- [x] "How It Works" shown as collapsible section
- [x] Key mechanisms shown as tag badges
- [x] Method summaries shown as expandable list
- [x] Graceful handling when fields are missing (old summaries)

---

### Phase C: Validate Tab Rename & Enhancement

#### Task C1: Rename Tab to "Summaries" or "Review" ✅ COMPLETED
**Priority:** High | **Effort:** 15 min | **Completed:** 2024-12-25

**Options:**
- "Summaries" - Clear, describes content
- "Review" - Action-oriented
- "Review Summaries" - Verbose but explicit

**Recommendation:** "Summaries" (short, clear)

**Changes:**
- `src/conductor_memory/server/sse.py` - Updated tab button text and data-tab attribute

**Acceptance Criteria:**
- [x] Tab renamed from "Validate" to "Summaries"
- [x] All references updated (JS, CSS selectors if any)

---

#### Task C2: Show Phase 2 Fields in Summary Panel ✅ COMPLETED
**Priority:** Medium | **Effort:** 45 min | **Completed:** 2024-12-25

**Current summary panel shows:**
- Purpose
- Pattern/Domain tags
- Exports

**Add:**
- "How It Works" section (collapsible)
- Key Mechanisms as tags
- Method Summaries as list

**Layout:**
```
┌─ Summary (qwen2.5-coder:1.5b) ─────────────────────────┐
│                                                        │
│ Purpose                                                │
│ Handles user authentication and session management    │
│                                                        │
│ [Service] [auth]                                       │
│                                                        │
│ Key Mechanisms                                         │
│ [caching] [retry-logic] [jwt-signing]                  │
│                                                        │
│ ▼ How It Works                                         │
│ Uses bcrypt for password hashing. Sessions stored...   │
│                                                        │
│ ▼ Methods (5)                                          │
│   • authenticate() - Validates credentials             │
│   • create_session() - Creates new Redis session       │
│   • ...                                                │
│                                                        │
│ Exports                                                │
│ AuthService, SessionManager, TokenValidator            │
└────────────────────────────────────────────────────────┘
```

**Changes:**
- `src/conductor_memory/server/sse.py` - Added `renderValidateSummaryPanel()` function that displays Phase 2 fields, updated `loadValidationFileContent()` to use it

**Acceptance Criteria:**
- [x] How It Works section visible and collapsible
- [x] Key Mechanisms displayed as styled tags
- [x] Method summaries displayed as list
- [x] Old summaries without Phase 2 fields still render correctly

---

#### Task C3: Show Simple File Indicator ✅ COMPLETED
**Priority:** Medium | **Effort:** 20 min | **Completed:** 2024-12-25

**In file list and summary panel:**
- Show badge: "Simple (barrel)" vs "LLM"
- Different styling for simple files (maybe grayed out or different icon)

**Decision:** Simple files appear in validation queue but are marked as "auto-approved" (per plan decision)

**Changes:**
- `src/conductor_memory/service/memory_service.py` - Added `simple_file` to validation queue API response
- `src/conductor_memory/server/sse.py` - Added CSS for simple file styling, updated `renderValidateFileList()` to show 🔹 badge for simple files, updated `loadValidationFileContent()` to show "Simple (reason)" badge in summary panel header

**Acceptance Criteria:**
- [x] Simple files clearly marked in validation queue (🔹 badge, slightly dimmed)
- [x] Simple files marked as auto-approved (decision per plan)
- [x] Summary panel shows "Simple (reason)" badge with pattern info

---

#### Task C4: Add Summary Statistics ✅ COMPLETED
**Priority:** Low | **Effort:** 1 hour | **Completed:** 2024-12-25

**Add stats section at top of Summaries tab:**

```
┌─ Summary Statistics ───────────────────────────────────┐
│                                                        │
│  By Pattern          By Domain          By Status      │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐ │
│  │ Service 45 │     │ auth    32 │     │ ✓ Appr 120│ │
│  │ Utility 38 │     │ api     28 │     │ ✗ Rej   15│ │
│  │ Model   25 │     │ core    24 │     │ ● Pend  89│ │
│  │ Config  18 │     │ db      20 │     │           │ │
│  │ ...        │     │ ...        │     │           │ │
│  └────────────┘     └────────────┘     └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Changes:**
- `src/conductor_memory/storage/chroma.py` - Updated `get_summary_stats()` to include `by_status` validation status distribution
- `src/conductor_memory/service/memory_service.py` - Added `get_summary_statistics_async()` method that aggregates stats across codebases
- `src/conductor_memory/server/sse.py` - Added `/api/summary-stats` endpoint, collapsible stats section HTML, CSS for 3-column stats layout, JavaScript to load/render stats

**Acceptance Criteria:**
- [x] Pattern distribution shown
- [x] Domain distribution shown
- [x] Approval status breakdown shown
- [x] Stats update when codebase filter changes

---

### Phase D: Search Tab Enhancements

#### Task D1: Show Summary Preview in Search Results ✅ COMPLETED
**Priority:** Medium | **Effort:** 30 min | **Completed:** 2024-12-25

**When "Include Summaries" is checked, show in each result:**
- Purpose snippet (truncated to ~100 chars)
- Pattern/Domain badges
- "Has How It Works" / "Has Methods" indicators

**Changes:**
- `src/conductor_memory/server/sse.py` - Added CSS for `.result-summary-preview`, added `renderSummaryPreview()` JS function, updated `renderSearchResults()` to conditionally render summary preview

**Acceptance Criteria:**
- [x] Summary fields visible in search results when enabled
- [x] Compact display that doesn't overwhelm result list

---

## API Changes Required

| Endpoint | Change |
|----------|--------|
| `GET /api/summarization` | ✅ Add `simple_count`, `llm_count`; remove `skipped`, `files_completed` (session fields) |
| `GET /api/file-summary` | Ensure returns `how_it_works`, `key_mechanisms`, `method_summaries`, `simple_file`, `simple_file_reason` |
| `POST /api/queue-summarization` | ✅ New - trigger summarization for codebase |
| `POST /api/invalidate-summaries` | ✅ New - clear all summaries |
| `POST /api/reindex` | ✅ New - trigger reindex for codebase |
| `GET /api/summary-stats` | New - aggregated summary statistics |

---

## Task Summary

| Phase | Task | Priority | Effort | Description | Status |
|-------|------|----------|--------|-------------|--------|
| A | A1 | High | 30 min | Fix summarization stats terminology | ✅ |
| A | A2 | High | 45 min | Add action buttons (queue, invalidate, reindex) | ✅ |
| B | B1 | Medium | 30 min | Show simple file indicator in Browse | ✅ |
| B | B2 | Medium | 45 min | Show Phase 2 fields in file details | ✅ |
| C | C1 | High | 15 min | Rename Validate tab to Summaries | ✅ |
| C | C2 | Medium | 45 min | Show Phase 2 fields in summary panel | ✅ |
| C | C3 | Medium | 20 min | Show simple file indicator in Summaries tab | ✅ |
| C | C4 | Low | 1 hour | Add summary statistics | ✅ |
| D | D1 | Medium | 30 min | Show summary preview in search results | ✅ |

**Total Estimated Effort:** ~5.5 hours

---

## Deferred Items (Future Enhancements)

Record in TODO.md:
- Call graph explorer - Visualize method relationships
- Dependency graph - Visualize file import relationships
- File centrality view - Show most important files by PageRank

---

## Decisions

1. **Stats tab "skipped" field** - **Remove entirely**
2. **Simple files in validation queue** - **Show but mark as "auto-approved"**
3. **Tab name** - **"Summaries"**
