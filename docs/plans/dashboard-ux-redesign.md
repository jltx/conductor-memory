# Dashboard UX Redesign Plan

**Created:** 2024-12-22
**Status:** Implemented
**Priority:** High
**Completed:** 2024-12-22

## Overview

Redesign the Search and Browse tabs in the web dashboard (`sse.py`) to improve usability, information density, and workflow efficiency.

---

## Part 1: Browse Tab Redesign

### Current Problems
1. Details panel is at bottom of page - requires scrolling multiple screens after clicking a row
2. No visual connection between selected row and details
3. Filter only supports path text matching
4. Pagination disconnected from content
5. No way to validate or regenerate LLM summaries

### Solution: Master-Detail Split Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Browse Data                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ [View: Files ▼] [Codebase ▼] [Has Summary ▼] [Pattern ▼] [Filter... 🔍] │
├────────────────────────────────┬────────────────────────────────────────┤
│ FILE LIST (scrollable)         │ DETAILS PANEL (scrollable)             │
│ ─────────────────────────────  │ ────────────────────────────────────── │
│                                │                                        │
│ ▸ src/core/models.py      [4] │ src/core/models.py                     │
│   src/core/vector.py      [2] │ ══════════════════                     │
│ ● src/search/hybrid.py    [6] │                                        │
│   src/server/sse.py      [12] │ Codebase: conductor-memory             │
│   src/config/server.py    [3] │ Chunks: 6  |  Indexed: 12/22 3:45 PM   │
│   ...                         │ Hash: a3f2b1c9...                      │
│                               │                                        │
│                               │ ┌─ Summary (qwen2.5-coder:1.5b) ─────┐ │
│                               │ │ Hybrid search engine combining     │ │
│                               │ │ semantic vector search with BM25   │ │
│                               │ │ keyword matching using RRF...      │ │
│                               │ └────────────────────────────────────┘ │
│                               │                                        │
│                               │ Chunks ──────────────────────────────  │
│                               │ ┌────────────────────────────────────┐ │
│                               │ │ [1] class HybridSearcher           │ │
│                               │ │ [2] def detect_search_mode         │ │
│                               │ │ [3] class BM25Index                │ │
│                               │ └────────────────────────────────────┘ │
│                               │                                        │
├────────────────────────────────┴────────────────────────────────────────┤
│ Showing 1-50 of 234 files    [< Prev] [1] [2] [3] ... [5] [Next >]      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 1.1 Layout Structure
- [x] Replace single-column layout with CSS Grid/Flexbox split layout
- [x] Left panel: 50% width, file list with scroll area
- [x] Right panel: 50% width, details with scroll
- [ ] Add resizable splitter (optional, nice-to-have)
- [ ] Ensure responsive behavior for smaller screens

#### 1.2 Quick Filters Toolbar
- [x] Add filter dropdowns above the split view:
  - **Has Summary**: All | Yes | No
  - **Pattern**: All | service | utility | model | config | etc.
  - **Domain**: All | api | database | auth | etc.
  - **Language**: All | Python | Kotlin | TypeScript | etc.
- [x] Keep existing text filter for path matching
- [x] Filters apply immediately (no submit button)
- [x] Clear filters button appears when filters active

#### 1.3 File List Improvements
- [x] Compact row design showing: path (truncated), chunk count, summary indicator, pattern tag
- [x] Selected row highlight that persists
- [x] Keyboard navigation (↑/↓ to move, Enter to select)
- [ ] Double-click to open in VS Code (future)

#### 1.4 Details Panel
- [x] Show file metadata at top (codebase, chunks, indexed date, hash)
- [x] Summary section with model info and formatted content
- [x] Expandable chunks with show more/less
- [ ] Copy button for summary and individual chunks

---

## Part 2: Summary Validation Mode

### Purpose
Allow users to review LLM-generated summaries, validate quality, and regenerate bad ones.

### UI Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Summary Validation                                         [Exit Mode]  │
├─────────────────────────────────────────────────────────────────────────┤
│ [Codebase ▼] [Status: Unreviewed ▼]      Progress: 45/234 reviewed     │
├─────────────────────────────────┬────────────────────────────────────────┤
│ ORIGINAL FILE CONTENT           │ LLM SUMMARY                           │
│ ─────────────────────────────── │ ────────────────────────────────────── │
│                                 │                                        │
│ """                             │ Model: qwen2.5-coder:1.5b              │
│ Hybrid search engine combining  │ Generated: 12/22/2024 2:30 PM          │
│ semantic and keyword search.    │ ──────────────────────────────         │
│ """                             │                                        │
│                                 │ Purpose: Provides hybrid search        │
│ import numpy as np              │ combining semantic vector similarity   │
│ from rank_bm25 import BM25Okapi │ with BM25 keyword matching.           │
│                                 │                                        │
│ class HybridSearcher:           │ Pattern: service                       │
│     """                         │ Domain: search                         │
│     Combines vector similarity  │                                        │
│     with BM25 keyword search    │ Exports:                               │
│     using Reciprocal Rank       │ - HybridSearcher (class)               │
│     Fusion.                     │ - BM25Index (class)                    │
│     """                         │ - detect_search_mode (function)        │
│     ...                         │                                        │
│                                 │                                        │
├─────────────────────────────────┴────────────────────────────────────────┤
│                                                                          │
│   [← Previous]    [✓ Approve]    [✗ Reject]    [🔄 Regenerate]   [Next →]│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 2.1 Validation Mode Toggle
- [x] Add "Validate" tab to main navigation
- [x] Separate validation view with dedicated layout
- [x] Return to browse by switching tabs

#### 2.2 Three-Column Layout (Redesigned 2024-12-22)
- [x] Left panel: File list with pagination and status icons
- [x] Middle panel: Original file content (read-only, scrollable)
- [x] Right panel: Summary with metadata
- [ ] Synchronized scrolling (optional, nice-to-have)

#### 2.3 Validation Actions
- [x] **Approve** (✓): Mark summary as validated, move to next
- [x] **Reject** (✗): Mark summary as rejected, move to next
- [x] **Regenerate** (↻): Call LLM to regenerate summary, show new result
- [x] **Skip**: Move to next without marking
- [x] Files disappear from "Unreviewed" list after approval/rejection

#### 2.4 Navigation & Progress
- [x] Previous/Next buttons (with page boundary crossing)
- [x] Progress indicator: "X reviewed / Y total"
- [x] Filter by status: All | Unreviewed | Approved | Rejected
- [x] Filter by pattern and domain
- [x] Configurable page size (20/30/40/50)
- [x] File list with click-to-select navigation
- [x] Keyboard shortcuts (←/→ for prev/next, ↑/↓ for list, A/R/S for actions)

#### 2.5 Backend Support
- [x] Add `validation_status` field to summary metadata
- [x] API endpoint: `POST /api/summary/validate` with status
- [x] API endpoint: `POST /api/summary/regenerate` to re-run LLM
- [x] API endpoint: `GET /api/validation-queue` with pagination and filters
- [x] API endpoint: `GET /api/file-content` for raw file
- [ ] Store validation history (who, when, old vs new)

---

## Part 3: Search Tab Improvements

### Current Problems
1. Results lack visual hierarchy - hard to distinguish code vs decisions vs lessons
2. Raw relevance score meaningless to users
3. Advanced filters are hidden and confusing
4. No copy functionality
5. Content preview is plain text without structure

### Solution: Enhanced Results with Better Organization

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Search Memory                                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ 🔍 [search query here...                                     ] [Go] │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│ ┌─ Filters ─────────────────────────────────────────────────────────┐   │
│ │ Codebase: [All ▼]  Mode: [Auto ▼]  Results: [10 ▼]               │   │
│ │                                                                    │   │
│ │ Types: [✓] Code  [✓] Decisions  [✓] Lessons  [ ] Conversations    │   │
│ │                                                                    │   │
│ │ [▸ More Filters]                                                   │   │
│ │   Languages: [python, kotlin    ]  Include Tags: [           ]    │   │
│ │   Classes:   [                  ]  Functions:    [           ]    │   │
│ │   [ ] Has Docstrings  [ ] Has Annotations  [✓] Include Summaries  │   │
│ └────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│ Found 23 results (45ms) • Mode: hybrid                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ ┌─ 🟢 CODE ──────────────────────────────────────────────────────────┐  │
│ │ src/search/hybrid.py                                    ████░ 0.89 │  │
│ │ ────────────────────────────────────────────────────────────────── │  │
│ │ class HybridSearcher:                                       [Copy] │  │
│ │     """Combines vector similarity with BM25 keyword search..."""   │  │
│ │                                                                    │  │
│ │ Tags: search, hybrid, bm25                                         │  │
│ │                                                      [▾ Show more] │  │
│ └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│ ┌─ 🔵 DECISION ──────────────────────────────────────────────────────┐  │
│ │ (pinned)                                                ████░ 0.82 │  │
│ │ ────────────────────────────────────────────────────────────────── │  │
│ │ DECISION: Use hybrid search with RRF fusion                 [Copy] │  │
│ │ RATIONALE: Pure semantic search misses exact matches...            │  │
│ │                                                                    │  │
│ │ Tags: architecture, search                                         │  │
│ └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│ ┌─ 🟡 LESSON ────────────────────────────────────────────────────────┐  │
│ │ (pinned)                                                ███░░ 0.71 │  │
│ │ ────────────────────────────────────────────────────────────────── │  │
│ │ LESSON: BM25 index must be rebuilt after document updates   [Copy] │  │
│ │ SYMPTOM: Search returned stale results after reindexing...         │  │
│ │                                                                    │  │
│ │ Tags: debugging, search, bm25                                      │  │
│ └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 3.1 Filter Reorganization
- [x] Move memory type filters (Code/Decisions/Lessons/Conversations) to prominent position
- [x] Make type filters checkboxes that are always visible
- [x] Group advanced filters under collapsible "More Filters" section
- [x] Add visual indication when filters are active (checkbox styling)

#### 3.2 Result Card Redesign
- [x] Color-coded type badge: green Code, blue Decision, yellow Lesson, gray Conversation
- [x] Visual relevance bar instead of just number (5 filled segments)
- [x] Source path prominent at top (separate row with background)
- [ ] Structured preview based on type:
  - Code: First line of function/class + docstring excerpt
  - Decision: DECISION line + RATIONALE excerpt
  - Lesson: LESSON line + SYMPTOM excerpt
- [x] Copy button per result
- [x] Pinned indicator for decisions/lessons

#### 3.3 Interaction Improvements
- [ ] Keyboard shortcut: `/` or `Ctrl+K` to focus search input
- [x] Enter to search (already works)
- [x] Click result to expand inline (show more/less)
- [x] Copy button copies formatted content

#### 3.4 Search Feedback
- [x] Show which filters are active in results header
- [x] "No results" state with suggestions (try different terms, broaden filters)

---

## Part 4: Shared Components

### 4.1 CSS Variables for Consistency
- [ ] Define color palette as CSS variables
- [ ] Consistent spacing scale
- [ ] Shared button/input styles

### 4.2 Keyboard Navigation
- [ ] Global shortcuts documented in footer or help modal
- [ ] Tab navigation between panels
- [ ] Escape to close modals/panels

### 4.3 Responsive Behavior
- [ ] Breakpoint at 768px for tablet
- [ ] Stack panels vertically on mobile
- [ ] Touch-friendly tap targets

---

## Implementation Order

### Phase 1: Browse Tab Master-Detail (Priority: High)
1. Layout restructure (split view)
2. Details panel in right column
3. File list with selection state
4. Basic keyboard navigation

### Phase 2: Browse Quick Filters (Priority: High)
1. Filter dropdowns
2. Filter state management
3. API support for filtering

### Phase 3: Search Tab Improvements (Priority: Medium)
1. Memory type checkboxes
2. Result card redesign with type badges
3. Copy buttons
4. Filter reorganization

### Phase 4: Summary Validation Mode (Priority: Medium)
1. Side-by-side layout
2. Validation actions (approve/reject)
3. Backend API for validation status
4. Regenerate functionality

### Phase 5: Polish (Priority: Low) ✓
1. ✓ Keyboard shortcuts (/, Ctrl+K for search; ←/→/A/R/S for validation)
2. ✓ Responsive design (mobile breakpoints at 900px and 600px)
3. Loading states (already present)
4. Error handling (already present)

---

## Technical Notes

### File Structure
All changes are in `src/conductor_memory/server/sse.py` which contains:
- HTML template (lines ~280-650)
- CSS styles (inline in `<style>` tag)
- JavaScript (inline in `<script>` tag)

### Considerations
- Keep everything in single file for simplicity (no external assets)
- No external JS libraries - vanilla JS only
- CSS Grid for layout, Flexbox for components
- Use CSS custom properties for theming

### API Endpoints Needed
- `GET /api/files` - already exists, may need filter params
- `GET /api/file-details` - already exists
- `POST /api/summary/validate` - NEW
- `POST /api/summary/regenerate` - NEW
- `GET /api/validation-queue` - NEW (for validation mode)

---

## Success Metrics

1. **Browse efficiency**: User can see file details without scrolling (details always visible)
2. **Filter usage**: Quick filters reduce time to find specific files
3. **Summary validation**: Users can review and validate summaries in a streamlined workflow
4. **Search clarity**: Users can immediately identify result types and relevance
