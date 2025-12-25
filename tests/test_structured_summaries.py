#!/usr/bin/env python3
"""
Test Structured Summary Storage (Task E2)

Tests for the structured JSON storage and retrieval of file summaries,
including Phase 2 fields (how_it_works, key_mechanisms, method_summaries).

Test Categories:
1. Full FileSummary stored as JSON - verify complete summary dict stored in ChromaDB
2. Phase 2 fields retrieved correctly - how_it_works, key_mechanisms, method_summaries
3. Simple file summaries stored correctly - simple_file=True and simple_file_reason
4. Metadata filtering works - has_how_it_works, has_method_summaries boolean flags
5. Backwards compatibility during migration - old format (just file path) still works
"""

import sys
import os
import json
import tempfile
import shutil
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for Chroma storage."""
    temp_dir = tempfile.mkdtemp(prefix="test_structured_summaries_")
    yield temp_dir
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@pytest.fixture
def chroma_client(temp_chroma_dir):
    """Create a Chroma client in a temporary directory."""
    import chromadb
    return chromadb.PersistentClient(path=temp_chroma_dir)


@pytest.fixture
def summary_index(chroma_client):
    """Create a SummaryIndexMetadata instance for testing."""
    from conductor_memory.storage.chroma import SummaryIndexMetadata
    return SummaryIndexMetadata(chroma_client, "test_summary_index")


# ============================================================================
# TEST DATA
# ============================================================================

FULL_SUMMARY_DATA = {
    "file_path": "src/service/user_service.py",
    "language": "python",
    "purpose": "Handles user authentication and session management",
    "pattern": "Service",
    "key_exports": ["UserService", "AuthError", "SessionManager"],
    "dependencies": ["bcrypt", "jwt", "redis"],
    "domain": "authentication",
    "model_used": "qwen-2.5-coder",
    "tokens_used": 250,
    "response_time_ms": 1500.5,
    "is_skeleton": False,
    "error": None,
    # Phase 2 fields
    "how_it_works": "Uses bcrypt for password hashing with configurable work factor. "
                   "JWT tokens are generated with expiry and stored in Redis for session tracking. "
                   "Session validation checks both token signature and Redis presence.",
    "key_mechanisms": [
        "password-hashing",
        "jwt-token-caching",
        "redis-session-store",
        "lazy-initialization"
    ],
    "method_summaries": {
        "authenticate": "Validates credentials against stored bcrypt hash, returns JWT on success",
        "create_session": "Generates JWT with user claims, stores in Redis with TTL",
        "validate_session": "Checks JWT signature and Redis session existence",
        "logout": "Removes session from Redis, adds token to blacklist"
    },
    "simple_file": False,
    "simple_file_reason": None
}

SIMPLE_FILE_SUMMARY_DATA = {
    "file_path": "src/utils/__init__.py",
    "language": "python",
    "purpose": "Re-exports 5 symbols from submodules: validators, formatters, helpers",
    "pattern": "Barrel",
    "key_exports": ["validate_email", "format_date", "slugify", "sanitize_html", "parse_json"],
    "dependencies": ["validators", "formatters", "helpers"],
    "domain": "infrastructure",
    "model_used": "template",
    "tokens_used": 0,
    "response_time_ms": 0.5,
    "is_skeleton": False,
    "error": None,
    "simple_file": True,
    "simple_file_reason": "barrel_reexport"
}

MINIMAL_SUMMARY_DATA = {
    "file_path": "src/config.py",
    "language": "python",
    "purpose": "Application configuration settings",
    "pattern": "Constants",
    "key_exports": ["CONFIG"],
    "dependencies": [],
    "domain": "configuration",
    "model_used": "llama3"
    # No Phase 2 fields (how_it_works, key_mechanisms, method_summaries)
}


# ============================================================================
# TEST CLASS: Full FileSummary Stored as JSON
# ============================================================================

class TestFullSummaryStoredAsJSON:
    """Test that the complete FileSummary dict is stored in ChromaDB document field."""
    
    def test_store_summary_creates_json_document(self, summary_index):
        """Verify that store_summary stores the full dict as JSON in document field."""
        summary_index.store_summary(
            file_path="src/service/user_service.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc123"
        )
        
        # Directly query the collection to check document content
        result = summary_index.collection.get(
            ids=["src/service/user_service.py"],
            include=["documents"]
        )
        
        assert result["ids"], "Summary should be stored"
        assert result["documents"], "Document should be present"
        
        # Parse the document as JSON
        stored_json = result["documents"][0]
        parsed = json.loads(stored_json)
        
        # Verify all fields are present
        assert parsed["purpose"] == FULL_SUMMARY_DATA["purpose"]
        assert parsed["pattern"] == FULL_SUMMARY_DATA["pattern"]
        assert parsed["key_exports"] == FULL_SUMMARY_DATA["key_exports"]
        assert parsed["domain"] == FULL_SUMMARY_DATA["domain"]
    
    def test_store_summary_includes_all_fields(self, summary_index):
        """Verify that all fields including Phase 2 are stored in the JSON document."""
        summary_index.store_summary(
            file_path="src/service/user_service.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc123"
        )
        
        result = summary_index.collection.get(
            ids=["src/service/user_service.py"],
            include=["documents"]
        )
        
        parsed = json.loads(result["documents"][0])
        
        # Check all expected fields
        expected_fields = [
            "file_path", "language", "purpose", "pattern", "key_exports",
            "dependencies", "domain", "model_used", "tokens_used",
            "response_time_ms", "is_skeleton", "error",
            "how_it_works", "key_mechanisms", "method_summaries",
            "simple_file", "simple_file_reason"
        ]
        
        for field in expected_fields:
            assert field in parsed, f"Field '{field}' should be in stored JSON"
    
    def test_store_summary_adds_timestamp(self, summary_index):
        """Verify that summarized_at timestamp is added if not present."""
        summary_data = {**FULL_SUMMARY_DATA}
        # Remove timestamp to test auto-addition
        summary_data.pop("summarized_at", None)
        
        summary_index.store_summary(
            file_path="test.py",
            summary_data=summary_data,
            content_hash="xyz789"
        )
        
        result = summary_index.collection.get(
            ids=["test.py"],
            include=["documents"]
        )
        
        parsed = json.loads(result["documents"][0])
        
        assert "summarized_at" in parsed, "summarized_at should be auto-added"
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(parsed["summarized_at"])
    
    def test_store_summary_preserves_timestamp(self, summary_index):
        """Verify that existing summarized_at timestamp is preserved."""
        timestamp = "2024-12-24T10:30:00.000000"
        summary_data = {**FULL_SUMMARY_DATA, "summarized_at": timestamp}
        
        summary_index.store_summary(
            file_path="test.py",
            summary_data=summary_data,
            content_hash="xyz789"
        )
        
        result = summary_index.collection.get(
            ids=["test.py"],
            include=["documents"]
        )
        
        parsed = json.loads(result["documents"][0])
        
        assert parsed["summarized_at"] == timestamp


# ============================================================================
# TEST CLASS: Phase 2 Fields Retrieved Correctly
# ============================================================================

class TestPhase2FieldsRetrieved:
    """Test that how_it_works, key_mechanisms, method_summaries are returned correctly."""
    
    def test_get_full_summary_returns_how_it_works(self, summary_index):
        """Verify how_it_works is returned by get_full_summary()."""
        summary_index.store_summary(
            file_path="src/feature.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="test123"
        )
        
        result = summary_index.get_full_summary("src/feature.py")
        
        assert result is not None
        assert "how_it_works" in result
        assert result["how_it_works"] == FULL_SUMMARY_DATA["how_it_works"]
    
    def test_get_full_summary_returns_key_mechanisms(self, summary_index):
        """Verify key_mechanisms list is returned by get_full_summary()."""
        summary_index.store_summary(
            file_path="src/feature.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="test123"
        )
        
        result = summary_index.get_full_summary("src/feature.py")
        
        assert result is not None
        assert "key_mechanisms" in result
        assert result["key_mechanisms"] == FULL_SUMMARY_DATA["key_mechanisms"]
        assert isinstance(result["key_mechanisms"], list)
        assert len(result["key_mechanisms"]) == 4
    
    def test_get_full_summary_returns_method_summaries(self, summary_index):
        """Verify method_summaries dict is returned by get_full_summary()."""
        summary_index.store_summary(
            file_path="src/feature.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="test123"
        )
        
        result = summary_index.get_full_summary("src/feature.py")
        
        assert result is not None
        assert "method_summaries" in result
        assert result["method_summaries"] == FULL_SUMMARY_DATA["method_summaries"]
        assert isinstance(result["method_summaries"], dict)
        assert "authenticate" in result["method_summaries"]
        assert "validate_session" in result["method_summaries"]
    
    def test_get_full_summary_returns_all_base_fields(self, summary_index):
        """Verify all base summary fields are returned."""
        summary_index.store_summary(
            file_path="src/feature.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="test123"
        )
        
        result = summary_index.get_full_summary("src/feature.py")
        
        assert result["purpose"] == FULL_SUMMARY_DATA["purpose"]
        assert result["pattern"] == FULL_SUMMARY_DATA["pattern"]
        assert result["domain"] == FULL_SUMMARY_DATA["domain"]
        assert result["key_exports"] == FULL_SUMMARY_DATA["key_exports"]
        assert result["dependencies"] == FULL_SUMMARY_DATA["dependencies"]
    
    def test_get_full_summary_missing_file_returns_none(self, summary_index):
        """Verify get_full_summary returns None for non-existent file."""
        result = summary_index.get_full_summary("nonexistent/file.py")
        
        assert result is None
    
    def test_get_full_summary_merges_metadata_fields(self, summary_index):
        """Verify that metadata fields are merged into the result."""
        summary_index.store_summary(
            file_path="src/feature.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="test_hash_123"
        )
        
        result = summary_index.get_full_summary("src/feature.py")
        
        # These should come from metadata merge
        assert result["content_hash"] == "test_hash_123"
        assert "validation_status" in result
        assert "summarized_at" in result


# ============================================================================
# TEST CLASS: Simple File Summaries Stored Correctly
# ============================================================================

class TestSimpleFileSummaryStorage:
    """Test that simple_file=True and simple_file_reason are stored/retrieved."""
    
    def test_store_simple_file_summary(self, summary_index):
        """Verify simple file summary is stored with simple_file fields."""
        summary_index.store_summary(
            file_path="src/utils/__init__.py",
            summary_data=SIMPLE_FILE_SUMMARY_DATA,
            content_hash="simple123"
        )
        
        result = summary_index.get_full_summary("src/utils/__init__.py")
        
        assert result is not None
        assert result["simple_file"] == True
        assert result["simple_file_reason"] == "barrel_reexport"
    
    def test_simple_file_metadata_flag(self, summary_index):
        """Verify simple_file is stored in metadata for filtering."""
        summary_index.store_summary(
            file_path="src/utils/__init__.py",
            summary_data=SIMPLE_FILE_SUMMARY_DATA,
            content_hash="simple123"
        )
        
        # Check metadata directly
        result = summary_index.collection.get(
            ids=["src/utils/__init__.py"],
            include=["metadatas"]
        )
        
        assert result["metadatas"][0]["simple_file"] == True
    
    def test_simple_file_purpose_and_pattern(self, summary_index):
        """Verify simple file has appropriate purpose and pattern."""
        summary_index.store_summary(
            file_path="src/utils/__init__.py",
            summary_data=SIMPLE_FILE_SUMMARY_DATA,
            content_hash="simple123"
        )
        
        result = summary_index.get_full_summary("src/utils/__init__.py")
        
        assert "Re-exports" in result["purpose"]
        assert result["pattern"] == "Barrel"
        assert result["model_used"] == "template"
    
    def test_non_simple_file_has_false_flag(self, summary_index):
        """Verify regular file has simple_file=False."""
        summary_index.store_summary(
            file_path="src/service.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="regular123"
        )
        
        result = summary_index.get_full_summary("src/service.py")
        
        assert result["simple_file"] == False
        assert result["simple_file_reason"] is None


# ============================================================================
# TEST CLASS: Metadata Filtering Works
# ============================================================================

class TestMetadataFiltering:
    """Test has_how_it_works, has_method_summaries boolean flags for filtering."""
    
    def test_has_how_it_works_true_when_present(self, summary_index):
        """Verify has_how_it_works=True when how_it_works is populated."""
        summary_index.store_summary(
            file_path="src/with_how.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        result = summary_index.collection.get(
            ids=["src/with_how.py"],
            include=["metadatas"]
        )
        
        assert result["metadatas"][0]["has_how_it_works"] == True
    
    def test_has_how_it_works_false_when_missing(self, summary_index):
        """Verify has_how_it_works=False when how_it_works is not present."""
        summary_index.store_summary(
            file_path="src/without_how.py",
            summary_data=MINIMAL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        result = summary_index.collection.get(
            ids=["src/without_how.py"],
            include=["metadatas"]
        )
        
        assert result["metadatas"][0]["has_how_it_works"] == False
    
    def test_has_method_summaries_true_when_present(self, summary_index):
        """Verify has_method_summaries=True when method_summaries is populated."""
        summary_index.store_summary(
            file_path="src/with_methods.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        result = summary_index.collection.get(
            ids=["src/with_methods.py"],
            include=["metadatas"]
        )
        
        assert result["metadatas"][0]["has_method_summaries"] == True
    
    def test_has_method_summaries_false_when_missing(self, summary_index):
        """Verify has_method_summaries=False when method_summaries is not present."""
        summary_index.store_summary(
            file_path="src/without_methods.py",
            summary_data=MINIMAL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        result = summary_index.collection.get(
            ids=["src/without_methods.py"],
            include=["metadatas"]
        )
        
        assert result["metadatas"][0]["has_method_summaries"] == False
    
    def test_query_by_has_how_it_works(self, summary_index):
        """Verify we can query/filter by has_how_it_works."""
        # Store one with how_it_works
        summary_index.store_summary(
            file_path="src/with_how.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        # Store one without
        summary_index.store_summary(
            file_path="src/without_how.py",
            summary_data=MINIMAL_SUMMARY_DATA,
            content_hash="def"
        )
        
        # Query for files with how_it_works
        result = summary_index.collection.get(
            where={"has_how_it_works": True},
            include=["metadatas"]
        )
        
        assert len(result["ids"]) == 1
        assert "src/with_how.py" in result["ids"]
    
    def test_query_by_has_method_summaries(self, summary_index):
        """Verify we can query/filter by has_method_summaries."""
        # Store one with method_summaries
        summary_index.store_summary(
            file_path="src/with_methods.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        # Store one without
        summary_index.store_summary(
            file_path="src/without_methods.py",
            summary_data=MINIMAL_SUMMARY_DATA,
            content_hash="def"
        )
        
        # Query for files with method_summaries
        result = summary_index.collection.get(
            where={"has_method_summaries": True},
            include=["metadatas"]
        )
        
        assert len(result["ids"]) == 1
        assert "src/with_methods.py" in result["ids"]
    
    def test_query_by_pattern(self, summary_index):
        """Verify we can query/filter by pattern metadata."""
        summary_index.store_summary(
            file_path="src/service.py",
            summary_data=FULL_SUMMARY_DATA,  # pattern=Service
            content_hash="abc"
        )
        
        summary_index.store_summary(
            file_path="src/__init__.py",
            summary_data=SIMPLE_FILE_SUMMARY_DATA,  # pattern=Barrel
            content_hash="def"
        )
        
        # Query for Service pattern
        result = summary_index.collection.get(
            where={"pattern": "Service"},
            include=["metadatas"]
        )
        
        assert len(result["ids"]) == 1
        assert "src/service.py" in result["ids"]
    
    def test_query_by_domain(self, summary_index):
        """Verify we can query/filter by domain metadata."""
        summary_index.store_summary(
            file_path="src/auth.py",
            summary_data=FULL_SUMMARY_DATA,  # domain=authentication
            content_hash="abc"
        )
        
        summary_index.store_summary(
            file_path="src/__init__.py",
            summary_data=SIMPLE_FILE_SUMMARY_DATA,  # domain=infrastructure
            content_hash="def"
        )
        
        # Query for authentication domain
        result = summary_index.collection.get(
            where={"domain": "authentication"},
            include=["metadatas"]
        )
        
        assert len(result["ids"]) == 1
        assert "src/auth.py" in result["ids"]


# ============================================================================
# TEST CLASS: Backwards Compatibility During Migration
# ============================================================================

class TestBackwardsCompatibility:
    """Test that old format summaries (document = file_path) still work."""
    
    def test_old_format_document_is_file_path(self, summary_index):
        """Simulate old format where document was just the file path."""
        # Manually insert old format entry directly into collection
        summary_index.collection.upsert(
            ids=["src/old_format.py"],
            documents=["src/old_format.py"],  # Old format: document = file path
            metadatas=[{
                "content_hash": "old_hash",
                "model": "old_model",
                "pattern": "Utility",
                "domain": "legacy"
            }]
        )
        
        # get_full_summary should handle this gracefully
        result = summary_index.get_full_summary("src/old_format.py")
        
        assert result is not None
        # In old format, purpose comes from document (which was the path)
        assert result["purpose"] == "src/old_format.py"
        # Metadata fields should still be accessible
        assert result["pattern"] == "Utility"
        assert result["domain"] == "legacy"
    
    def test_old_format_with_plain_text_purpose(self, summary_index):
        """Simulate old format where document was plain text (purpose string)."""
        # Manually insert old format entry
        summary_index.collection.upsert(
            ids=["src/text_format.py"],
            documents=["This is a utility file for string processing."],  # Plain text
            metadatas=[{
                "content_hash": "text_hash",
                "model": "gpt-4",
                "pattern": "Utility",
                "domain": "strings"
            }]
        )
        
        # get_full_summary should handle this gracefully
        result = summary_index.get_full_summary("src/text_format.py")
        
        assert result is not None
        # Purpose should be the plain text content
        assert result["purpose"] == "This is a utility file for string processing."
        assert result["pattern"] == "Utility"
        assert result["domain"] == "strings"
    
    def test_old_format_no_phase2_fields(self, summary_index):
        """Verify old format entries don't have Phase 2 fields (graceful handling)."""
        # Manually insert old format entry
        summary_index.collection.upsert(
            ids=["src/old_no_phase2.py"],
            documents=["Old format summary text"],
            metadatas=[{
                "content_hash": "hash",
                "model": "llama2"
            }]
        )
        
        result = summary_index.get_full_summary("src/old_no_phase2.py")
        
        assert result is not None
        # Phase 2 fields should not cause errors
        # They'll either be missing or have default values from metadata merge
        assert result.get("has_how_it_works", False) == False
        assert result.get("has_method_summaries", False) == False
    
    def test_update_summary_info_preserves_json_document(self, summary_index):
        """Verify update_summary_info() doesn't overwrite structured JSON document."""
        # First store a proper structured summary
        summary_index.store_summary(
            file_path="src/preserved.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="original"
        )
        
        # Now call the legacy update_summary_info method
        summary_index.update_summary_info(
            file_path="src/preserved.py",
            content_hash="updated_hash",
            summary_chunk_id="chunk_123",
            model="new_model",
            pattern="NewPattern",
            domain="new_domain"
        )
        
        # The JSON document should still be preserved
        result = summary_index.get_full_summary("src/preserved.py")
        
        # Check that the structured data is still there
        assert result["how_it_works"] == FULL_SUMMARY_DATA["how_it_works"]
        assert result["key_mechanisms"] == FULL_SUMMARY_DATA["key_mechanisms"]
        assert result["method_summaries"] == FULL_SUMMARY_DATA["method_summaries"]
        
        # But metadata should be updated
        assert result["content_hash"] == "updated_hash"
    
    def test_update_summary_metadata_preserves_json_document(self, summary_index):
        """Verify update_summary_metadata() doesn't overwrite structured JSON document."""
        # First store a proper structured summary
        summary_index.store_summary(
            file_path="src/meta_test.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="original"
        )
        
        # Now call update_summary_metadata
        summary_index.update_summary_metadata(
            file_path="src/meta_test.py",
            metadata={"validation_status": "approved", "custom_field": "custom_value"}
        )
        
        # The JSON document should still be preserved
        result = summary_index.get_full_summary("src/meta_test.py")
        
        # Check that the structured data is still there
        assert result["how_it_works"] == FULL_SUMMARY_DATA["how_it_works"]
        assert result["purpose"] == FULL_SUMMARY_DATA["purpose"]
        
        # And metadata should be updated
        assert result["validation_status"] == "approved"


# ============================================================================
# TEST CLASS: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_how_it_works_treated_as_missing(self, summary_index):
        """Empty string how_it_works should result in has_how_it_works=False."""
        summary_data = {**MINIMAL_SUMMARY_DATA, "how_it_works": ""}
        
        summary_index.store_summary(
            file_path="src/empty_how.py",
            summary_data=summary_data,
            content_hash="abc"
        )
        
        result = summary_index.collection.get(
            ids=["src/empty_how.py"],
            include=["metadatas"]
        )
        
        # Empty string should evaluate to False in bool()
        assert result["metadatas"][0]["has_how_it_works"] == False
    
    def test_empty_method_summaries_treated_as_missing(self, summary_index):
        """Empty dict method_summaries should result in has_method_summaries=False."""
        summary_data = {**MINIMAL_SUMMARY_DATA, "method_summaries": {}}
        
        summary_index.store_summary(
            file_path="src/empty_methods.py",
            summary_data=summary_data,
            content_hash="abc"
        )
        
        result = summary_index.collection.get(
            ids=["src/empty_methods.py"],
            include=["metadatas"]
        )
        
        # Empty dict should evaluate to False in bool()
        assert result["metadatas"][0]["has_method_summaries"] == False
    
    def test_unicode_in_summary_data(self, summary_index):
        """Verify unicode characters in summary data are handled correctly."""
        summary_data = {
            **MINIMAL_SUMMARY_DATA,
            "purpose": "Handles i18n: 你好世界, émojis 🎉, and special chars: äöü",
            "how_it_works": "Uses UTF-8 encoding for all strings → properly"
        }
        
        summary_index.store_summary(
            file_path="src/unicode.py",
            summary_data=summary_data,
            content_hash="unicode123"
        )
        
        result = summary_index.get_full_summary("src/unicode.py")
        
        assert "你好世界" in result["purpose"]
        assert "🎉" in result["purpose"]
        assert "→" in result["how_it_works"]
    
    def test_very_long_how_it_works(self, summary_index):
        """Verify very long how_it_works text is stored correctly."""
        long_text = "This explains how it works. " * 500  # ~15KB of text
        
        summary_data = {
            **MINIMAL_SUMMARY_DATA,
            "how_it_works": long_text
        }
        
        summary_index.store_summary(
            file_path="src/long_how.py",
            summary_data=summary_data,
            content_hash="long123"
        )
        
        result = summary_index.get_full_summary("src/long_how.py")
        
        assert result["how_it_works"] == long_text
    
    def test_many_method_summaries(self, summary_index):
        """Verify many method summaries are stored correctly."""
        method_summaries = {f"method_{i}": f"Does thing {i}" for i in range(50)}
        
        summary_data = {
            **MINIMAL_SUMMARY_DATA,
            "method_summaries": method_summaries
        }
        
        summary_index.store_summary(
            file_path="src/many_methods.py",
            summary_data=summary_data,
            content_hash="many123"
        )
        
        result = summary_index.get_full_summary("src/many_methods.py")
        
        assert len(result["method_summaries"]) == 50
        assert result["method_summaries"]["method_25"] == "Does thing 25"
    
    def test_store_summary_upserts(self, summary_index):
        """Verify storing same file path twice updates rather than duplicates."""
        summary_index.store_summary(
            file_path="src/upsert.py",
            summary_data={**MINIMAL_SUMMARY_DATA, "purpose": "First version"},
            content_hash="v1"
        )
        
        summary_index.store_summary(
            file_path="src/upsert.py",
            summary_data={**MINIMAL_SUMMARY_DATA, "purpose": "Second version"},
            content_hash="v2"
        )
        
        result = summary_index.get_full_summary("src/upsert.py")
        
        assert result["purpose"] == "Second version"
        assert result["content_hash"] == "v2"


# ============================================================================
# TEST CLASS: get_full_summary Boolean Flags from Metadata
# ============================================================================

class TestGetFullSummaryMetadataFlags:
    """Test that get_full_summary correctly returns boolean flags from metadata."""
    
    def test_has_how_it_works_in_result(self, summary_index):
        """Verify has_how_it_works is included in get_full_summary result."""
        summary_index.store_summary(
            file_path="src/test.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        result = summary_index.get_full_summary("src/test.py")
        
        assert "has_how_it_works" in result
        assert result["has_how_it_works"] == True
    
    def test_has_method_summaries_in_result(self, summary_index):
        """Verify has_method_summaries is included in get_full_summary result."""
        summary_index.store_summary(
            file_path="src/test.py",
            summary_data=FULL_SUMMARY_DATA,
            content_hash="abc"
        )
        
        result = summary_index.get_full_summary("src/test.py")
        
        assert "has_method_summaries" in result
        assert result["has_method_summaries"] == True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Task E2: Structured Summary Storage Tests")
    print("=" * 70)
    print("\nTest Categories:")
    print("  1. Full FileSummary stored as JSON")
    print("  2. Phase 2 fields retrieved correctly")
    print("  3. Simple file summaries stored correctly")
    print("  4. Metadata filtering works")
    print("  5. Backwards compatibility during migration")
    print("\n" + "=" * 70)
    
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
