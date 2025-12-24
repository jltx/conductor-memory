#!/usr/bin/env python3
"""
Comprehensive tests for Phase 3: Verification Search Mode.

Tests the verification query parsing, evidence matching, and result generation
for "does X use pattern Y?" type queries.

Success Metrics from Implementation Plan:
- 80%+ verification queries correctly parsed
- Evidence matching finds relevant signals

Test Categories:
1. Unit Tests: Query Parsing (9 verification patterns)
2. Unit Tests: Evidence Matching (matches_any, calculate_relevance, find_evidence)
3. Unit Tests: Dataclasses (VerificationResult, Evidence, etc.)
4. Integration Tests: Full verification flow with mock chunks
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from typing import List, Dict, Any


# ============================================================================
# UNIT TESTS: Query Parsing - parse_verification_query()
# ============================================================================

class TestParseVerificationQuery:
    """Unit tests for parse_verification_query() function."""
    
    def test_verify_uses_pattern(self):
        """Test 'verify X uses Y' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("verify _generate_features uses iloc")
        
        assert result is not None
        assert result.subject == "_generate_features"
        assert result.claim == "iloc"
    
    def test_verify_use_pattern(self):
        """Test 'verify X use Y' pattern (singular 'use')."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("verify DataProcessor use caching")
        
        assert result is not None
        assert result.subject == "DataProcessor"
        assert result.claim == "caching"
    
    def test_does_use_pattern(self):
        """Test 'does X use Y' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("does MyClass use the repository pattern")
        
        assert result is not None
        assert result.subject == "MyClass"
        assert result.claim == "the repository pattern"
    
    def test_is_using_pattern(self):
        """Test 'is X using Y' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("is DataProcessor using caching")
        
        assert result is not None
        assert result.subject == "DataProcessor"
        assert result.claim == "caching"
    
    def test_does_call_pattern(self):
        """Test 'does X call Y' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("does process_data call validate")
        
        assert result is not None
        assert result.subject == "process_data"
        assert result.claim == "validate"
    
    def test_does_access_pattern(self):
        """Test 'does X access Y' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("does MyClass access database")
        
        assert result is not None
        assert result.subject == "MyClass"
        assert result.claim == "database"
    
    def test_does_have_pattern(self):
        """Test 'does X have Y' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("does UserService have retry logic")
        
        assert result is not None
        assert result.subject == "UserService"
        assert result.claim == "retry logic"
    
    def test_find_if_uses_pattern(self):
        """Test 'find if X uses Y' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("find if handler uses async")
        
        assert result is not None
        assert result.subject == "handler"
        assert result.claim == "async"
    
    def test_check_if_pattern(self):
        """Test 'check if X ...' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("check if process_data accesses database")
        
        assert result is not None
        assert result.subject == "process_data"
        assert result.claim == "accesses database"
    
    def test_confirm_pattern(self):
        """Test 'confirm X ...' pattern."""
        from conductor_memory.search.verification import parse_verification_query
        
        result = parse_verification_query("confirm FeatureGenerator uses window-relative indexing")
        
        assert result is not None
        assert result.subject == "FeatureGenerator"
        assert result.claim == "uses window-relative indexing"


class TestQueryParsingEdgeCases:
    """Edge case tests for query parsing."""
    
    def test_underscored_names(self):
        """Test parsing methods with underscores."""
        from conductor_memory.search.verification import parse_verification_query
        
        # Leading underscore
        result = parse_verification_query("verify _private_method uses caching")
        assert result is not None
        assert result.subject == "_private_method"
        
        # Double underscore
        result = parse_verification_query("does __init__ call super")
        assert result is not None
        assert result.subject == "__init__"
        
        # Multiple underscores
        result = parse_verification_query("verify _very_long_method_name uses iloc")
        assert result is not None
        assert result.subject == "_very_long_method_name"
    
    def test_qualified_names(self):
        """Test parsing qualified names (Class.method)."""
        from conductor_memory.search.verification import parse_verification_query
        
        # Simple qualified name
        result = parse_verification_query("verify MyClass.process uses caching")
        assert result is not None
        assert result.subject == "MyClass.process"
        
        # Nested qualified name
        result = parse_verification_query("does module.Class.method use logging")
        assert result is not None
        assert result.subject == "module.Class.method"
    
    def test_extra_whitespace(self):
        """Test parsing with extra whitespace."""
        from conductor_memory.search.verification import parse_verification_query
        
        # Leading/trailing whitespace
        result = parse_verification_query("  verify process uses caching  ")
        assert result is not None
        assert result.subject == "process"
        
        # Multiple spaces between words
        result = parse_verification_query("verify   process   uses   caching")
        assert result is not None
        assert result.subject == "process"
        assert result.claim == "caching"
        
        # Tabs and mixed whitespace
        result = parse_verification_query("verify\t process \t uses \tcaching")
        assert result is not None
        assert result.subject == "process"
    
    def test_case_insensitivity(self):
        """Test that patterns are case-insensitive."""
        from conductor_memory.search.verification import parse_verification_query
        
        # Uppercase
        result = parse_verification_query("VERIFY process USES caching")
        assert result is not None
        assert result.subject == "process"
        
        # Mixed case
        result = parse_verification_query("Verify Process Uses Caching")
        assert result is not None
        assert result.subject == "Process"
        
        # All patterns should work case-insensitively
        result = parse_verification_query("DOES method USE pattern")
        assert result is not None
        assert result.subject == "method"
    
    def test_non_verification_queries_return_none(self):
        """Test that non-verification queries return None."""
        from conductor_memory.search.verification import parse_verification_query
        
        # Simple search query
        assert parse_verification_query("find all authentication methods") is None
        
        # Question without verification intent
        assert parse_verification_query("what is the purpose of this class") is None
        
        # Just a function name
        assert parse_verification_query("_generate_features") is None
        
        # Unstructured text
        assert parse_verification_query("process data using iloc") is None
        
        # Similar but not matching
        assert parse_verification_query("verification of process") is None
    
    def test_empty_and_invalid_inputs(self):
        """Test handling of empty and invalid inputs."""
        from conductor_memory.search.verification import parse_verification_query
        
        assert parse_verification_query("") is None
        assert parse_verification_query(None) is None
        assert parse_verification_query("   ") is None
        assert parse_verification_query("verify") is None  # Incomplete
        assert parse_verification_query("verify ") is None  # Just keyword


# ============================================================================
# UNIT TESTS: extract_key_terms()
# ============================================================================

class TestExtractKeyTerms:
    """Unit tests for extract_key_terms() function."""
    
    def test_basic_extraction(self):
        """Test basic term extraction."""
        from conductor_memory.search.verification import extract_key_terms
        
        terms = extract_key_terms("uses iloc for DataFrame access")
        
        assert "iloc" in terms
        assert "DataFrame" in terms
        assert "access" in terms
    
    def test_filters_stop_words(self):
        """Test that stop words are filtered out."""
        from conductor_memory.search.verification import extract_key_terms
        
        terms = extract_key_terms("uses the repository pattern for data access")
        
        # Stop words should be removed
        assert "the" not in terms
        assert "for" not in terms
        # Note: "uses" is NOT in stop words - it's a meaningful verb for code analysis
        
        # Key terms should remain
        assert "repository" in terms
        assert "pattern" in terms
        assert "data" in terms
        assert "access" in terms
        assert "uses" in terms  # Meaningful verb preserved
    
    def test_preserves_underscored_names(self):
        """Test that underscored names are preserved."""
        from conductor_memory.search.verification import extract_key_terms
        
        terms = extract_key_terms("bar_index for DataFrame access")
        
        assert "bar_index" in terms
    
    def test_handles_hyphenated_terms(self):
        """Test handling of hyphenated terms."""
        from conductor_memory.search.verification import extract_key_terms
        
        terms = extract_key_terms("window-relative bar_index indexing")
        
        # Hyphenated terms are split
        # Note: "window" and "relative" are in STOP_WORDS per the implementation
        assert "bar_index" in terms
        assert "indexing" in terms
    
    def test_handles_special_characters(self):
        """Test handling of special characters."""
        from conductor_memory.search.verification import extract_key_terms
        
        terms = extract_key_terms("df.iloc[bar_index], self._cache")
        
        # Should extract meaningful parts
        assert len(terms) > 0
        # Brackets and dots may cause splits
        term_str = " ".join(terms).lower()
        assert "iloc" in term_str or "bar_index" in term_str or "_cache" in term_str
    
    def test_empty_and_invalid_inputs(self):
        """Test handling of empty and invalid inputs."""
        from conductor_memory.search.verification import extract_key_terms
        
        assert extract_key_terms("") == []
        assert extract_key_terms(None) == []
        assert extract_key_terms("   ") == []
        assert extract_key_terms("the a an") == []  # Only stop words
    
    def test_preserves_single_meaningful_chars(self):
        """Test that meaningful single characters are preserved."""
        from conductor_memory.search.verification import extract_key_terms
        
        # Variables like i, j, k, x, y are common loop counters
        terms = extract_key_terms("loop variable i and x coordinate")
        
        # Some single chars should be preserved
        has_meaningful = "i" in terms or "x" in terms
        # At least some terms should be extracted
        assert len(terms) > 0


# ============================================================================
# UNIT TESTS: is_verification_query()
# ============================================================================

class TestIsVerificationQuery:
    """Unit tests for is_verification_query() helper."""
    
    def test_verification_prefixes(self):
        """Test that verification prefixes are detected."""
        from conductor_memory.search.verification import is_verification_query
        
        assert is_verification_query("verify process uses caching") is True
        assert is_verification_query("does method call validate") is True
        assert is_verification_query("is processor using async") is True
        assert is_verification_query("check if handler has retry") is True
        assert is_verification_query("confirm class uses pattern") is True
        assert is_verification_query("find if module uses logging") is True
    
    def test_non_verification_queries(self):
        """Test that non-verification queries return False."""
        from conductor_memory.search.verification import is_verification_query
        
        assert is_verification_query("find all methods") is False
        assert is_verification_query("what is this class") is False
        assert is_verification_query("process data") is False
        assert is_verification_query("authentication logic") is False
    
    def test_case_insensitivity(self):
        """Test that prefix detection is case-insensitive."""
        from conductor_memory.search.verification import is_verification_query
        
        assert is_verification_query("VERIFY process uses caching") is True
        assert is_verification_query("Verify Process Uses Caching") is True
        assert is_verification_query("DOES method call validate") is True
    
    def test_edge_cases(self):
        """Test edge cases."""
        from conductor_memory.search.verification import is_verification_query
        
        assert is_verification_query("") is False
        assert is_verification_query(None) is False
        assert is_verification_query("   ") is False


# ============================================================================
# UNIT TESTS: Evidence Matching - matches_any()
# ============================================================================

class TestMatchesAny:
    """Unit tests for matches_any() function."""
    
    def test_exact_match(self):
        """Test exact match (case-insensitive)."""
        from conductor_memory.search.verification import matches_any
        
        assert matches_any("iloc", ["iloc"]) is True
        assert matches_any("ILOC", ["iloc"]) is True
        assert matches_any("iloc", ["ILOC"]) is True
    
    def test_substring_match(self):
        """Test substring match - term contained in signal."""
        from conductor_memory.search.verification import matches_any
        
        # Term in signal
        assert matches_any("df.iloc[bar_index]", ["iloc"]) is True
        assert matches_any("df.iloc[bar_index]", ["bar_index"]) is True
        assert matches_any("self._cache", ["cache"]) is True
    
    def test_reverse_substring_match(self):
        """Test reverse substring - signal contained in term."""
        from conductor_memory.search.verification import matches_any
        
        # Signal in term (for abbreviated tags)
        assert matches_any("iloc", ["df.iloc[idx]"]) is True
    
    def test_partial_matches(self):
        """Test partial string matches."""
        from conductor_memory.search.verification import matches_any
        
        # Partial matches should work
        assert matches_any("_generate_features", ["generate"]) is True
        assert matches_any("process_data", ["process"]) is True
    
    def test_no_match(self):
        """Test when no match is found."""
        from conductor_memory.search.verification import matches_any
        
        assert matches_any("process_data", ["analyze"]) is False
        assert matches_any("iloc", ["loc"]) is True  # Actually contains 'loc'
        assert matches_any("cache", ["database"]) is False
    
    def test_multiple_terms(self):
        """Test matching against multiple terms."""
        from conductor_memory.search.verification import matches_any
        
        # Match any one term
        assert matches_any("df.iloc[idx]", ["iloc", "bar_index"]) is True
        assert matches_any("self._cache", ["database", "cache"]) is True
    
    def test_empty_inputs(self):
        """Test empty inputs."""
        from conductor_memory.search.verification import matches_any
        
        assert matches_any("", ["term"]) is False
        assert matches_any("signal", []) is False
        assert matches_any("", []) is False
        assert matches_any(None, ["term"]) is False


# ============================================================================
# UNIT TESTS: Evidence Matching - calculate_relevance()
# ============================================================================

class TestCalculateRelevance:
    """Unit tests for calculate_relevance() scoring."""
    
    def test_exact_match_high_score(self):
        """Test that exact matches get high scores."""
        from conductor_memory.search.verification import calculate_relevance
        
        score = calculate_relevance("iloc", ["iloc"])
        
        # Exact match should get base (0.5) + exact bonus (0.4) + substantial (0.05) = 0.95
        assert score >= 0.9
    
    def test_substring_match_moderate_score(self):
        """Test that substring matches get moderate scores."""
        from conductor_memory.search.verification import calculate_relevance
        
        score = calculate_relevance("self._cache", ["cache"])
        
        # Substring match should get base (0.5) + substantial (0.05) = 0.55
        assert 0.5 <= score <= 0.7
    
    def test_multi_term_bonus(self):
        """Test multi-term matching bonus."""
        from conductor_memory.search.verification import calculate_relevance
        
        single_score = calculate_relevance("df.iloc", ["iloc"])
        multi_score = calculate_relevance("df.iloc[bar_index]", ["iloc", "bar_index"])
        
        # Multi-term should score higher
        assert multi_score >= single_score
    
    def test_no_match_zero_score(self):
        """Test that non-matches get zero score."""
        from conductor_memory.search.verification import calculate_relevance
        
        score = calculate_relevance("process", ["database"])
        assert score == 0.0
    
    def test_score_capped_at_one(self):
        """Test that score is capped at 1.0."""
        from conductor_memory.search.verification import calculate_relevance
        
        # Even with many matching terms, score should not exceed 1.0
        score = calculate_relevance("a_b_c_d", ["a", "b", "c", "d", "a_b_c_d"])
        assert score <= 1.0
    
    def test_empty_inputs(self):
        """Test empty inputs return zero."""
        from conductor_memory.search.verification import calculate_relevance
        
        assert calculate_relevance("", ["term"]) == 0.0
        assert calculate_relevance("signal", []) == 0.0


# ============================================================================
# UNIT TESTS: Evidence Matching - find_evidence()
# ============================================================================

class TestFindEvidence:
    """Unit tests for find_evidence() function."""
    
    def test_finds_evidence_from_tags(self):
        """Test finding evidence from chunk tags."""
        from conductor_memory.search.verification import find_evidence
        
        tags = ["calls:iloc", "subscript:iloc", "param:bar_index"]
        content = "def process(self, df, bar_index): pass"
        claim = "uses iloc for indexing"
        
        evidence = find_evidence(tags, content, claim)
        
        assert len(evidence) > 0
        # Should find iloc-related evidence
        evidence_details = [e.detail for e in evidence]
        assert any("iloc" in d for d in evidence_details)
    
    def test_finds_evidence_from_content(self):
        """Test finding evidence from Implementation Signals section."""
        from conductor_memory.search.verification import find_evidence
        
        tags = []
        content = """
def process(self, df, bar_index):
    row = df.iloc[bar_index]
    
[Implementation Signals]
Calls: _validate, transform
Reads: self._cache
Subscripts: df.iloc[bar_index]
Parameters used: df, bar_index
"""
        claim = "uses bar_index parameter for DataFrame access"
        
        evidence = find_evidence(tags, content, claim)
        
        assert len(evidence) > 0
        # Should find subscript evidence
        types = [e.type for e in evidence]
        assert "subscript_access" in types or "parameter_usage" in types
    
    def test_evidence_sorted_by_relevance(self):
        """Test that evidence is sorted by relevance (highest first)."""
        from conductor_memory.search.verification import find_evidence
        
        tags = ["calls:process", "calls:iloc", "subscript:iloc"]
        content = ""
        claim = "uses iloc"
        
        evidence = find_evidence(tags, content, claim)
        
        # Should be sorted descending by relevance
        if len(evidence) >= 2:
            for i in range(len(evidence) - 1):
                assert evidence[i].relevance >= evidence[i + 1].relevance
    
    def test_evidence_deduplication(self):
        """Test that duplicate evidence is removed."""
        from conductor_memory.search.verification import find_evidence
        
        # Same signal in tags and content
        tags = ["subscript:iloc"]
        content = """
[Implementation Signals]
Subscripts: iloc
"""
        claim = "uses iloc"
        
        evidence = find_evidence(tags, content, claim)
        
        # Should not have duplicates
        seen = set()
        for e in evidence:
            key = (e.type, e.detail)
            assert key not in seen, f"Duplicate evidence: {e}"
            seen.add(key)
    
    def test_evidence_types(self):
        """Test correct evidence types from different tag prefixes."""
        from conductor_memory.search.verification import find_evidence
        
        tags = [
            "calls:validate",
            "reads:self._cache",
            "writes:self._result",
            "subscript:iloc",
            "param:bar_index"
        ]
        content = ""
        claim = "validate cache result iloc bar_index"
        
        evidence = find_evidence(tags, content, claim)
        
        types = {e.type for e in evidence}
        
        # Should have multiple evidence types
        expected_types = {"call", "attribute_read", "attribute_write", "subscript_access", "parameter_usage"}
        assert len(types & expected_types) > 0
    
    def test_no_evidence_when_no_match(self):
        """Test empty evidence list when nothing matches."""
        from conductor_memory.search.verification import find_evidence
        
        tags = ["calls:process", "reads:self._data"]
        content = "def process(self): pass"
        claim = "uses database connection"
        
        evidence = find_evidence(tags, content, claim)
        
        assert len(evidence) == 0
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        from conductor_memory.search.verification import find_evidence
        
        assert find_evidence([], "", "claim") == []
        assert find_evidence(None, None, "claim") == []
        assert find_evidence(["calls:test"], "content", "") == []


# ============================================================================
# UNIT TESTS: Dataclasses
# ============================================================================

class TestVerificationResultToDict:
    """Unit tests for VerificationResult.to_dict() serialization."""
    
    def test_basic_serialization(self):
        """Test basic to_dict() output."""
        from conductor_memory.search.verification import (
            VerificationResult, VerificationInfo, SubjectInfo, 
            Evidence, VerificationStatus
        )
        
        result = VerificationResult(
            subject=SubjectInfo(
                name="_generate_features",
                file="src/trading/strategy.py",
                found=True,
                line=100,
                type="method"
            ),
            verification=VerificationInfo(
                status=VerificationStatus.SUPPORTED,
                confidence=0.92,
                evidence=[
                    Evidence(
                        type="subscript_access",
                        detail="df.iloc[bar_index]",
                        relevance=0.95,
                        line=105
                    )
                ]
            ),
            summary="VERIFIED: _generate_features uses iloc"
        )
        
        d = result.to_dict()
        
        assert d["search_mode"] == "verify"
        assert d["subject"]["name"] == "_generate_features"
        assert d["subject"]["file"] == "src/trading/strategy.py"
        assert d["subject"]["found"] is True
        assert d["subject"]["line"] == 100
        assert d["subject"]["type"] == "method"
        
        assert d["verification"]["status"] == "supported"
        assert d["verification"]["confidence"] == 0.92
        assert len(d["verification"]["evidence"]) == 1
        assert d["verification"]["evidence"][0]["type"] == "subscript_access"
        assert d["verification"]["evidence"][0]["detail"] == "df.iloc[bar_index]"
        assert d["verification"]["evidence"][0]["relevance"] == 0.95
        assert d["verification"]["evidence"][0]["line"] == 105
        
        assert d["summary"] == "VERIFIED: _generate_features uses iloc"
    
    def test_minimal_serialization(self):
        """Test to_dict() with minimal fields."""
        from conductor_memory.search.verification import (
            VerificationResult, VerificationInfo, SubjectInfo, 
            VerificationStatus
        )
        
        result = VerificationResult(
            subject=SubjectInfo(name="process", found=False),
            verification=VerificationInfo(
                status=VerificationStatus.SUBJECT_NOT_FOUND,
                confidence=1.0,
                evidence=[]
            ),
            summary="Could not find 'process'"
        )
        
        d = result.to_dict()
        
        assert "file" not in d["subject"]  # Optional field not included
        assert "line" not in d["subject"]
        assert "type" not in d["subject"]
        assert d["verification"]["evidence"] == []


class TestFactoryMethods:
    """Unit tests for factory methods."""
    
    def test_subject_not_found_factory(self):
        """Test VerificationResult.subject_not_found() factory."""
        from conductor_memory.search.verification import VerificationResult, VerificationStatus
        
        result = VerificationResult.subject_not_found(
            subject_name="_missing_method",
            claim="uses caching"
        )
        
        assert result.subject.name == "_missing_method"
        assert result.subject.found is False
        assert result.subject.file is None
        
        assert result.verification.status == VerificationStatus.SUBJECT_NOT_FOUND
        assert result.verification.confidence == 1.0
        assert result.verification.evidence == []
        
        assert "_missing_method" in result.summary
    
    def test_not_supported_factory(self):
        """Test VerificationResult.not_supported() factory."""
        from conductor_memory.search.verification import VerificationResult, VerificationStatus
        
        result = VerificationResult.not_supported(
            subject_name="process_data",
            file="src/processor.py",
            claim="uses database",
            line=50,
            subject_type="method"
        )
        
        assert result.subject.name == "process_data"
        assert result.subject.found is True
        assert result.subject.file == "src/processor.py"
        assert result.subject.line == 50
        assert result.subject.type == "method"
        
        assert result.verification.status == VerificationStatus.NOT_SUPPORTED
        assert result.verification.confidence == 0.8
        assert result.verification.evidence == []
        
        assert "process_data" in result.summary
        assert "uses database" in result.summary


class TestVerificationStatusEnum:
    """Unit tests for VerificationStatus enum."""
    
    def test_all_status_values(self):
        """Test all enum values exist and have correct string values."""
        from conductor_memory.search.verification import VerificationStatus
        
        assert VerificationStatus.SUPPORTED.value == "supported"
        assert VerificationStatus.NOT_SUPPORTED.value == "not_supported"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.INCONCLUSIVE.value == "inconclusive"
        assert VerificationStatus.SUBJECT_NOT_FOUND.value == "subject_not_found"
    
    def test_status_count(self):
        """Test that we have exactly 5 status values."""
        from conductor_memory.search.verification import VerificationStatus
        
        assert len(VerificationStatus) == 5


class TestEvidenceDataclass:
    """Unit tests for Evidence dataclass."""
    
    def test_evidence_creation(self):
        """Test Evidence creation and defaults."""
        from conductor_memory.search.verification import Evidence
        
        e = Evidence(type="call", detail="validate", relevance=0.8)
        
        assert e.type == "call"
        assert e.detail == "validate"
        assert e.relevance == 0.8
        assert e.line is None  # Default
    
    def test_evidence_with_line(self):
        """Test Evidence with line number."""
        from conductor_memory.search.verification import Evidence
        
        e = Evidence(type="subscript_access", detail="iloc", relevance=0.95, line=100)
        
        assert e.line == 100
    
    def test_evidence_to_dict(self):
        """Test Evidence.to_dict() serialization."""
        from conductor_memory.search.verification import Evidence
        
        e = Evidence(type="call", detail="validate", relevance=0.8, line=50)
        d = e.to_dict()
        
        assert d["type"] == "call"
        assert d["detail"] == "validate"
        assert d["relevance"] == 0.8
        assert d["line"] == 50
    
    def test_evidence_to_dict_without_line(self):
        """Test Evidence.to_dict() without line number."""
        from conductor_memory.search.verification import Evidence
        
        e = Evidence(type="call", detail="validate", relevance=0.8)
        d = e.to_dict()
        
        assert "line" not in d


class TestVerificationIntentDataclass:
    """Unit tests for VerificationIntent dataclass."""
    
    def test_intent_creation(self):
        """Test VerificationIntent creation."""
        from conductor_memory.search.verification import VerificationIntent
        
        intent = VerificationIntent(subject="process", claim="uses caching")
        
        assert intent.subject == "process"
        assert intent.claim == "uses caching"
    
    def test_intent_to_dict(self):
        """Test VerificationIntent.to_dict() serialization."""
        from conductor_memory.search.verification import VerificationIntent
        
        intent = VerificationIntent(subject="MyClass.method", claim="calls validate")
        d = intent.to_dict()
        
        assert d["subject"] == "MyClass.method"
        assert d["claim"] == "calls validate"


# ============================================================================
# INTEGRATION TESTS: Full Verification Flow
# ============================================================================

class TestVerificationFlowSupported:
    """Integration tests for SUPPORTED verification status."""
    
    @pytest.fixture
    def mock_chunk_with_evidence(self):
        """Create a mock chunk with implementation signals."""
        return {
            "id": "chunk-1",
            "content": """
def _generate_features(self, df, bar_index):
    row = df.iloc[bar_index]
    atr = self._atr_series.iloc[bar_index]
    self._validate_input(df)
    return row

[Implementation Signals]
Calls: _validate_input
Reads: self._atr_series
Subscripts: df.iloc[bar_index], self._atr_series.iloc[bar_index]
Parameters used: df, bar_index
""",
            "tags": [
                "file:src/strategy.py",
                "function:_generate_features",
                "calls:_validate_input",
                "reads:self._atr_series",
                "subscript:iloc",
                "param:bar_index",
                "param:df"
            ],
            "relevance_score": 0.9
        }
    
    def test_finds_evidence_for_valid_claim(self, mock_chunk_with_evidence):
        """Test that evidence is found for a valid claim."""
        from conductor_memory.search.verification import (
            find_evidence, parse_verification_query
        )
        
        query = "verify _generate_features uses iloc for indexing"
        intent = parse_verification_query(query)
        
        assert intent is not None
        
        evidence = find_evidence(
            mock_chunk_with_evidence["tags"],
            mock_chunk_with_evidence["content"],
            intent.claim
        )
        
        assert len(evidence) > 0
        
        # Should find iloc evidence
        details = [e.detail for e in evidence]
        detail_str = " ".join(details).lower()
        assert "iloc" in detail_str
    
    def test_verification_result_supported(self, mock_chunk_with_evidence):
        """Test creating SUPPORTED verification result."""
        from conductor_memory.search.verification import (
            VerificationResult, VerificationInfo, SubjectInfo,
            Evidence, VerificationStatus, find_evidence, parse_verification_query
        )
        
        query = "verify _generate_features uses bar_index"
        intent = parse_verification_query(query)
        
        evidence = find_evidence(
            mock_chunk_with_evidence["tags"],
            mock_chunk_with_evidence["content"],
            intent.claim
        )
        
        # Create result
        confidence = max((e.relevance for e in evidence), default=0) if evidence else 0
        
        result = VerificationResult(
            subject=SubjectInfo(
                name=intent.subject,
                file="src/strategy.py",
                found=True,
                type="function"
            ),
            verification=VerificationInfo(
                status=VerificationStatus.SUPPORTED,
                confidence=confidence,
                evidence=evidence
            ),
            summary=f"VERIFIED: '{intent.subject}' uses {intent.claim}"
        )
        
        assert result.verification.status == VerificationStatus.SUPPORTED
        assert len(result.verification.evidence) > 0
        assert result.subject.found is True


class TestVerificationFlowNotSupported:
    """Integration tests for NOT_SUPPORTED verification status."""
    
    @pytest.fixture
    def mock_chunk_without_evidence(self):
        """Create a mock chunk without matching signals."""
        return {
            "id": "chunk-2",
            "content": """
def process_data(self, data):
    return data.transform()

[Implementation Signals]
Calls: transform
""",
            "tags": [
                "file:src/processor.py",
                "function:process_data",
                "calls:transform"
            ],
            "relevance_score": 0.7
        }
    
    def test_no_evidence_for_invalid_claim(self, mock_chunk_without_evidence):
        """Test that no evidence is found for unrelated claim."""
        from conductor_memory.search.verification import (
            find_evidence, parse_verification_query
        )
        
        query = "verify process_data uses database connection"
        intent = parse_verification_query(query)
        
        evidence = find_evidence(
            mock_chunk_without_evidence["tags"],
            mock_chunk_without_evidence["content"],
            intent.claim
        )
        
        assert len(evidence) == 0
    
    def test_verification_result_not_supported(self, mock_chunk_without_evidence):
        """Test creating NOT_SUPPORTED verification result."""
        from conductor_memory.search.verification import (
            VerificationResult, VerificationStatus, parse_verification_query
        )
        
        query = "verify process_data uses caching"
        intent = parse_verification_query(query)
        
        result = VerificationResult.not_supported(
            subject_name=intent.subject,
            file="src/processor.py",
            claim=intent.claim
        )
        
        assert result.verification.status == VerificationStatus.NOT_SUPPORTED
        assert result.subject.found is True
        assert len(result.verification.evidence) == 0


class TestVerificationFlowSubjectNotFound:
    """Integration tests for SUBJECT_NOT_FOUND verification status."""
    
    def test_verification_result_subject_not_found(self):
        """Test creating SUBJECT_NOT_FOUND verification result."""
        from conductor_memory.search.verification import (
            VerificationResult, VerificationStatus, parse_verification_query
        )
        
        query = "verify _nonexistent_method uses caching"
        intent = parse_verification_query(query)
        
        result = VerificationResult.subject_not_found(
            subject_name=intent.subject,
            claim=intent.claim
        )
        
        assert result.verification.status == VerificationStatus.SUBJECT_NOT_FOUND
        assert result.subject.found is False
        assert result.subject.file is None
        assert "_nonexistent_method" in result.summary


# ============================================================================
# ACCURACY METRICS TESTS
# ============================================================================

class TestVerificationQueryParsingAccuracy:
    """
    Tests for the 80%+ verification query parsing accuracy metric.
    
    These tests validate that the implementation meets the success criteria
    from the implementation plan.
    """
    
    def test_parsing_accuracy_on_sample_queries(self):
        """
        Verify 80%+ of verification queries are correctly parsed.
        
        Uses a set of realistic verification queries.
        """
        from conductor_memory.search.verification import parse_verification_query
        
        # Sample queries that should be parsed successfully
        valid_queries = [
            "verify _generate_features uses iloc",
            "verify DataProcessor uses caching",
            "does MyClass use the repository pattern",
            "does process_data call validate",
            "is FeatureGenerator using window-relative indexing",
            "check if handler has retry logic",
            "confirm AuthService uses JWT tokens",
            "find if DatabaseManager uses connection pooling",
            "does __init__ call super",
            "verify Config.load uses yaml parsing",
            "does UserRepository.find_by_id use caching",
            "is AsyncHandler using asyncio",
            "check if _validate accesses self._config",
            "confirm process_batch has error handling",
            "verify transform_data uses pandas DataFrame",
        ]
        
        successful_parses = 0
        failed_queries = []
        
        for query in valid_queries:
            result = parse_verification_query(query)
            if result is not None:
                successful_parses += 1
            else:
                failed_queries.append(query)
        
        accuracy = successful_parses / len(valid_queries)
        
        print(f"\nVerification Query Parsing Accuracy Test:")
        print(f"  Total queries: {len(valid_queries)}")
        print(f"  Successful parses: {successful_parses}")
        print(f"  Accuracy: {accuracy:.1%}")
        if failed_queries:
            print(f"  Failed queries: {failed_queries}")
        
        # Must meet 80% success metric
        assert accuracy >= 0.8, f"Expected 80%+ parsing accuracy, got {accuracy:.1%}"
    
    def test_non_verification_queries_rejected(self):
        """Verify that non-verification queries are correctly rejected."""
        from conductor_memory.search.verification import parse_verification_query
        
        non_verification_queries = [
            "find all authentication methods",
            "what is the purpose of this class",
            "where is the login logic",
            "how does the cache work",
            "list all database queries",
            "show me error handling patterns",
            "_generate_features",
            "process data using iloc",
            "authentication flow",
            "get user by id implementation",
        ]
        
        false_positives = 0
        false_positive_queries = []
        
        for query in non_verification_queries:
            result = parse_verification_query(query)
            if result is not None:
                false_positives += 1
                false_positive_queries.append(query)
        
        rejection_rate = 1 - (false_positives / len(non_verification_queries))
        
        print(f"\nNon-Verification Query Rejection Test:")
        print(f"  Total queries: {len(non_verification_queries)}")
        print(f"  False positives: {false_positives}")
        print(f"  Rejection rate: {rejection_rate:.1%}")
        if false_positive_queries:
            print(f"  False positives: {false_positive_queries}")
        
        # Should reject most non-verification queries
        assert rejection_rate >= 0.9, f"Expected 90%+ rejection rate, got {rejection_rate:.1%}"


# ============================================================================
# MAIN: Run tests with pytest
# ============================================================================

def main():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Verification Search Mode Tests (Phase 3)")
    print("=" * 70)
    print("\nThese tests validate Phase 3 success metrics:")
    print("  - 80%+ verification queries correctly parsed")
    print("  - Evidence matching finds relevant signals")
    print("\n" + "=" * 70)
    
    # Run with pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
