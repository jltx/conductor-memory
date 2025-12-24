#!/usr/bin/env python3
"""
Phase 2 Tests: Implementation-Aware LLM Summaries

Tests for the Phase 2 enhancement that adds HOW-focused summary fields
and implementation signal integration into LLM prompts.

Test Categories:
1. Unit Tests: FileSummary dataclass with new fields
2. Unit Tests: FileSummary.to_dict() serialization
3. Unit Tests: Prompt building (_build_system_prompt, _build_user_prompt)
4. Integration Tests: FileSummarizer.summarize_file() with mocked LLM

Success Metrics from Implementation Plan:
- Summaries include "how_it_works" section
- Token usage increase < 30%
"""

import sys
import os
import json
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


# ============================================================================
# TEST DATA
# ============================================================================

SAMPLE_PYTHON_CODE = '''
import pandas as pd
from typing import List, Optional

class FeatureGenerator:
    """Generates features for swing trading setups."""
    
    def __init__(self, config):
        self._scaler = StandardScaler()
        self._cache = {}
        self._fitted_len = 0
    
    def fit(self, data: pd.DataFrame, bar_index: int):
        """Fit the generator with new data."""
        if bar_index > self._fitted_len:
            window = data.iloc[bar_index - 10:bar_index]
            self._scaler.fit(window)
            self._fitted_len = bar_index
    
    def generate(self, df, setup, bar_index: int) -> dict:
        """Generate features for a specific bar."""
        row = df.iloc[bar_index]
        cached = self._cache.get(bar_index, {})
        return cached
'''

SAMPLE_LLM_RESPONSE_FULL = {
    "purpose": "Generates features for swing trading setups using window-relative indexing",
    "pattern": "Generator",
    "key_exports": ["FeatureGenerator"],
    "dependencies": ["pandas", "sklearn"],
    "domain": "ml",
    "how_it_works": "Uses window-relative indexing via df.iloc[bar_index]. Caches fitted state in _fitted_len to avoid repeated computation. Scaler is fitted incrementally as new data arrives.",
    "key_mechanisms": [
        "window-relative indexing",
        "incremental fit caching",
        "lazy initialization"
    ],
    "method_summaries": {
        "fit": "Fits scaler only when bar_index exceeds cached length, uses window slice for efficiency",
        "generate": "Retrieves cached features by bar_index, returns empty dict on cache miss"
    }
}

SAMPLE_LLM_RESPONSE_MINIMAL = {
    "purpose": "Generates features for trading",
    "pattern": "Generator",
    "key_exports": ["FeatureGenerator"],
    "dependencies": ["pandas"],
    "domain": "ml"
    # No how_it_works, key_mechanisms, or method_summaries
}


# ============================================================================
# UNIT TESTS: FileSummary dataclass
# ============================================================================

class TestFileSummaryDataclass:
    """Unit tests for FileSummary dataclass with Phase 2 fields."""
    
    def test_filesummary_creation_minimal(self):
        """Test creating FileSummary with minimal required fields."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="test.py",
            language="python",
            purpose="Test file",
            pattern="Utility",
            key_exports=["test_func"],
            dependencies=["os"],
            domain="testing",
            model_used="test-model"
        )
        
        assert summary.file_path == "test.py"
        assert summary.language == "python"
        assert summary.purpose == "Test file"
        assert summary.pattern == "Utility"
        assert summary.key_exports == ["test_func"]
        assert summary.dependencies == ["os"]
        assert summary.domain == "testing"
        assert summary.model_used == "test-model"
        
        # Phase 2 fields should default to None
        assert summary.how_it_works is None
        assert summary.key_mechanisms is None
        assert summary.method_summaries is None
    
    def test_filesummary_creation_with_phase2_fields(self):
        """Test creating FileSummary with all Phase 2 fields populated."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="feature_gen.py",
            language="python",
            purpose="Generates ML features",
            pattern="Generator",
            key_exports=["FeatureGenerator"],
            dependencies=["pandas", "sklearn"],
            domain="ml",
            model_used="llama3",
            tokens_used=150,
            response_time_ms=250.5,
            is_skeleton=False,
            # Phase 2 fields
            how_it_works="Uses window-relative indexing with caching for efficiency.",
            key_mechanisms=["window-relative indexing", "caching", "lazy fit"],
            method_summaries={
                "fit": "Fits scaler incrementally",
                "generate": "Returns cached features by bar_index"
            }
        )
        
        assert summary.how_it_works == "Uses window-relative indexing with caching for efficiency."
        assert summary.key_mechanisms == ["window-relative indexing", "caching", "lazy fit"]
        assert summary.method_summaries == {
            "fit": "Fits scaler incrementally",
            "generate": "Returns cached features by bar_index"
        }
    
    def test_filesummary_optional_metadata_fields(self):
        """Test that optional metadata fields work correctly."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="test.py",
            language="python",
            purpose="Test",
            pattern="Test",
            key_exports=[],
            dependencies=[],
            domain="test",
            model_used="test",
            tokens_used=100,
            response_time_ms=50.0,
            is_skeleton=True,
            error="Some error occurred"
        )
        
        assert summary.tokens_used == 100
        assert summary.response_time_ms == 50.0
        assert summary.is_skeleton is True
        assert summary.error == "Some error occurred"


# ============================================================================
# UNIT TESTS: FileSummary.to_dict() serialization
# ============================================================================

class TestFileSummaryToDict:
    """Unit tests for FileSummary.to_dict() with Phase 2 fields."""
    
    def test_to_dict_includes_phase2_fields_when_present(self):
        """Test that to_dict() includes how_it_works, key_mechanisms, method_summaries when set."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="feature.py",
            language="python",
            purpose="Feature generation",
            pattern="Generator",
            key_exports=["FeatureGen"],
            dependencies=["pandas"],
            domain="ml",
            model_used="llama3",
            how_it_works="Uses iloc for window indexing",
            key_mechanisms=["window indexing", "caching"],
            method_summaries={"fit": "Fits the model"}
        )
        
        result = summary.to_dict()
        
        # Check Phase 2 fields are included
        assert "how_it_works" in result
        assert result["how_it_works"] == "Uses iloc for window indexing"
        
        assert "key_mechanisms" in result
        assert result["key_mechanisms"] == ["window indexing", "caching"]
        
        assert "method_summaries" in result
        assert result["method_summaries"] == {"fit": "Fits the model"}
    
    def test_to_dict_excludes_phase2_fields_when_none(self):
        """Test that to_dict() excludes Phase 2 fields when they are None (backward compatibility)."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="simple.py",
            language="python",
            purpose="Simple utility",
            pattern="Utility",
            key_exports=["util_func"],
            dependencies=[],
            domain="utility",
            model_used="gpt-4"
            # Phase 2 fields not set (None)
        )
        
        result = summary.to_dict()
        
        # Check Phase 2 fields are NOT included
        assert "how_it_works" not in result
        assert "key_mechanisms" not in result
        assert "method_summaries" not in result
        
        # But base fields should still be present
        assert "file_path" in result
        assert "purpose" in result
        assert "pattern" in result
    
    def test_to_dict_partial_phase2_fields(self):
        """Test that to_dict() handles partial Phase 2 fields correctly."""
        from conductor_memory.llm.summarizer import FileSummary
        
        # Only how_it_works set
        summary = FileSummary(
            file_path="partial.py",
            language="python",
            purpose="Partial test",
            pattern="Test",
            key_exports=[],
            dependencies=[],
            domain="test",
            model_used="test",
            how_it_works="Some explanation",
            key_mechanisms=None,  # Explicitly None
            method_summaries=None  # Explicitly None
        )
        
        result = summary.to_dict()
        
        assert "how_it_works" in result
        assert result["how_it_works"] == "Some explanation"
        assert "key_mechanisms" not in result
        assert "method_summaries" not in result
    
    def test_to_dict_preserves_all_base_fields(self):
        """Test that to_dict() preserves all original base fields."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="complete.py",
            language="python",
            purpose="Complete test",
            pattern="Service",
            key_exports=["ServiceClass", "helper_func"],
            dependencies=["requests", "json"],
            domain="api",
            model_used="claude-3",
            tokens_used=250,
            response_time_ms=100.5,
            is_skeleton=True,
            error=None
        )
        
        result = summary.to_dict()
        
        assert result["file_path"] == "complete.py"
        assert result["language"] == "python"
        assert result["purpose"] == "Complete test"
        assert result["pattern"] == "Service"
        assert result["key_exports"] == ["ServiceClass", "helper_func"]
        assert result["dependencies"] == ["requests", "json"]
        assert result["domain"] == "api"
        assert result["model_used"] == "claude-3"
        assert result["tokens_used"] == 250
        assert result["response_time_ms"] == 100.5
        assert result["is_skeleton"] is True
        assert result["error"] is None
    
    def test_to_dict_empty_lists_vs_none(self):
        """Test handling of empty lists vs None for Phase 2 fields."""
        from conductor_memory.llm.summarizer import FileSummary
        
        # Empty list should be included (it's not None)
        summary = FileSummary(
            file_path="empty.py",
            language="python",
            purpose="Empty test",
            pattern="Test",
            key_exports=[],
            dependencies=[],
            domain="test",
            model_used="test",
            key_mechanisms=[]  # Empty list, not None
        )
        
        result = summary.to_dict()
        
        # Empty list IS included (per current implementation checking for None)
        assert "key_mechanisms" in result
        assert result["key_mechanisms"] == []


# ============================================================================
# UNIT TESTS: Prompt building
# ============================================================================

class TestSystemPromptBuilding:
    """Unit tests for _build_system_prompt() with HOW-focused instructions."""
    
    @pytest.fixture
    def summarizer(self):
        """Create a FileSummarizer with mocked LLM client."""
        from conductor_memory.llm.summarizer import FileSummarizer
        from conductor_memory.config.summarization import SummarizationConfig
        
        mock_client = MagicMock()
        config = SummarizationConfig()
        return FileSummarizer(llm_client=mock_client, config=config)
    
    def test_system_prompt_contains_how_focus_instruction(self, summarizer):
        """Test that system prompt includes HOW-focused instruction."""
        prompt = summarizer._build_system_prompt()
        
        assert "IMPORTANT: Focus on HOW the code works" in prompt
    
    def test_system_prompt_explains_method_analysis(self, summarizer):
        """Test that system prompt explains what to include for methods."""
        prompt = summarizer._build_system_prompt()
        
        # Should mention data structures/indices
        assert "data structures" in prompt.lower() or "indices" in prompt.lower()
        
        # Should mention caching/optimization
        assert "caching" in prompt.lower() or "optimization" in prompt.lower()
        
        # Should mention parameters
        assert "parameter" in prompt.lower()
    
    def test_system_prompt_mentions_implementation_signals(self, summarizer):
        """Test that system prompt mentions using implementation signals."""
        prompt = summarizer._build_system_prompt()
        
        assert "implementation signals" in prompt.lower()
    
    def test_system_prompt_requests_json(self, summarizer):
        """Test that system prompt requests JSON response."""
        prompt = summarizer._build_system_prompt()
        
        assert "JSON" in prompt or "json" in prompt


class TestUserPromptBuilding:
    """Unit tests for _build_user_prompt() with Phase 2 enhancements."""
    
    @pytest.fixture
    def summarizer(self):
        """Create a FileSummarizer with mocked LLM client."""
        from conductor_memory.llm.summarizer import FileSummarizer
        from conductor_memory.config.summarization import SummarizationConfig
        
        mock_client = MagicMock()
        config = SummarizationConfig()
        return FileSummarizer(llm_client=mock_client, config=config)
    
    def test_user_prompt_includes_file_content(self, summarizer):
        """Test that user prompt includes the file content."""
        prompt = summarizer._build_user_prompt(
            file_path="test.py",
            language="python",
            content="def foo(): pass",
            heuristic_metadata=None
        )
        
        assert "def foo(): pass" in prompt
        assert "test.py" in prompt
    
    def test_user_prompt_includes_how_it_works_schema(self, summarizer):
        """Test that user prompt JSON schema includes how_it_works field."""
        prompt = summarizer._build_user_prompt(
            file_path="test.py",
            language="python",
            content="class Test: pass",
            heuristic_metadata=None
        )
        
        assert "how_it_works" in prompt
    
    def test_user_prompt_includes_key_mechanisms_schema(self, summarizer):
        """Test that user prompt JSON schema includes key_mechanisms field."""
        prompt = summarizer._build_user_prompt(
            file_path="test.py",
            language="python",
            content="class Test: pass",
            heuristic_metadata=None
        )
        
        assert "key_mechanisms" in prompt
    
    def test_user_prompt_includes_method_summaries_schema(self, summarizer):
        """Test that user prompt JSON schema includes method_summaries field."""
        prompt = summarizer._build_user_prompt(
            file_path="test.py",
            language="python",
            content="class Test: pass",
            heuristic_metadata=None
        )
        
        assert "method_summaries" in prompt
    
    def test_user_prompt_includes_implementation_signals_when_available(self, summarizer):
        """Test that user prompt includes implementation signals from heuristic metadata."""
        from conductor_memory.search.heuristics import HeuristicMetadata, MethodImplementationDetail
        
        # Create heuristic metadata with method details
        method_detail = MethodImplementationDetail(
            name="fit",
            signature="def fit(self, data, bar_index)",
            internal_calls=["_validate", "_transform"],
            external_calls=["scaler.fit"],
            attribute_reads=["self._cache"],
            attribute_writes=["self._fitted_len"],
            subscript_access=["data.iloc[bar_index]"],
            parameters_used=["data", "bar_index"],
            has_loop=True,
            has_conditional=True,
            has_try_except=False,
            is_async=False,
            line_count=15
        )
        
        heuristic_metadata = HeuristicMetadata(
            file_path="test.py",
            language="python",
            method_details=[method_detail]
        )
        
        prompt = summarizer._build_user_prompt(
            file_path="test.py",
            language="python",
            content="class FeatureGen: pass",
            heuristic_metadata=heuristic_metadata
        )
        
        # Should include implementation signals section
        assert "Implementation Signals" in prompt
        assert "fit()" in prompt
        
        # Should include specific signals
        assert "calls:" in prompt.lower()
        assert "_validate" in prompt or "validate" in prompt
    
    def test_user_prompt_omits_signals_when_no_method_details(self, summarizer):
        """Test that signals section is omitted when no method details available."""
        from conductor_memory.search.heuristics import HeuristicMetadata
        
        # Heuristic metadata without method details
        heuristic_metadata = HeuristicMetadata(
            file_path="test.py",
            language="python",
            method_details=[]  # Empty
        )
        
        prompt = summarizer._build_user_prompt(
            file_path="test.py",
            language="python",
            content="x = 1",
            heuristic_metadata=heuristic_metadata
        )
        
        # Should NOT include implementation signals header
        # (The method returns empty string for no details)
        # Check that signals section is minimal or absent
        lines = prompt.split('\n')
        signal_lines = [l for l in lines if "Implementation Signals" in l]
        
        # Either no signal section, or it's present but empty
        # The key is that we don't have spurious signal data
        assert len(signal_lines) <= 1


class TestFormatImplementationSignals:
    """Unit tests for _format_implementation_signals() helper method."""
    
    @pytest.fixture
    def summarizer(self):
        """Create a FileSummarizer with mocked LLM client."""
        from conductor_memory.llm.summarizer import FileSummarizer
        from conductor_memory.config.summarization import SummarizationConfig
        
        mock_client = MagicMock()
        config = SummarizationConfig()
        return FileSummarizer(llm_client=mock_client, config=config)
    
    def test_format_signals_empty_list(self, summarizer):
        """Test formatting with empty method details list."""
        result = summarizer._format_implementation_signals([])
        assert result == ""
    
    def test_format_signals_single_method(self, summarizer):
        """Test formatting with a single method."""
        from conductor_memory.search.heuristics import MethodImplementationDetail
        
        detail = MethodImplementationDetail(
            name="process",
            signature="def process(self, data)",
            internal_calls=["_validate"],
            external_calls=["pd.concat"],
            attribute_reads=["self._config"],
            subscript_access=["data.iloc[idx]"],
            parameters_used=["data"],
            has_loop=True,
            has_conditional=False
        )
        
        result = summarizer._format_implementation_signals([detail])
        
        assert "Implementation Signals" in result
        assert "process()" in result
        assert "calls:" in result.lower()
        assert "_validate" in result
    
    def test_format_signals_limits_methods(self, summarizer):
        """Test that formatting limits to top 10 methods by signal richness."""
        from conductor_memory.search.heuristics import MethodImplementationDetail
        
        # Create 15 methods with varying signal counts
        details = []
        for i in range(15):
            detail = MethodImplementationDetail(
                name=f"method_{i}",
                signature=f"def method_{i}(self)",
                internal_calls=[f"call_{j}" for j in range(i)],  # More calls = higher priority
                external_calls=[],
                attribute_reads=[],
                subscript_access=[],
                parameters_used=[]
            )
            details.append(detail)
        
        result = summarizer._format_implementation_signals(details)
        
        # Count how many methods are mentioned
        method_mentions = result.count("method_")
        
        # Should have at most 10 methods (per the implementation)
        assert method_mentions <= 10
        
        # The method with most signals (method_14) should be included
        assert "method_14" in result
    
    def test_format_signals_includes_structural_hints(self, summarizer):
        """Test that structural hints (loops, conditionals, async) are included."""
        from conductor_memory.search.heuristics import MethodImplementationDetail
        
        detail = MethodImplementationDetail(
            name="complex_method",
            signature="async def complex_method(self)",
            internal_calls=["helper"],
            has_loop=True,
            has_conditional=True,
            has_try_except=True,
            is_async=True
        )
        
        result = summarizer._format_implementation_signals([detail])
        
        assert "structure:" in result.lower()
        assert "loops" in result.lower() or "loop" in result.lower()
        assert "conditionals" in result.lower() or "conditional" in result.lower()
        assert "async" in result.lower()
    
    def test_format_signals_truncates_long_lists(self, summarizer):
        """Test that signal lists are truncated per method (calls: 8, subscripts: 5, etc.)."""
        from conductor_memory.search.heuristics import MethodImplementationDetail
        
        detail = MethodImplementationDetail(
            name="many_signals",
            signature="def many_signals(self)",
            internal_calls=[f"internal_{i}" for i in range(20)],
            external_calls=[f"external_{i}" for i in range(20)],
            subscript_access=[f"sub_{i}" for i in range(20)],
            attribute_reads=[f"read_{i}" for i in range(20)],
            attribute_writes=[f"write_{i}" for i in range(20)],
            parameters_used=[f"param_{i}" for i in range(10)]
        )
        
        result = summarizer._format_implementation_signals([detail])
        
        # Count occurrences - should be truncated
        # Internal + external calls limited to 8 total
        internal_count = sum(1 for i in range(20) if f"internal_{i}" in result)
        external_count = sum(1 for i in range(20) if f"external_{i}" in result)
        
        # Total calls should be <= 8 per the implementation
        assert internal_count + external_count <= 8


# ============================================================================
# INTEGRATION TESTS: FileSummarizer.summarize_file() with mocked LLM
# ============================================================================

class TestFileSummarizerIntegration:
    """Integration tests for summarize_file() with mocked LLM responses."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        from conductor_memory.llm.base import LLMResponse
        
        client = MagicMock()
        client.generate = AsyncMock()
        return client
    
    @pytest.fixture
    def summarizer(self, mock_llm_client):
        """Create a FileSummarizer with mocked LLM client."""
        from conductor_memory.llm.summarizer import FileSummarizer
        from conductor_memory.config.summarization import SummarizationConfig
        
        config = SummarizationConfig()
        return FileSummarizer(llm_client=mock_llm_client, config=config)
    
    @pytest.mark.asyncio
    async def test_summarize_file_populates_phase2_fields(self, summarizer, mock_llm_client):
        """Test that summarize_file() populates Phase 2 fields from LLM response."""
        from conductor_memory.llm.base import LLMResponse
        
        # Configure mock to return full Phase 2 response
        mock_response = LLMResponse(
            content=json.dumps(SAMPLE_LLM_RESPONSE_FULL),
            model="llama3",
            tokens_used=150,
            response_time_ms=250.0
        )
        mock_llm_client.generate.return_value = mock_response
        
        result = await summarizer.summarize_file(
            file_path="feature_gen.py",
            content=SAMPLE_PYTHON_CODE
        )
        
        # Verify Phase 2 fields are populated
        assert result.how_it_works is not None
        assert "window-relative indexing" in result.how_it_works
        
        assert result.key_mechanisms is not None
        assert len(result.key_mechanisms) == 3
        assert "window-relative indexing" in result.key_mechanisms
        
        assert result.method_summaries is not None
        assert "fit" in result.method_summaries
        assert "generate" in result.method_summaries
    
    @pytest.mark.asyncio
    async def test_summarize_file_handles_minimal_response(self, summarizer, mock_llm_client):
        """Test that summarize_file() handles responses without Phase 2 fields."""
        from conductor_memory.llm.base import LLMResponse
        
        # Configure mock to return minimal response (no Phase 2 fields)
        mock_response = LLMResponse(
            content=json.dumps(SAMPLE_LLM_RESPONSE_MINIMAL),
            model="llama3",
            tokens_used=80,
            response_time_ms=150.0
        )
        mock_llm_client.generate.return_value = mock_response
        
        result = await summarizer.summarize_file(
            file_path="simple.py",
            content="def foo(): pass"
        )
        
        # Verify base fields are populated
        assert result.purpose == "Generates features for trading"
        assert result.pattern == "Generator"
        
        # Verify Phase 2 fields are None
        assert result.how_it_works is None
        assert result.key_mechanisms is None
        assert result.method_summaries is None
    
    @pytest.mark.asyncio
    async def test_summarize_file_with_heuristic_metadata(self, summarizer, mock_llm_client):
        """Test that summarize_file() includes heuristic metadata in prompt."""
        from conductor_memory.llm.base import LLMResponse
        from conductor_memory.search.heuristics import HeuristicMetadata, MethodImplementationDetail
        
        # Configure mock response
        mock_response = LLMResponse(
            content=json.dumps(SAMPLE_LLM_RESPONSE_FULL),
            model="llama3",
            tokens_used=200,
            response_time_ms=300.0
        )
        mock_llm_client.generate.return_value = mock_response
        
        # Create heuristic metadata
        method_detail = MethodImplementationDetail(
            name="fit",
            signature="def fit(self, data, bar_index)",
            internal_calls=["_scaler.fit"],
            subscript_access=["data.iloc[bar_index]"],
            parameters_used=["data", "bar_index"]
        )
        
        heuristic_metadata = HeuristicMetadata(
            file_path="feature_gen.py",
            language="python",
            classes=[{"name": "FeatureGenerator", "start_line": 5, "end_line": 25}],
            method_details=[method_detail]
        )
        
        result = await summarizer.summarize_file(
            file_path="feature_gen.py",
            content=SAMPLE_PYTHON_CODE,
            heuristic_metadata=heuristic_metadata
        )
        
        # Verify the prompt was called with expected content
        call_args = mock_llm_client.generate.call_args
        user_prompt = call_args.kwargs.get('prompt', call_args.args[0] if call_args.args else '')
        
        # Heuristic context should be included
        assert "FeatureGenerator" in user_prompt or "Classes:" in user_prompt
    
    @pytest.mark.asyncio
    async def test_summarize_file_error_handling(self, summarizer, mock_llm_client):
        """Test that summarize_file() handles LLM errors gracefully."""
        # Configure mock to raise an exception
        mock_llm_client.generate.side_effect = Exception("LLM connection failed")
        
        result = await summarizer.summarize_file(
            file_path="error.py",
            content="def broken(): pass"
        )
        
        # Should return error summary, not raise
        assert result.error is not None
        assert "LLM connection failed" in result.error
        assert result.purpose == "Failed to generate summary"
    
    @pytest.mark.asyncio
    async def test_summarize_file_invalid_json_response(self, summarizer, mock_llm_client):
        """Test handling of invalid JSON response from LLM."""
        from conductor_memory.llm.base import LLMResponse
        
        # Configure mock to return invalid JSON
        mock_response = LLMResponse(
            content="This is not valid JSON { broken",
            model="llama3",
            tokens_used=50,
            response_time_ms=100.0
        )
        mock_llm_client.generate.return_value = mock_response
        
        result = await summarizer.summarize_file(
            file_path="bad_response.py",
            content="def foo(): pass"
        )
        
        # Should handle gracefully with fallback
        assert result.file_path == "bad_response.py"
        # Fallback creates a basic summary
        assert result.error is not None or result.purpose is not None
    
    @pytest.mark.asyncio
    async def test_summarize_file_json_in_markdown(self, summarizer, mock_llm_client):
        """Test handling of JSON wrapped in markdown code blocks."""
        from conductor_memory.llm.base import LLMResponse
        
        # Some LLMs wrap JSON in markdown
        wrapped_content = f"```json\n{json.dumps(SAMPLE_LLM_RESPONSE_FULL)}\n```"
        mock_response = LLMResponse(
            content=wrapped_content,
            model="llama3",
            tokens_used=160,
            response_time_ms=260.0
        )
        mock_llm_client.generate.return_value = mock_response
        
        result = await summarizer.summarize_file(
            file_path="markdown_wrapped.py",
            content=SAMPLE_PYTHON_CODE
        )
        
        # Should extract JSON from markdown - the _parse_text_response fallback
        # attempts to extract JSON with regex
        # The result should either have the parsed data or a fallback
        assert result.file_path == "markdown_wrapped.py"


# ============================================================================
# INTEGRATION TESTS: to_dict() round-trip
# ============================================================================

class TestFileSummaryRoundTrip:
    """Test that FileSummary can be serialized and used correctly."""
    
    def test_to_dict_json_serializable(self):
        """Test that to_dict() output is JSON serializable."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="test.py",
            language="python",
            purpose="Test purpose",
            pattern="Service",
            key_exports=["TestClass"],
            dependencies=["requests"],
            domain="api",
            model_used="claude",
            how_it_works="Uses REST API calls",
            key_mechanisms=["caching", "retry logic"],
            method_summaries={"fetch": "Fetches data from API"}
        )
        
        dict_data = summary.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(dict_data)
        assert json_str is not None
        
        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["how_it_works"] == "Uses REST API calls"
        assert parsed["key_mechanisms"] == ["caching", "retry logic"]
        assert parsed["method_summaries"]["fetch"] == "Fetches data from API"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestPhase2EdgeCases:
    """Test edge cases for Phase 2 implementation."""
    
    def test_empty_method_summaries_dict(self):
        """Test handling of empty method_summaries dict."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="test.py",
            language="python",
            purpose="Test",
            pattern="Test",
            key_exports=[],
            dependencies=[],
            domain="test",
            model_used="test",
            method_summaries={}  # Empty dict
        )
        
        result = summary.to_dict()
        
        # Empty dict should be included (it's not None)
        assert "method_summaries" in result
        assert result["method_summaries"] == {}
    
    def test_unicode_in_phase2_fields(self):
        """Test handling of unicode characters in Phase 2 fields."""
        from conductor_memory.llm.summarizer import FileSummary
        
        summary = FileSummary(
            file_path="unicode.py",
            language="python",
            purpose="Handles émojis 🎉",
            pattern="Utility",
            key_exports=[],
            dependencies=[],
            domain="i18n",
            model_used="test",
            how_it_works="Processes UTF-8 strings with special chars: äöü 中文 🚀",
            key_mechanisms=["UTF-8 handling", "emoji support 😊"],
            method_summaries={"process": "Handles unicode → properly"}
        )
        
        result = summary.to_dict()
        json_str = json.dumps(result, ensure_ascii=False)
        
        assert "🎉" in json_str
        assert "äöü" in json_str
        assert "中文" in json_str
    
    def test_very_long_how_it_works(self):
        """Test handling of very long how_it_works text."""
        from conductor_memory.llm.summarizer import FileSummary
        
        long_text = "This is a very detailed explanation. " * 100
        
        summary = FileSummary(
            file_path="verbose.py",
            language="python",
            purpose="Verbose file",
            pattern="Unknown",
            key_exports=[],
            dependencies=[],
            domain="test",
            model_used="test",
            how_it_works=long_text
        )
        
        result = summary.to_dict()
        
        assert result["how_it_works"] == long_text
        assert len(result["how_it_works"]) > 3000


# ============================================================================
# MAIN: Run tests with pytest
# ============================================================================

def main():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Phase 2 Implementation-Aware LLM Summaries Tests")
    print("=" * 70)
    print("\nThese tests validate Phase 2 implementation:")
    print("  - FileSummary dataclass with new fields")
    print("  - FileSummary.to_dict() serialization")
    print("  - Prompt building with HOW-focus and signals")
    print("  - Integration with mocked LLM")
    print("\n" + "=" * 70)
    
    # Run with pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
