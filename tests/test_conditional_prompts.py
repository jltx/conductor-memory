"""
Tests for Task C2: Conditional Prompt Building.

Verifies that prompts are correctly built based on config options.
"""

import pytest
from conductor_memory.config.summarization import SummarizationConfig
from conductor_memory.llm.summarizer import FileSummarizer
from conductor_memory.llm.base import LLMClient, LLMResponse
from conductor_memory.search.heuristics import HeuristicMetadata, MethodImplementationDetail


class MockLLMClient(LLMClient):
    """Mock LLM client that captures prompts for testing."""
    
    def __init__(self):
        # Initialize base class with minimal config
        super().__init__(base_url="http://localhost", model="mock", timeout=10.0)
        self.last_prompt = None
        self.last_system_prompt = None
        
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return LLMResponse(
            content='{"purpose": "test", "pattern": "Test", "key_exports": [], "dependencies": [], "domain": "test"}',
            model="mock",
            tokens_used=100,
            response_time_ms=50.0
        )
    
    async def health_check(self) -> bool:
        return True
    
    async def list_models(self) -> list[str]:
        return ["mock"]


class TestConditionalSystemPrompt:
    """Tests for conditional system prompt building."""
    
    def test_all_features_enabled(self):
        """System prompt includes all features when enabled."""
        config = SummarizationConfig(
            include_how_it_works=True,
            include_implementation_signals=True,
            include_method_summaries=True
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_system_prompt()
        
        # Should include how_it_works instructions
        assert "IMPORTANT: Focus on HOW the code works" in prompt
        assert "What data structures/indices it uses" in prompt
        
        # Should include implementation signals instructions
        assert "Use the implementation signals provided" in prompt
    
    def test_how_it_works_disabled(self):
        """System prompt excludes how_it_works when disabled."""
        config = SummarizationConfig(
            include_how_it_works=False,
            include_implementation_signals=True
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_system_prompt()
        
        # Should NOT include how_it_works instructions
        assert "IMPORTANT: Focus on HOW the code works" not in prompt
        assert "What data structures/indices it uses" not in prompt
        
        # Should still include implementation signals
        assert "Use the implementation signals provided" in prompt
    
    def test_implementation_signals_disabled(self):
        """System prompt excludes signals instructions when disabled."""
        config = SummarizationConfig(
            include_how_it_works=True,
            include_implementation_signals=False
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_system_prompt()
        
        # Should include how_it_works
        assert "IMPORTANT: Focus on HOW the code works" in prompt
        
        # Should NOT include signals instructions
        assert "Use the implementation signals provided" not in prompt
    
    def test_all_features_disabled(self):
        """System prompt is minimal when all Phase 2 features disabled."""
        config = SummarizationConfig(
            include_how_it_works=False,
            include_implementation_signals=False
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_system_prompt()
        
        # Should only have base prompt
        assert "code analysis expert" in prompt
        assert "IMPORTANT: Focus on HOW" not in prompt
        assert "Use the implementation signals" not in prompt


class TestConditionalUserPrompt:
    """Tests for conditional user prompt building."""
    
    def test_all_features_enabled_json_schema(self):
        """User prompt includes all JSON fields when features enabled."""
        config = SummarizationConfig(
            include_how_it_works=True,
            include_method_summaries=True,
            include_implementation_signals=True
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_user_prompt(
            "test.py", "python", "def foo(): pass", None
        )
        
        # Should include all JSON fields
        assert '"how_it_works"' in prompt
        assert '"key_mechanisms"' in prompt
        assert '"method_summaries"' in prompt
    
    def test_how_it_works_disabled_json_schema(self):
        """User prompt excludes how_it_works JSON fields when disabled."""
        config = SummarizationConfig(
            include_how_it_works=False,
            include_method_summaries=True
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_user_prompt(
            "test.py", "python", "def foo(): pass", None
        )
        
        # Should NOT include how_it_works fields
        assert '"how_it_works"' not in prompt
        assert '"key_mechanisms"' not in prompt
        
        # Should still include method_summaries
        assert '"method_summaries"' in prompt
    
    def test_method_summaries_disabled_json_schema(self):
        """User prompt excludes method_summaries JSON field when disabled."""
        config = SummarizationConfig(
            include_how_it_works=True,
            include_method_summaries=False
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_user_prompt(
            "test.py", "python", "def foo(): pass", None
        )
        
        # Should include how_it_works
        assert '"how_it_works"' in prompt
        assert '"key_mechanisms"' in prompt
        
        # Should NOT include method_summaries
        assert '"method_summaries"' not in prompt
    
    def test_all_phase2_disabled_json_schema(self):
        """User prompt has minimal JSON schema when all Phase 2 features disabled."""
        config = SummarizationConfig(
            include_how_it_works=False,
            include_method_summaries=False,
            include_implementation_signals=False
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        prompt = summarizer._build_user_prompt(
            "test.py", "python", "def foo(): pass", None
        )
        
        # Should have base fields only
        assert '"purpose"' in prompt
        assert '"pattern"' in prompt
        assert '"domain"' in prompt
        
        # Should NOT have Phase 2 fields
        assert '"how_it_works"' not in prompt
        assert '"key_mechanisms"' not in prompt
        assert '"method_summaries"' not in prompt
    
    def test_implementation_signals_included_when_enabled(self):
        """User prompt includes implementation signals section when enabled."""
        config = SummarizationConfig(
            include_implementation_signals=True
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        # Create heuristic metadata with method details
        method_detail = MethodImplementationDetail(
            name="test_method",
            signature="def test_method(data):",
            internal_calls=["helper_func"],
            external_calls=["external_api"],
            subscript_access=["df.iloc"],
            attribute_reads=["self._cache"],
            attribute_writes=["self._result"],
            parameters_used=["data"],
            has_loop=True,
            has_conditional=True,
            has_try_except=False,
            is_async=False
        )
        metadata = HeuristicMetadata(
            file_path="test.py",
            language="python",
            method_details=[method_detail]
        )
        
        prompt = summarizer._build_user_prompt(
            "test.py", "python", "def test_method(data): pass", metadata
        )
        
        # Should include implementation signals
        assert "Implementation Signals" in prompt
        assert "test_method()" in prompt
        assert "helper_func" in prompt
    
    def test_implementation_signals_excluded_when_disabled(self):
        """User prompt excludes implementation signals when disabled."""
        config = SummarizationConfig(
            include_implementation_signals=False
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        # Create heuristic metadata with method details
        method_detail = MethodImplementationDetail(
            name="test_method",
            signature="def test_method(data):",
            internal_calls=["helper_func"],
            external_calls=["external_api"]
        )
        metadata = HeuristicMetadata(
            file_path="test.py",
            language="python",
            method_details=[method_detail]
        )
        
        prompt = summarizer._build_user_prompt(
            "test.py", "python", "def test_method(data): pass", metadata
        )
        
        # Should NOT include implementation signals
        assert "Implementation Signals" not in prompt


class TestSimpleFileDetectionConfig:
    """Tests for enable_simple_file_detection config."""
    
    @pytest.mark.asyncio
    async def test_simple_file_skips_llm_when_enabled(self):
        """Simple files skip LLM when detection is enabled."""
        config = SummarizationConfig(
            enable_simple_file_detection=True
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        # Create metadata marking file as simple
        metadata = HeuristicMetadata(
            file_path="__init__.py",
            language="python",
            is_simple_file=True,
            simple_file_reason="barrel_reexport"
        )
        
        result = await summarizer.summarize_file(
            "__init__.py", 
            "from .module import func",
            metadata
        )
        
        # Should use template, not LLM
        assert result.simple_file is True
        assert result.model_used == "template"
        assert client.last_prompt is None  # LLM was not called
    
    @pytest.mark.asyncio
    async def test_simple_file_uses_llm_when_disabled(self):
        """Simple files use LLM when detection is disabled."""
        config = SummarizationConfig(
            enable_simple_file_detection=False
        )
        client = MockLLMClient()
        summarizer = FileSummarizer(client, config)
        
        # Create metadata marking file as simple
        metadata = HeuristicMetadata(
            file_path="__init__.py",
            language="python",
            is_simple_file=True,
            simple_file_reason="barrel_reexport"
        )
        
        result = await summarizer.summarize_file(
            "__init__.py", 
            "from .module import func",
            metadata
        )
        
        # Should use LLM despite being marked simple
        assert result.simple_file is False
        assert result.model_used == "mock"
        assert client.last_prompt is not None  # LLM was called


class TestPromptSizeReduction:
    """Tests verifying prompt size reduction when features disabled."""
    
    def test_prompt_smaller_without_implementation_signals(self):
        """Disabling implementation signals reduces prompt size."""
        # With signals
        config_with = SummarizationConfig(include_implementation_signals=True)
        summarizer_with = FileSummarizer(MockLLMClient(), config_with)
        
        # Without signals
        config_without = SummarizationConfig(include_implementation_signals=False)
        summarizer_without = FileSummarizer(MockLLMClient(), config_without)
        
        method_detail = MethodImplementationDetail(
            name="process",
            signature="def process():",
            internal_calls=["parse", "validate", "transform"],
            external_calls=["api.call", "db.query"],
            subscript_access=["df.iloc"],
            attribute_reads=["self._data", "self._config"],
            has_loop=True,
            has_conditional=True
        )
        metadata = HeuristicMetadata(
            file_path="test.py",
            language="python",
            method_details=[method_detail]
        )
        
        prompt_with = summarizer_with._build_user_prompt(
            "test.py", "python", "def process(): pass", metadata
        )
        prompt_without = summarizer_without._build_user_prompt(
            "test.py", "python", "def process(): pass", metadata
        )
        
        # Prompt without signals should be smaller
        assert len(prompt_without) < len(prompt_with)
    
    def test_prompt_smaller_without_phase2_fields(self):
        """Disabling Phase 2 fields reduces prompt size."""
        # With Phase 2
        config_with = SummarizationConfig(
            include_how_it_works=True,
            include_method_summaries=True
        )
        summarizer_with = FileSummarizer(MockLLMClient(), config_with)
        
        # Without Phase 2
        config_without = SummarizationConfig(
            include_how_it_works=False,
            include_method_summaries=False
        )
        summarizer_without = FileSummarizer(MockLLMClient(), config_without)
        
        prompt_with = summarizer_with._build_user_prompt(
            "test.py", "python", "def foo(): pass", None
        )
        prompt_without = summarizer_without._build_user_prompt(
            "test.py", "python", "def foo(): pass", None
        )
        
        # Prompt without Phase 2 fields should be smaller
        assert len(prompt_without) < len(prompt_with)
    
    def test_system_prompt_smaller_without_phase2(self):
        """System prompt is smaller when Phase 2 features disabled."""
        # With Phase 2
        config_with = SummarizationConfig(
            include_how_it_works=True,
            include_implementation_signals=True
        )
        summarizer_with = FileSummarizer(MockLLMClient(), config_with)
        
        # Without Phase 2
        config_without = SummarizationConfig(
            include_how_it_works=False,
            include_implementation_signals=False
        )
        summarizer_without = FileSummarizer(MockLLMClient(), config_without)
        
        sys_with = summarizer_with._build_system_prompt()
        sys_without = summarizer_without._build_system_prompt()
        
        # System prompt without Phase 2 should be smaller
        assert len(sys_without) < len(sys_with)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
