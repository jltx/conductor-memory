"""
Test suite for memory type enhancements:
- New memory types: WARNING, CONVENTION, EXPERIMENT
- New store tools: memory_store_warning, memory_store_convention, memory_store_experiment
- New query tools: memory_get_warnings, memory_get_conventions, memory_get_experiments
- Lifecycle tools: memory_deprecate_decision, memory_update_experiment, memory_get_active_decisions
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from conductor_memory.core.models import MemoryType, MemoryChunk, RoleEnum
from conductor_memory.config.server import ServerConfig


class TestMemoryTypeEnum:
    """Test the new memory type enum values"""

    def test_warning_type_exists(self):
        """Test WARNING memory type is defined"""
        assert hasattr(MemoryType, 'WARNING')
        assert MemoryType.WARNING.value == "warning"

    def test_convention_type_exists(self):
        """Test CONVENTION memory type is defined"""
        assert hasattr(MemoryType, 'CONVENTION')
        assert MemoryType.CONVENTION.value == "convention"

    def test_experiment_type_exists(self):
        """Test EXPERIMENT memory type is defined"""
        assert hasattr(MemoryType, 'EXPERIMENT')
        assert MemoryType.EXPERIMENT.value == "experiment"

    def test_all_memory_types(self):
        """Test all expected memory types are present"""
        expected_types = ['code', 'conversation', 'decision', 'lesson', 'warning', 'convention', 'experiment']
        actual_types = [mt.value for mt in MemoryType]
        for expected in expected_types:
            assert expected in actual_types, f"Missing memory type: {expected}"

    def test_memory_type_string_enum(self):
        """Test MemoryType is a string enum for JSON serialization"""
        # MemoryType inherits from str, so it compares equal to string values
        assert isinstance(MemoryType.WARNING, str)
        assert MemoryType.WARNING == "warning"
        # .value gives the string value
        assert MemoryType.CONVENTION.value == "convention"


class TestMemoryStoreWarning:
    """Test memory_store_warning tool functionality"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                # Mock store_async to return success
                async def mock_store(**kwargs):
                    return {
                        "success": True,
                        "id": "test-warning-id",
                        "memory_type": kwargs.get("memory_type"),
                        "tags": kwargs.get("tags", [])
                    }
                service.store_async = mock_store
                yield service

    @pytest.mark.asyncio
    async def test_store_warning_basic(self, mock_memory_service):
        """Test basic warning storage"""
        from conductor_memory.server.sse import memory_store_warning
        
        # Patch the global memory_service
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_store_warning(
                content="API rate limit reached, implement backoff",
                severity="high",
                tags=["api", "rate-limiting"]
            )
            
            assert result["success"] is True
            assert result["memory_type"] == "warning"
            assert "warning" in result["tags"]
            assert "severity:high" in result["tags"]
            assert "api" in result["tags"]
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_store_warning_severity_levels(self, mock_memory_service):
        """Test all severity levels are valid"""
        from conductor_memory.server.sse import memory_store_warning
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            for severity in ["low", "medium", "high", "critical"]:
                result = await memory_store_warning(
                    content=f"Test warning with {severity} severity",
                    severity=severity
                )
                assert result["success"] is True
                assert f"severity:{severity}" in result["tags"]
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_store_warning_default_severity(self, mock_memory_service):
        """Test default severity is medium"""
        from conductor_memory.server.sse import memory_store_warning
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_store_warning(content="Test warning")
            assert "severity:medium" in result["tags"]
        finally:
            sse_module.memory_service = original_service


class TestMemoryStoreConvention:
    """Test memory_store_convention tool functionality"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                async def mock_store(**kwargs):
                    return {
                        "success": True,
                        "id": "test-convention-id",
                        "memory_type": kwargs.get("memory_type"),
                        "tags": kwargs.get("tags", [])
                    }
                service.store_async = mock_store
                yield service

    @pytest.mark.asyncio
    async def test_store_convention_basic(self, mock_memory_service):
        """Test basic convention storage"""
        from conductor_memory.server.sse import memory_store_convention
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_store_convention(
                content="All API responses must use UnifiedResponse format",
                pattern_name="api-response",
                applies_to=["src/api/*.py", "src/routes/*.py"],
                tags=["api", "formatting"]
            )
            
            assert result["success"] is True
            assert result["memory_type"] == "convention"
            assert "convention" in result["tags"]
            assert "pattern:api-response" in result["tags"]
            assert "applies:src/api/*.py" in result["tags"]
            assert "applies:src/routes/*.py" in result["tags"]
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_store_convention_without_applies_to(self, mock_memory_service):
        """Test convention without applies_to (applies to all)"""
        from conductor_memory.server.sse import memory_store_convention
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_store_convention(
                content="Use snake_case for function names",
                pattern_name="naming-convention"
            )
            
            assert result["success"] is True
            assert "pattern:naming-convention" in result["tags"]
            # No applies: tags when applies_to is None
            applies_tags = [t for t in result["tags"] if t.startswith("applies:")]
            assert len(applies_tags) == 0
        finally:
            sse_module.memory_service = original_service


class TestMemoryStoreExperiment:
    """Test memory_store_experiment tool functionality"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                async def mock_store(**kwargs):
                    return {
                        "success": True,
                        "id": "test-experiment-id",
                        "memory_type": kwargs.get("memory_type"),
                        "tags": kwargs.get("tags", []),
                        "content": kwargs.get("content", "")
                    }
                service.store_async = mock_store
                yield service

    @pytest.mark.asyncio
    async def test_store_experiment_basic(self, mock_memory_service):
        """Test basic experiment storage"""
        from conductor_memory.server.sse import memory_store_experiment
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_store_experiment(
                hypothesis="MACD crossover predicts reversals",
                methodology="Backtest on 2 years of data",
                outcome="failure",
                result="Only 48% accuracy",
                metrics={"accuracy": 0.48, "sample_size": 1000},
                tags=["trading", "technical-analysis"]
            )
            
            assert result["success"] is True
            assert result["memory_type"] == "experiment"
            assert "experiment" in result["tags"]
            assert "outcome:failure" in result["tags"]
            assert "HYPOTHESIS:" in result["content"]
            assert "METHODOLOGY:" in result["content"]
            assert "RESULT:" in result["content"]
            assert "METRICS:" in result["content"]
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_store_experiment_pending(self, mock_memory_service):
        """Test storing experiment with pending outcome"""
        from conductor_memory.server.sse import memory_store_experiment
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_store_experiment(
                hypothesis="New caching strategy improves latency",
                methodology="A/B test with 1000 users"
            )
            
            assert result["success"] is True
            assert "outcome:pending" in result["tags"]
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_store_experiment_all_outcomes(self, mock_memory_service):
        """Test all outcome values"""
        from conductor_memory.server.sse import memory_store_experiment
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            for outcome in ["pending", "success", "failure", "inconclusive"]:
                result = await memory_store_experiment(
                    hypothesis=f"Test hypothesis for {outcome}",
                    outcome=outcome
                )
                assert result["success"] is True
                assert f"outcome:{outcome}" in result["tags"]
        finally:
            sse_module.memory_service = original_service


class TestMemoryGetWarnings:
    """Test memory_get_warnings query tool"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                async def mock_search(**kwargs):
                    # Return mock warnings based on include_tags
                    include_tags = kwargs.get("include_tags", [])
                    results = [
                        {"id": "warn-1", "tags": ["warning", "severity:high"], "content": "High severity warning"},
                        {"id": "warn-2", "tags": ["warning", "severity:medium"], "content": "Medium severity warning"},
                        {"id": "warn-3", "tags": ["warning", "severity:low"], "content": "Low severity warning"},
                    ]
                    
                    # Filter by severity if specified
                    if any("severity:" in t for t in include_tags):
                        severity_tag = next(t for t in include_tags if "severity:" in t)
                        results = [r for r in results if severity_tag in r["tags"]]
                    
                    return {"results": results}
                
                service.search_async = mock_search
                yield service

    @pytest.mark.asyncio
    async def test_get_warnings_all(self, mock_memory_service):
        """Test getting all warnings"""
        from conductor_memory.server.sse import memory_get_warnings
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_warnings()
            
            assert "warnings" in result
            assert "count" in result
            assert result["count"] == 3
            assert result["severity_filter"] is None
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_get_warnings_by_severity(self, mock_memory_service):
        """Test filtering warnings by severity"""
        from conductor_memory.server.sse import memory_get_warnings
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_warnings(severity="high")
            
            assert result["count"] == 1
            assert result["severity_filter"] == "high"
            assert all("severity:high" in w["tags"] for w in result["warnings"])
        finally:
            sse_module.memory_service = original_service


class TestMemoryGetConventions:
    """Test memory_get_conventions query tool"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                async def mock_search(**kwargs):
                    results = [
                        {"id": "conv-1", "tags": ["convention", "pattern:api-response", "applies:src/api/*.py"]},
                        {"id": "conv-2", "tags": ["convention", "pattern:error-handling", "applies:*.py"]},
                        {"id": "conv-3", "tags": ["convention", "pattern:naming"]},  # No applies, applies to all
                    ]
                    
                    include_tags = kwargs.get("include_tags", [])
                    if any("pattern:" in t for t in include_tags):
                        pattern_tag = next(t for t in include_tags if "pattern:" in t)
                        results = [r for r in results if pattern_tag in r["tags"]]
                    
                    return {"results": results}
                
                service.search_async = mock_search
                yield service

    @pytest.mark.asyncio
    async def test_get_conventions_all(self, mock_memory_service):
        """Test getting all conventions"""
        from conductor_memory.server.sse import memory_get_conventions
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_conventions()
            
            assert "conventions" in result
            assert "count" in result
            assert result["count"] == 3
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_get_conventions_by_pattern(self, mock_memory_service):
        """Test filtering conventions by pattern name"""
        from conductor_memory.server.sse import memory_get_conventions
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_conventions(pattern_name="api-response")
            
            assert result["pattern_filter"] == "api-response"
            assert result["count"] == 1
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_get_conventions_by_file(self, mock_memory_service):
        """Test filtering conventions by applicable file"""
        from conductor_memory.server.sse import memory_get_conventions
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_conventions(applies_to_file="src/api/users.py")
            
            assert result["file_filter"] == "src/api/users.py"
            # Should match api-response (src/api/*.py), error-handling (*.py), and naming (no applies = all)
            assert result["count"] >= 1
        finally:
            sse_module.memory_service = original_service


class TestMemoryGetExperiments:
    """Test memory_get_experiments query tool"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                async def mock_search(**kwargs):
                    results = [
                        {"id": "exp-1", "tags": ["experiment", "outcome:success"]},
                        {"id": "exp-2", "tags": ["experiment", "outcome:failure"]},
                        {"id": "exp-3", "tags": ["experiment", "outcome:pending"]},
                        {"id": "exp-4", "tags": ["experiment", "outcome:failure"]},
                    ]
                    
                    include_tags = kwargs.get("include_tags", [])
                    if any("outcome:" in t for t in include_tags):
                        outcome_tag = next(t for t in include_tags if "outcome:" in t)
                        results = [r for r in results if outcome_tag in r["tags"]]
                    
                    return {"results": results}
                
                service.search_async = mock_search
                yield service

    @pytest.mark.asyncio
    async def test_get_experiments_all(self, mock_memory_service):
        """Test getting all experiments"""
        from conductor_memory.server.sse import memory_get_experiments
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_experiments()
            
            assert "experiments" in result
            assert "count" in result
            assert "by_outcome" in result
            assert result["count"] == 4
            assert result["by_outcome"]["success"] == 1
            assert result["by_outcome"]["failure"] == 2
            assert result["by_outcome"]["pending"] == 1
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_get_experiments_by_outcome(self, mock_memory_service):
        """Test filtering experiments by outcome"""
        from conductor_memory.server.sse import memory_get_experiments
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_experiments(outcome="failure")
            
            assert result["outcome_filter"] == "failure"
            assert result["count"] == 2
        finally:
            sse_module.memory_service = original_service


class TestMemoryDecisionLifecycle:
    """Test decision lifecycle tools"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                async def mock_store(**kwargs):
                    return {
                        "success": True,
                        "id": "deprecation-record-id",
                        "content": kwargs.get("content", ""),
                        "tags": kwargs.get("tags", [])
                    }
                
                async def mock_search(**kwargs):
                    include_tags = kwargs.get("include_tags", [])
                    exclude_tags = kwargs.get("exclude_tags", [])
                    
                    results = [
                        {"id": "dec-1", "tags": ["decision"]},
                        {"id": "dec-2", "tags": ["decision", "status:deprecated"]},
                        {"id": "dec-3", "tags": ["decision"]},
                    ]
                    
                    # Apply exclude filter
                    if exclude_tags:
                        results = [r for r in results if not any(t in r["tags"] for t in exclude_tags)]
                    
                    return {"results": results}
                
                service.store_async = mock_store
                service.search_async = mock_search
                yield service

    @pytest.mark.asyncio
    async def test_deprecate_decision(self, mock_memory_service):
        """Test deprecating a decision"""
        from conductor_memory.server.sse import memory_deprecate_decision
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_deprecate_decision(
                decision_id="dec-123",
                reason="Replaced with better approach",
                superseded_by="dec-456"
            )
            
            assert result["success"] is True
            assert result["deprecated_id"] == "dec-123"
            assert result["reason"] == "Replaced with better approach"
            assert result["superseded_by"] == "dec-456"
            assert "deprecation_record_id" in result
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_get_active_decisions(self, mock_memory_service):
        """Test getting active (non-deprecated) decisions"""
        from conductor_memory.server.sse import memory_get_active_decisions
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_get_active_decisions()
            
            assert "decisions" in result
            assert "count" in result
            assert result["include_deprecated"] is False
            # Should exclude deprecated decision
            assert result["count"] == 2
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_get_active_decisions_include_deprecated(self, mock_memory_service):
        """Test getting all decisions including deprecated"""
        from conductor_memory.server.sse import memory_get_active_decisions
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            # Need to modify mock to not filter when include_deprecated=True
            async def mock_search_all(**kwargs):
                return {"results": [
                    {"id": "dec-1", "tags": ["decision"]},
                    {"id": "dec-2", "tags": ["decision", "status:deprecated"]},
                    {"id": "dec-3", "tags": ["decision"]},
                ]}
            mock_memory_service.search_async = mock_search_all
            
            result = await memory_get_active_decisions(include_deprecated=True)
            
            assert result["include_deprecated"] is True
            assert result["count"] == 3
        finally:
            sse_module.memory_service = original_service


class TestMemoryUpdateExperiment:
    """Test experiment update tool"""

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service for testing"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.service.memory_service import MemoryService
                config = ServerConfig()
                service = MemoryService(config)
                
                async def mock_store(**kwargs):
                    return {
                        "success": True,
                        "id": "update-record-id",
                        "content": kwargs.get("content", ""),
                        "tags": kwargs.get("tags", [])
                    }
                
                service.store_async = mock_store
                yield service

    @pytest.mark.asyncio
    async def test_update_experiment(self, mock_memory_service):
        """Test updating an experiment with results"""
        from conductor_memory.server.sse import memory_update_experiment
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_update_experiment(
                experiment_id="exp-123",
                result="Strategy achieved 65% accuracy",
                outcome="success",
                metrics={"accuracy": 0.65, "sharpe": 1.2}
            )
            
            assert result["success"] is True
            assert result["experiment_id"] == "exp-123"
            assert result["outcome"] == "success"
            assert "update_record_id" in result
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_update_experiment_partial(self, mock_memory_service):
        """Test partial experiment update (just outcome)"""
        from conductor_memory.server.sse import memory_update_experiment
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = mock_memory_service
        
        try:
            result = await memory_update_experiment(
                experiment_id="exp-456",
                outcome="inconclusive"
            )
            
            assert result["success"] is True
            assert result["outcome"] == "inconclusive"
        finally:
            sse_module.memory_service = original_service


class TestServiceNotInitialized:
    """Test error handling when memory service is not initialized"""

    @pytest.mark.asyncio
    async def test_store_warning_no_service(self):
        """Test store_warning returns error when service not initialized"""
        from conductor_memory.server.sse import memory_store_warning
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = None
        
        try:
            result = await memory_store_warning(content="test")
            assert "error" in result
            assert result["success"] is False
        finally:
            sse_module.memory_service = original_service

    @pytest.mark.asyncio
    async def test_get_warnings_no_service(self):
        """Test get_warnings returns error when service not initialized"""
        from conductor_memory.server.sse import memory_get_warnings
        
        import conductor_memory.server.sse as sse_module
        original_service = sse_module.memory_service
        sse_module.memory_service = None
        
        try:
            result = await memory_get_warnings()
            assert "error" in result
        finally:
            sse_module.memory_service = original_service


# Integration tests that require a running server
class TestIntegration:
    """Integration tests that hit the actual running server"""

    @pytest.fixture
    def server_url(self):
        return "http://127.0.0.1:9820"

    @pytest.mark.skip(reason="Requires running server - run manually")
    def test_store_and_retrieve_warning(self, server_url):
        """End-to-end test: store warning and retrieve it"""
        import requests
        import uuid
        
        test_id = str(uuid.uuid4())[:8]
        
        # Store a warning via API
        # Note: This would need to be done via MCP protocol
        # For now, this is a placeholder for manual testing
        pass

    @pytest.mark.skip(reason="Requires running server - run manually")
    def test_store_and_retrieve_convention(self, server_url):
        """End-to-end test: store convention and retrieve it"""
        pass

    @pytest.mark.skip(reason="Requires running server - run manually")
    def test_experiment_lifecycle(self, server_url):
        """End-to-end test: create experiment, update with results"""
        pass


# Run with: python -m pytest tests/test_memory_type_enhancements.py -v
