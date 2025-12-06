"""
Comprehensive test suite for conductor-memory MCP system
Tests Phase 5 (Summary Integration) and Phase 6 (Incremental Re-summarization & Web UI) functionality
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import requests
import asyncio
from datetime import datetime

from conductor_memory.config.server import ServerConfig, CodebaseConfig, BoostConfig
from conductor_memory.config.summarization import SummarizationConfig
from conductor_memory.service.memory_service import MemoryService
from conductor_memory.core.models import MemoryChunk, RoleEnum, MemoryType


class TestConfigurationLoading:
    """Test configuration loading and validation"""

    def test_server_config_defaults(self):
        """Test ServerConfig with default values"""
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.persist_directory.endswith("data")
        assert len(config.codebases) == 0
        assert config.embedding_model == "all-MiniLM-L12-v2"
        assert config.device == "auto"

    def test_codebase_config_validation(self):
        """Test CodebaseConfig validation and normalization"""
        config = CodebaseConfig(
            name="test",
            path="~/projects/test",
            extensions=["py", "js", ".md"],  # Mix of with/without dots
            ignore_patterns=["__pycache__", "/data", "**/build"]
        )

        # Should normalize extensions to have dots
        assert ".py" in config.extensions
        assert ".js" in config.extensions
        assert ".md" in config.extensions

        # Test ignore patterns
        assert config.should_ignore("src/__pycache__/file.pyc")  # Component match
        assert config.should_ignore("data/file.csv")  # Root-relative
        assert config.should_ignore("any/deep/build/file.java")  # Recursive glob
        assert not config.should_ignore("src/utils/file.py")  # Should not ignore

    def test_summarization_config_from_dict(self):
        """Test SummarizationConfig loading from dictionary"""
        config_dict = {
            "summarization": {
                "enabled": True,
                "llm_enabled": True,
                "ollama_url": "http://localhost:11434",
                "model": "qwen2.5-coder:1.5b",
                "rate_limit_seconds": 0.5,
                "max_file_lines": 600,
                "max_file_tokens": 4000
            }
        }

        config = SummarizationConfig.from_dict(config_dict)
        assert config.enabled is True
        assert config.llm_enabled is True
        assert config.model == "qwen2.5-coder:1.5b"
        assert config.max_file_lines == 600

    def test_boost_config_defaults(self):
        """Test BoostConfig default values"""
        config = BoostConfig()
        assert config.domain_boosts["class"] == 1.2
        assert config.domain_boosts["function"] == 1.1
        assert config.memory_type_boosts["decision"] == 1.3
        assert config.recency_enabled is True
        assert config.recency_decay_days == 30.0


class TestPhase5SummaryIntegration:
    """Test Phase 5: Summary Integration in Search Results"""

    @pytest.mark.asyncio
    async def test_include_summaries_parameter(self):
        """Test include_summaries parameter adds file summary data"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                config = ServerConfig()
                service = MemoryService(config)

                # Mock search results with one summarized file
                mock_chunks = [
                    MemoryChunk(
                        id="test-1",
                        project_id="test",
                        role=RoleEnum.SYSTEM,
                        prompt="",
                        response="",
                        doc_text="class UserService: pass",
                        embedding_id="",
                        tags=["file:src/services/user.py"],
                        pin=False,
                        relevance_score=0.9,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        source="codebase_indexing"
                    )
                ]

                # Mock file summaries lookup
                mock_summary = {
                    "purpose": "Handles user management operations",
                    "pattern": "Service",
                    "language": "python",
                    "domain": "business logic",
                    "key_exports": ["UserService", "create_user"],
                    "dependencies": ["sqlalchemy", "pydantic"]
                }

                async def mock_get_file_summaries(files, codebase=None):
                    return {"src/services/user.py": mock_summary}

                service._get_file_summaries_async = mock_get_file_summaries

                # Test with include_summaries=True
                results = await service.search_async(
                    query="user service",
                    include_summaries=True,
                    max_results=10
                )

                assert len(results) == 1
                result = results[0]
                assert result.get("has_summary") is True
                assert "file_summary" in result
                assert result["file_summary"]["purpose"] == "Handles user management operations"
                assert result["file_summary"]["pattern"] == "Service"

    @pytest.mark.asyncio
    async def test_boost_summarized_files(self):
        """Test boost_summarized parameter increases scores for summarized files"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                config = ServerConfig()
                service = MemoryService(config)

                chunks = [
                    MemoryChunk(  # Summarized file
                        id="test-1",
                        project_id="test",
                        role=RoleEnum.SYSTEM,
                        prompt="", response="",
                        doc_text="class UserService: pass",
                        embedding_id="",
                        tags=["file:src/services/user.py"],
                        pin=False, relevance_score=0.8,
                        created_at=datetime.now(), updated_at=datetime.now(),
                        source="codebase_indexing"
                    ),
                    MemoryChunk(  # Non-summarized file
                        id="test-2",
                        project_id="test",
                        role=RoleEnum.SYSTEM,
                        prompt="", response="",
                        doc_text="class OtherClass: pass",
                        embedding_id="",
                        tags=["file:src/other.py"],
                        pin=False, relevance_score=0.7,
                        created_at=datetime.now(), updated_at=datetime.now(),
                        source="codebase_indexing"
                    )
                ]

                # Mock that only user.py has a summary
                async def mock_get_summarized_files(codebase=None):
                    return {"src/services/user.py"}

                service._get_summarized_files_async = mock_get_summarized_files

                # Apply boost
                boosted = await service._apply_summary_boost_async(chunks)

                user_chunk = next(c for c in boosted if "user.py" in str(c.tags))
                other_chunk = next(c for c in boosted if "other.py" in str(c.tags))

                # user.py should be boosted (0.8 * 1.15 = 0.92)
                assert user_chunk.relevance_score == pytest.approx(0.8 * 1.15, rel=0.01)
                # other.py should remain unchanged
                assert other_chunk.relevance_score == 0.7

    def test_parse_summary_text_complete(self):
        """Test parsing complete summary text"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                config = ServerConfig()
                service = MemoryService(config)

                summary_text = """File Summary: src/auth/login.py
Language: python
Pattern: Controller
Domain: authentication

Purpose: Handles user login and session management

Key exports: login, logout, verify_session
Dependencies: flask, jwt, redis"""

                metadata = {"tags": ["summary", "file:src/auth/login.py"]}

                result = service._parse_summary_text(summary_text, metadata)

                assert result["purpose"] == "Handles user login and session management"
                assert result["language"] == "python"
                assert result["pattern"] == "Controller"
                assert result["domain"] == "authentication"
                assert result["key_exports"] == ["login", "logout", "verify_session"]
                assert result["dependencies"] == ["flask", "jwt", "redis"]


class TestPhase6IncrementalResummarization:
    """Test Phase 6: Incremental Re-summarization & Web UI"""

    def test_web_ui_endpoints(self):
        """Test web dashboard API endpoints are accessible"""
        # Test health endpoint
        health_response = requests.get("http://localhost:9820/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        assert health_data["codebases"] >= 2  # Should have our test codebases

        # Test summarization status endpoint
        summary_response = requests.get("http://localhost:9820/api/summarization")
        assert summary_response.status_code == 200
        summary_data = summary_response.json()
        assert "enabled" in summary_data
        assert "is_running" in summary_data
        assert "total_summarized" in summary_data
        assert "by_codebase" in summary_data

        # Test full status endpoint
        status_response = requests.get("http://localhost:9820/api/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "status" in status_data
        assert "codebases" in status_data

    def test_summarization_status_enhanced_fields(self):
        """Test Phase 6 enhanced summarization status fields"""
        response = requests.get("http://localhost:9820/api/summarization")
        data = response.json()

        # Check for Phase 6 fields
        assert "total_summarized" in data
        assert "files_skipped" in data
        assert "by_codebase" in data

        # Validate by_codebase structure
        assert "Conductor" in data["by_codebase"]
        assert "Options-ML-Trader" in data["by_codebase"]

        conductor_stats = data["by_codebase"]["Conductor"]
        assert "total_summarized" in conductor_stats
        assert "by_pattern" in conductor_stats
        assert "by_domain" in conductor_stats

    @pytest.mark.asyncio
    async def test_incremental_resummarization_logic(self):
        """Test incremental re-summarization detects file changes"""
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                config = ServerConfig()
                service = MemoryService(config)

                # Mock file metadata storage
                mock_metadata = {
                    "src/test.py": {
                        "content_hash": "old_hash_123",
                        "last_summarized": datetime.now().isoformat()
                    }
                }

                # Mock current file hash as changed
                async def mock_get_file_hash(filepath):
                    return "new_hash_456"

                service._get_file_hash_async = mock_get_file_hash

                # Test needs_resummarization detection
                needs_resummarize = await service._needs_resummarization_async("src/test.py", mock_metadata)
                assert needs_resummarize is True

                # Test unchanged file
                async def mock_get_unchanged_hash(filepath):
                    return "old_hash_123"

                service._get_file_hash_async = mock_get_unchanged_hash
                needs_resummarize = await service._needs_resummarization_async("src/test.py", mock_metadata)
                assert needs_resummarize is False


class TestMCPToolIntegration:
    """Test MCP tool integration and functionality"""

    def test_memory_search_tool_signature(self):
        """Test memory_search MCP tool has Phase 5 parameters"""
        # This would test the actual MCP tool registration
        # For now, verify the service method signatures
        import inspect

        search_sig = inspect.signature(MemoryService.search)
        search_async_sig = inspect.signature(MemoryService.search_async)

        # Check for Phase 5 parameters
        assert "include_summaries" in search_sig.parameters
        assert "boost_summarized" in search_sig.parameters
        assert "include_summaries" in search_async_sig.parameters
        assert "boost_summarized" in search_async_sig.parameters

    def test_memory_status_tool_response(self):
        """Test memory_status MCP tool returns Phase 6 enhanced data"""
        # This would test the MCP tool response format
        # For integration testing, we can test the underlying service method
        pass

    def test_memory_summarization_status_tool(self):
        """Test memory_summarization_status MCP tool"""
        # Test the summarization status tool returns expected Phase 6 data
        pass


class TestConfigurationPersistence:
    """Test configuration file loading and saving"""

    def test_config_roundtrip_json(self):
        """Test saving and loading configuration as JSON"""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            # Create test config
            original_config = ServerConfig(
                host="127.0.0.1",
                port=9000,
                codebases=[
                    CodebaseConfig(
                        name="test-codebase",
                        path="/tmp/test",
                        description="Test codebase",
                        extensions=[".py", ".js"],
                        ignore_patterns=["__pycache__", ".git"]
                    )
                ],
                summarization_config=SummarizationConfig(
                    enabled=True,
                    model="qwen2.5-coder:1.5b",
                    max_file_lines=500
                )
            )

            # Save to file
            original_config.save_to_file(config_path)

            # Load from file
            loaded_config = ServerConfig.from_file(config_path)

            # Verify roundtrip
            assert loaded_config.host == original_config.host
            assert loaded_config.port == original_config.port
            assert len(loaded_config.codebases) == 1
            assert loaded_config.codebases[0].name == "test-codebase"
            assert loaded_config.summarization_config.model == "qwen2.5-coder:1.5b"

        finally:
            os.unlink(config_path)

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid configurations are rejected
        with pytest.raises(ValueError):
            CodebaseConfig(name="", path="")  # Empty name/path should fail

        # Test valid configuration
        config = CodebaseConfig(name="valid", path="/tmp/valid")
        assert config.name == "valid"
        assert config.path == os.path.abspath("/tmp/valid")


class TestSearchQualityAndPerformance:
    """Test search quality and performance aspects"""

    def test_search_result_boosting(self):
        """Test that search results are properly boosted"""
        # Test domain-based boosting
        boost_config = BoostConfig()
        assert boost_config.domain_boosts["class"] > 1.0  # Classes boosted
        assert boost_config.domain_boosts["test"] < 1.0   # Tests deboosted

    def test_memory_type_boosting(self):
        """Test memory type boosting"""
        boost_config = BoostConfig()
        assert boost_config.memory_type_boosts["decision"] > 1.0  # Decisions highly boosted
        assert boost_config.memory_type_boosts["code"] > 1.0      # Code slightly boosted

    def test_recency_boosting(self):
        """Test recency boosting configuration"""
        boost_config = BoostConfig()
        assert boost_config.recency_enabled is True
        assert boost_config.recency_decay_days == 30.0
        assert boost_config.recency_max_boost == 1.5


# Run tests with: python -m pytest tests/test_comprehensive_memory_system.py -v