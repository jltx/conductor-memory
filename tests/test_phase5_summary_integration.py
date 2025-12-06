"""
Tests for Phase 5: Summary Integration in Search Results

Tests the include_summaries parameter and boost_summarized functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from conductor_memory.core.models import MemoryChunk, RoleEnum, MemoryType
from conductor_memory.service.memory_service import MemoryService


class TestExtractFilePath:
    """Test _extract_file_path_from_chunk helper"""
    
    def test_extract_file_path_present(self):
        """Should extract file path from chunk tags"""
        chunk = MemoryChunk(
            id="test-1",
            project_id="test",
            role=RoleEnum.SYSTEM,
            prompt="",
            response="",
            doc_text="def hello(): pass",
            embedding_id="",
            tags=["file:src/hello.py", "ext:.py", "codebase:test"],
            pin=False,
            relevance_score=0.9,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="codebase_indexing"
        )
        
        # Create minimal service mock
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.config.server import ServerConfig
                config = ServerConfig()
                service = MemoryService(config)
                
                result = service._extract_file_path_from_chunk(chunk)
                assert result == "src/hello.py"
    
    def test_extract_file_path_missing(self):
        """Should return None if no file tag present"""
        chunk = MemoryChunk(
            id="test-1",
            project_id="test",
            role=RoleEnum.USER,
            prompt="",
            response="",
            doc_text="Hello world",
            embedding_id="",
            tags=["conversation", "user"],
            pin=False,
            relevance_score=0.5,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="api"
        )
        
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.config.server import ServerConfig
                config = ServerConfig()
                service = MemoryService(config)
                
                result = service._extract_file_path_from_chunk(chunk)
                assert result is None


class TestParseSummaryText:
    """Test _parse_summary_text helper"""
    
    def test_parse_complete_summary(self):
        """Should parse all fields from summary text"""
        summary_text = """File Summary: src/auth/login.py
Language: python
Pattern: Controller
Domain: authentication

Purpose: Handles user login and session management

Key exports: login, logout, verify_session
Dependencies: flask, jwt, redis"""
        
        metadata = {"tags": ["summary", "file:src/auth/login.py"]}
        
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.config.server import ServerConfig
                config = ServerConfig()
                service = MemoryService(config)
                
                result = service._parse_summary_text(summary_text, metadata)
                
                assert result["purpose"] == "Handles user login and session management"
                assert result["language"] == "python"
                assert result["pattern"] == "Controller"
                assert result["domain"] == "authentication"
                assert result["key_exports"] == ["login", "logout", "verify_session"]
                assert result["dependencies"] == ["flask", "jwt", "redis"]
    
    def test_parse_minimal_summary(self):
        """Should handle missing fields gracefully"""
        summary_text = """File Summary: src/utils.py
Purpose: Utility functions"""
        
        metadata = {"tags": []}
        
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.config.server import ServerConfig
                config = ServerConfig()
                service = MemoryService(config)
                
                result = service._parse_summary_text(summary_text, metadata)
                
                assert result["purpose"] == "Utility functions"
                assert result["raw_summary"] == summary_text
                assert "language" not in result or result.get("language") is None


class TestSummaryBoost:
    """Test summary boost logic"""
    
    @pytest.mark.asyncio
    async def test_boost_applied_to_summarized_files(self):
        """Should apply boost to chunks from summarized files"""
        chunks = [
            MemoryChunk(
                id="test-1",
                project_id="test",
                role=RoleEnum.SYSTEM,
                prompt="",
                response="",
                doc_text="class UserService:",
                embedding_id="",
                tags=["file:src/services/user.py"],
                pin=False,
                relevance_score=0.8,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="codebase_indexing"
            ),
            MemoryChunk(
                id="test-2",
                project_id="test",
                role=RoleEnum.SYSTEM,
                prompt="",
                response="",
                doc_text="class OtherClass:",
                embedding_id="",
                tags=["file:src/other.py"],
                pin=False,
                relevance_score=0.7,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="codebase_indexing"
            ),
        ]
        
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.config.server import ServerConfig
                config = ServerConfig()
                service = MemoryService(config)
                
                # Mock that only user.py has a summary
                async def mock_get_summarized_files(codebase=None):
                    return {"src/services/user.py"}
                
                service._get_summarized_files_async = mock_get_summarized_files
                
                # Apply boost
                result = await service._apply_summary_boost_async(chunks)
                
                # user.py chunk should be boosted
                user_chunk = next(c for c in result if "user.py" in str(c.tags))
                other_chunk = next(c for c in result if "other.py" in str(c.tags))
                
                # user.py should now have higher score (0.8 * 1.15 = 0.92)
                assert user_chunk.relevance_score == pytest.approx(0.8 * 1.15, rel=0.01)
                # other.py should stay the same
                assert other_chunk.relevance_score == 0.7
    
    @pytest.mark.asyncio
    async def test_no_boost_when_no_summaries(self):
        """Should not modify scores when no summaries exist"""
        chunks = [
            MemoryChunk(
                id="test-1",
                project_id="test",
                role=RoleEnum.SYSTEM,
                prompt="",
                response="",
                doc_text="class UserService:",
                embedding_id="",
                tags=["file:src/services/user.py"],
                pin=False,
                relevance_score=0.8,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="codebase_indexing"
            ),
        ]
        
        with patch('conductor_memory.service.memory_service.SentenceTransformerEmbedder'):
            with patch('conductor_memory.service.memory_service.ChromaVectorStore'):
                from conductor_memory.config.server import ServerConfig
                config = ServerConfig()
                service = MemoryService(config)
                
                # Mock no summaries
                async def mock_get_summarized_files(codebase=None):
                    return set()
                
                service._get_summarized_files_async = mock_get_summarized_files
                
                original_score = chunks[0].relevance_score
                result = await service._apply_summary_boost_async(chunks)
                
                # Score should be unchanged
                assert result[0].relevance_score == original_score


class TestSearchWithSummaries:
    """Test search method with include_summaries parameter"""
    
    def test_search_signature_includes_new_params(self):
        """Verify search method has the new parameters"""
        import inspect
        sig = inspect.signature(MemoryService.search)
        params = list(sig.parameters.keys())
        
        assert "include_summaries" in params
        assert "boost_summarized" in params
    
    def test_search_async_signature_includes_new_params(self):
        """Verify search_async method has the new parameters"""
        import inspect
        sig = inspect.signature(MemoryService.search_async)
        params = list(sig.parameters.keys())
        
        assert "include_summaries" in params
        assert "boost_summarized" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
