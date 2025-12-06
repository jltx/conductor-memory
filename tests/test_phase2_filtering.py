#!/usr/bin/env python3
"""
Phase 2 Tag Filtering Tests

Tests the include_tags and exclude_tags functionality.
"""

import sys
sys.path.insert(0, '../src')

from conductor_memory.service.memory_service import MemoryService
from conductor_memory.core.models import MemoryChunk, RoleEnum
from datetime import datetime


def test_tag_matches():
    """Test the _tag_matches helper method"""
    from conductor_memory.config.server import ServerConfig
    
    config = ServerConfig()
    config.persist_directory = "./data/test_phase2"
    service = MemoryService(config)
    
    # Test exact match
    chunk_tags = {"domain:class", "ext:.py", "file:test.py"}
    
    assert service._tag_matches("domain:class", chunk_tags) == True
    assert service._tag_matches("ext:.py", chunk_tags) == True
    assert service._tag_matches("domain:function", chunk_tags) == False
    assert service._tag_matches("nonexistent", chunk_tags) == False
    
    # Test prefix match with wildcard
    assert service._tag_matches("domain:*", chunk_tags) == True  # matches domain:class
    assert service._tag_matches("ext:*", chunk_tags) == True     # matches ext:.py
    assert service._tag_matches("file:*", chunk_tags) == True    # matches file:test.py
    assert service._tag_matches("module:*", chunk_tags) == False # no module: tags
    
    print("test_tag_matches PASSED")


def test_filter_by_tags_include():
    """Test include_tags filtering"""
    from conductor_memory.config.server import ServerConfig
    
    config = ServerConfig()
    config.persist_directory = "./data/test_phase2"
    service = MemoryService(config)
    
    # Create test chunks with different tags
    chunks = [
        MemoryChunk(
            id="1", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Class definition",
            embedding_id="", tags=["domain:class", "ext:.py"],
            pin=False, relevance_score=0.9,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="2", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Function definition",
            embedding_id="", tags=["domain:function", "ext:.py"],
            pin=False, relevance_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="3", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Test function",
            embedding_id="", tags=["domain:test", "ext:.py"],
            pin=False, relevance_score=0.7,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="4", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Java class",
            embedding_id="", tags=["domain:class", "ext:.java"],
            pin=False, relevance_score=0.6,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
    ]
    
    # Test include exact tag
    filtered = service._filter_by_tags(chunks, include_tags=["domain:class"], exclude_tags=None)
    assert len(filtered) == 2  # chunks 1 and 4
    assert {c.id for c in filtered} == {"1", "4"}
    
    # Test include with prefix wildcard
    filtered = service._filter_by_tags(chunks, include_tags=["ext:.py"], exclude_tags=None)
    assert len(filtered) == 3  # chunks 1, 2, 3
    assert {c.id for c in filtered} == {"1", "2", "3"}
    
    # Test include multiple tags (OR logic)
    filtered = service._filter_by_tags(chunks, include_tags=["domain:class", "domain:function"], exclude_tags=None)
    assert len(filtered) == 3  # chunks 1, 2, 4
    assert {c.id for c in filtered} == {"1", "2", "4"}
    
    print("test_filter_by_tags_include PASSED")


def test_filter_by_tags_exclude():
    """Test exclude_tags filtering"""
    from conductor_memory.config.server import ServerConfig
    
    config = ServerConfig()
    config.persist_directory = "./data/test_phase2"
    service = MemoryService(config)
    
    chunks = [
        MemoryChunk(
            id="1", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Class definition",
            embedding_id="", tags=["domain:class", "ext:.py"],
            pin=False, relevance_score=0.9,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="2", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Test class",
            embedding_id="", tags=["domain:test", "ext:.py"],
            pin=False, relevance_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="3", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Imports",
            embedding_id="", tags=["domain:imports", "ext:.py"],
            pin=False, relevance_score=0.7,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
    ]
    
    # Exclude test code
    filtered = service._filter_by_tags(chunks, include_tags=None, exclude_tags=["domain:test"])
    assert len(filtered) == 2
    assert {c.id for c in filtered} == {"1", "3"}
    
    # Exclude multiple tags
    filtered = service._filter_by_tags(chunks, include_tags=None, exclude_tags=["domain:test", "domain:imports"])
    assert len(filtered) == 1
    assert filtered[0].id == "1"
    
    print("test_filter_by_tags_exclude PASSED")


def test_filter_by_tags_combined():
    """Test combined include and exclude filtering"""
    from conductor_memory.config.server import ServerConfig
    
    config = ServerConfig()
    config.persist_directory = "./data/test_phase2"
    service = MemoryService(config)
    
    chunks = [
        MemoryChunk(
            id="1", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Python class",
            embedding_id="", tags=["domain:class", "ext:.py"],
            pin=False, relevance_score=0.9,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="2", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Python test",
            embedding_id="", tags=["domain:test", "ext:.py"],
            pin=False, relevance_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="3", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Java class",
            embedding_id="", tags=["domain:class", "ext:.java"],
            pin=False, relevance_score=0.7,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
    ]
    
    # Include Python files, exclude tests
    filtered = service._filter_by_tags(
        chunks, 
        include_tags=["ext:.py"], 
        exclude_tags=["domain:test"]
    )
    assert len(filtered) == 1
    assert filtered[0].id == "1"
    
    # Include any domain, exclude tests
    filtered = service._filter_by_tags(
        chunks, 
        include_tags=["domain:*"], 
        exclude_tags=["domain:test"]
    )
    assert len(filtered) == 2
    assert {c.id for c in filtered} == {"1", "3"}
    
    print("test_filter_by_tags_combined PASSED")


def test_filter_preserves_order():
    """Test that filtering preserves chunk order"""
    from conductor_memory.config.server import ServerConfig
    
    config = ServerConfig()
    config.persist_directory = "./data/test_phase2"
    service = MemoryService(config)
    
    chunks = [
        MemoryChunk(
            id="1", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="First",
            embedding_id="", tags=["keep"],
            pin=False, relevance_score=0.9,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="2", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Remove",
            embedding_id="", tags=["remove"],
            pin=False, relevance_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
        MemoryChunk(
            id="3", project_id="test", role=RoleEnum.SYSTEM,
            prompt="", response="", doc_text="Third",
            embedding_id="", tags=["keep"],
            pin=False, relevance_score=0.7,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="test"
        ),
    ]
    
    filtered = service._filter_by_tags(chunks, include_tags=["keep"], exclude_tags=None)
    assert len(filtered) == 2
    assert filtered[0].id == "1"  # Order preserved
    assert filtered[1].id == "3"
    
    print("test_filter_preserves_order PASSED")


def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n" + "="*60)
    print("Phase 2: Tag Filtering Tests")
    print("="*60 + "\n")
    
    test_tag_matches()
    test_filter_by_tags_include()
    test_filter_by_tags_exclude()
    test_filter_by_tags_combined()
    test_filter_preserves_order()
    
    print("\n" + "="*60)
    print("ALL PHASE 2 TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
