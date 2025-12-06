#!/usr/bin/env python3
"""
Test script for Phase 1: Core Boosting implementation
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from conductor_memory.config.server import ServerConfig, BoostConfig
from conductor_memory.search.boosting import BoostCalculator
from conductor_memory.core.models import MemoryChunk, MemoryType, RoleEnum


def test_boost_config():
    """Test BoostConfig creation and serialization"""
    print("Testing BoostConfig...")
    
    # Test default config
    config = BoostConfig()
    print(f"Default domain boosts: {config.domain_boosts}")
    print(f"Default memory type boosts: {config.memory_type_boosts}")
    print(f"Recency enabled: {config.recency_enabled}")
    
    # Test serialization
    config_dict = config.to_dict()
    print(f"Serialized config keys: {list(config_dict.keys())}")
    
    # Test deserialization
    config2 = BoostConfig.from_dict(config_dict)
    assert config2.domain_boosts == config.domain_boosts
    assert config2.memory_type_boosts == config.memory_type_boosts
    print("[OK] BoostConfig serialization works")


def test_server_config_with_boost():
    """Test ServerConfig with BoostConfig integration"""
    print("\nTesting ServerConfig with BoostConfig...")
    
    # Test default server config includes boost config
    server_config = ServerConfig()
    assert hasattr(server_config, 'boost_config')
    assert isinstance(server_config.boost_config, BoostConfig)
    
    # Test serialization
    config_dict = server_config.to_dict()
    assert 'boost_config' in config_dict
    print(f"Server config includes boost_config: {'boost_config' in config_dict}")
    
    # Test deserialization
    server_config2 = ServerConfig.from_dict(config_dict)
    assert hasattr(server_config2, 'boost_config')
    assert server_config2.boost_config.domain_boosts == server_config.boost_config.domain_boosts
    print("[OK] ServerConfig with BoostConfig works")


def test_boost_calculator():
    """Test BoostCalculator functionality"""
    print("\nTesting BoostCalculator...")
    
    config = BoostConfig()
    calculator = BoostCalculator(config)
    
    # Create test chunks with different characteristics
    now = datetime.now()
    
    # Recent class chunk
    class_chunk = MemoryChunk(
        id="test_class_001",
        project_id="test",
        role=RoleEnum.ASSISTANT,
        prompt="",
        response="",
        doc_text="class TestClass:\n    def __init__(self):\n        pass",
        embedding_id="",
        tags=["domain:class", "file:test.py", "ext:.py"],
        pin=False,
        relevance_score=0.8,
        created_at=now - timedelta(days=1),  # Recent
        updated_at=now,
        source="codebase_indexing",
        memory_type=MemoryType.CODE
    )
    
    # Old test function chunk
    test_chunk = MemoryChunk(
        id="test_func_001",
        project_id="test",
        role=RoleEnum.ASSISTANT,
        prompt="",
        response="",
        doc_text="def test_something():\n    assert True",
        embedding_id="",
        tags=["domain:test", "file:test_file.py", "ext:.py"],
        pin=False,
        relevance_score=0.8,
        created_at=now - timedelta(days=60),  # Old
        updated_at=now,
        source="codebase_indexing",
        memory_type=MemoryType.CODE
    )
    
    # Decision chunk (should get high boost) - make it recent to test memory type boost
    decision_chunk = MemoryChunk(
        id="decision_001",
        project_id="test",
        role=RoleEnum.ASSISTANT,
        prompt="",
        response="",
        doc_text="DECISION: Use FastAPI for the web framework",
        embedding_id="",
        tags=["decision", "architecture"],
        pin=True,
        relevance_score=0.8,
        created_at=now - timedelta(hours=1),  # Very recent
        updated_at=now,
        source="manual",
        memory_type=MemoryType.DECISION
    )
    
    # Test individual boosts
    class_boost = calculator.calculate_boost(class_chunk)
    test_boost = calculator.calculate_boost(test_chunk)
    decision_boost = calculator.calculate_boost(decision_chunk)
    
    print(f"Class chunk boost: {class_boost:.3f}")
    print(f"Test chunk boost: {test_boost:.3f}")
    print(f"Decision chunk boost: {decision_boost:.3f}")
    
    # Verify expected relationships
    assert class_boost > test_boost, "Class should be boosted more than test"
    assert decision_boost > 1.0, "Decision should be boosted above baseline"
    assert test_boost < 1.0, "Test should be penalized"
    
    # Test memory type boost specifically by comparing similar age chunks
    # Decision memory type (1.3) should beat code memory type (1.1) when other factors are equal
    memory_type_decision_factor = config.memory_type_boosts["decision"]
    memory_type_code_factor = config.memory_type_boosts["code"]
    assert memory_type_decision_factor > memory_type_code_factor, "Decision memory type should have higher boost than code"
    
    print("[OK] BoostCalculator produces expected relative boosts")
    
    # Test query-specific domain boosts
    query_boosts = {"class": 2.0, "test": 0.1}
    class_boost_custom = calculator.calculate_boost(class_chunk, query_boosts)
    test_boost_custom = calculator.calculate_boost(test_chunk, query_boosts)
    
    print(f"Class chunk with custom boost: {class_boost_custom:.3f}")
    print(f"Test chunk with custom boost: {test_boost_custom:.3f}")
    
    assert class_boost_custom > class_boost, "Custom boost should increase class boost"
    assert test_boost_custom < test_boost, "Custom boost should decrease test boost"
    
    print("[OK] Query-specific domain boosts work")


def test_apply_boosts_to_chunks():
    """Test applying boosts to a list of chunks"""
    print("\nTesting apply_boosts_to_chunks...")
    
    config = BoostConfig()
    calculator = BoostCalculator(config)
    
    now = datetime.now()
    
    # Create chunks with same initial relevance score
    chunks = [
        MemoryChunk(
            id=f"chunk_{i}",
            project_id="test",
            role=RoleEnum.ASSISTANT,
            prompt="",
            response="",
            doc_text=f"Content {i}",
            embedding_id="",
            tags=[f"domain:{domain}", "file:test.py"],
            pin=False,
            relevance_score=0.5,  # Same initial score
            created_at=now - timedelta(days=i),  # Different ages
            updated_at=now,
            source="codebase_indexing",
            memory_type=MemoryType.CODE
        )
        for i, domain in enumerate(["class", "function", "test", "private"])
    ]
    
    # Store original scores
    original_scores = [chunk.relevance_score for chunk in chunks]
    
    # Apply boosts
    boosted_chunks = calculator.apply_boosts_to_chunks(chunks)
    
    # Verify scores changed
    new_scores = [chunk.relevance_score for chunk in boosted_chunks]
    print(f"Original scores: {original_scores}")
    print(f"Boosted scores: {[f'{s:.3f}' for s in new_scores]}")
    
    # Verify chunks are the same objects (modified in place)
    assert boosted_chunks is chunks, "Should modify chunks in place"
    
    # Verify relative ordering makes sense
    class_score = chunks[0].relevance_score  # domain:class
    test_score = chunks[2].relevance_score   # domain:test
    assert class_score > test_score, "Class should score higher than test after boosting"
    
    print("[OK] apply_boosts_to_chunks works correctly")


def main():
    """Run all tests"""
    print("=== Phase 1 Boosting Implementation Tests ===\n")
    
    try:
        test_boost_config()
        test_server_config_with_boost()
        test_boost_calculator()
        test_apply_boosts_to_chunks()
        
        print("\n=== All tests passed! ===")
        print("Phase 1 implementation is working correctly.")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()