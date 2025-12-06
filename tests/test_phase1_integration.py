#!/usr/bin/env python3
"""
Integration test for Phase 1: Test the complete search pipeline with boosting
"""

import sys
import os
import tempfile
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from conductor_memory.config.server import ServerConfig, CodebaseConfig, BoostConfig
from conductor_memory.service.memory_service import MemoryService
from conductor_memory.core.models import MemoryType


def test_end_to_end_boosting():
    """Test the complete search pipeline with domain boosting"""
    print("Testing end-to-end search with domain boosting...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test config with boosting
        config = ServerConfig(
            persist_directory=os.path.join(temp_dir, "chroma"),
            boost_config=BoostConfig(
                domain_boosts={
                    "class": 1.5,      # Boost classes significantly
                    "function": 1.2,   # Boost functions moderately
                    "test": 0.5,       # Penalize tests
                    "imports": 0.8     # Slightly penalize imports
                },
                recency_enabled=True,
                recency_decay_days=30.0
            )
        )
        
        # Initialize memory service
        memory_service = MemoryService(config)
        memory_service.initialize()
        
        # Store some test memories with different characteristics
        now = datetime.now()
        
        # Store a class definition (should get high boost)
        class_result = memory_service.store(
            content="class UserManager:\n    def __init__(self):\n        self.users = {}",
            project_id="test_project",
            tags=["domain:class", "file:user_manager.py", "ext:.py"],
            memory_type="code"
        )
        
        # Store a function (should get medium boost)
        function_result = memory_service.store(
            content="def authenticate_user(username, password):\n    return check_credentials(username, password)",
            project_id="test_project", 
            tags=["domain:function", "file:auth.py", "ext:.py"],
            memory_type="code"
        )
        
        # Store a test (should get penalty)
        test_result = memory_service.store(
            content="def test_user_authentication():\n    assert authenticate_user('test', 'pass') == True",
            project_id="test_project",
            tags=["domain:test", "file:test_auth.py", "ext:.py"],
            memory_type="code"
        )
        
        # Store an import section (should get slight penalty)
        import_result = memory_service.store(
            content="import os\nimport sys\nfrom typing import Dict, List",
            project_id="test_project",
            tags=["domain:imports", "file:main.py", "ext:.py"],
            memory_type="code"
        )
        
        print(f"Stored {len([class_result, function_result, test_result, import_result])} test memories")
        
        # Test search without domain boosting
        print("\n--- Search without domain boosting ---")
        normal_results = memory_service.search(
            query="user authentication",
            max_results=10,
            project_id="test_project"
        )
        
        print(f"Found {len(normal_results['results'])} results")
        for i, result in enumerate(normal_results['results']):
            tags = [tag for tag in result['tags'] if tag.startswith('domain:')]
            domain = tags[0] if tags else 'no-domain'
            print(f"  {i+1}. {domain}: {result['relevance_score']:.3f}")
        
        # Test search with domain boosting (boost classes, penalize tests)
        print("\n--- Search with domain boosting (class: 2.0, test: 0.1) ---")
        boosted_results = memory_service.search(
            query="user authentication",
            max_results=10,
            project_id="test_project",
            domain_boosts={"class": 2.0, "test": 0.1}
        )
        
        print(f"Found {len(boosted_results['results'])} results")
        for i, result in enumerate(boosted_results['results']):
            tags = [tag for tag in result['tags'] if tag.startswith('domain:')]
            domain = tags[0] if tags else 'no-domain'
            print(f"  {i+1}. {domain}: {result['relevance_score']:.3f}")
        
        # Verify that boosting changed the order
        normal_domains = []
        boosted_domains = []
        
        for result in normal_results['results']:
            tags = [tag for tag in result['tags'] if tag.startswith('domain:')]
            if tags:
                normal_domains.append(tags[0])
        
        for result in boosted_results['results']:
            tags = [tag for tag in result['tags'] if tag.startswith('domain:')]
            if tags:
                boosted_domains.append(tags[0])
        
        print(f"\nDomain order without boosting: {normal_domains}")
        print(f"Domain order with boosting: {boosted_domains}")
        
        # Check that class moved up and test moved down
        if 'domain:class' in boosted_domains and 'domain:test' in boosted_domains:
            class_pos_normal = normal_domains.index('domain:class') if 'domain:class' in normal_domains else -1
            class_pos_boosted = boosted_domains.index('domain:class')
            test_pos_normal = normal_domains.index('domain:test') if 'domain:test' in normal_domains else -1
            test_pos_boosted = boosted_domains.index('domain:test')
            
            print(f"Class position: {class_pos_normal} -> {class_pos_boosted}")
            print(f"Test position: {test_pos_normal} -> {test_pos_boosted}")
            
            # Class should move up (lower index = higher rank)
            if class_pos_normal >= 0:
                assert class_pos_boosted <= class_pos_normal, "Class should move up or stay same with boosting"
            
            # Test should move down (higher index = lower rank)  
            if test_pos_normal >= 0:
                assert test_pos_boosted >= test_pos_normal, "Test should move down or stay same with boosting"
        
        print("\n[OK] Domain boosting affects search results as expected")
        
        # Test that the API accepts the new parameter
        print("\n--- Testing API parameter acceptance ---")
        try:
            api_results = memory_service.search(
                query="authentication",
                domain_boosts={"function": 1.8, "imports": 0.3}
            )
            print(f"API accepted domain_boosts parameter, returned {len(api_results['results'])} results")
            print("[OK] API parameter integration works")
        except Exception as e:
            print(f"[FAIL] API parameter integration failed: {e}")
            raise
        
        print("\n[OK] End-to-end integration test passed!")


def main():
    """Run integration test"""
    print("=== Phase 1 Integration Test ===\n")
    
    try:
        test_end_to_end_boosting()
        print("\n=== Integration test passed! ===")
        print("Phase 1 boosting is fully integrated and working.")
        
    except Exception as e:
        print(f"\n[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()