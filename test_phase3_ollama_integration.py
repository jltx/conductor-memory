#!/usr/bin/env python3
"""
Test Phase 3: Ollama Integration

Tests the LLM client and file summarizer functionality.
"""

import asyncio
import json
import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from conductor_memory.llm.ollama_client import OllamaClient
from conductor_memory.llm.summarizer import FileSummarizer
from conductor_memory.config.summarization import SummarizationConfig
from conductor_memory.search.heuristics import HeuristicExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ollama_health():
    """Test Ollama service health check."""
    print("Testing Ollama Health Check...")
    
    # Use default model (qwen2.5-coder:1.5b)
    client = OllamaClient()
    
    try:
        is_healthy = await client.health_check()
        if is_healthy:
            print("PASS: Ollama is running and model is available")
            
            # List available models
            models = await client.list_models()
            print(f"Available models: {models}")
            
            return True
        else:
            print("FAIL: Ollama health check failed")
            print("   Make sure Ollama is running: ollama serve")
            print("   And the model is pulled: ollama pull qwen2.5-coder:7b-instruct-q4_K_M")
            return False
            
    except Exception as e:
        print(f"FAIL: Ollama connection error: {e}")
        return False
    finally:
        await client.close()


async def test_simple_generation():
    """Test basic LLM text generation."""
    print("\nTesting Simple LLM Generation...")
    
    # Use default model (qwen2.5-coder:1.5b)
    client = OllamaClient()
    
    try:
        response = await client.generate(
            prompt="What is the purpose of a Python __init__.py file?",
            system_prompt="You are a helpful coding assistant. Provide concise, accurate answers.",
            temperature=0.1,
            max_tokens=100
        )
        
        print(f"PASS: LLM Response:")
        print(f"   Model: {response.model}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Time: {response.response_time_ms:.1f}ms")
        print(f"   Content: {response.content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"FAIL: LLM generation failed: {e}")
        return False
    finally:
        await client.close()


async def test_file_summarization():
    """Test file summarization with real Python code."""
    print("\nTesting File Summarization...")
    
    # Create test Python file content
    test_file_content = '''"""
User service module for handling user operations.

This module provides the UserService class for managing user data,
authentication, and user-related business logic.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User data model."""
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True


class UserService:
    """Service class for user operations."""
    
    def __init__(self, database_url: str):
        """Initialize user service with database connection."""
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)
    
    async def create_user(self, username: str, email: str) -> User:
        """
        Create a new user account.
        
        Args:
            username: Unique username
            email: User email address
            
        Returns:
            Created User object
            
        Raises:
            ValueError: If username or email already exists
        """
        self.logger.info(f"Creating user: {username}")
        
        # Validate input
        if not username or not email:
            raise ValueError("Username and email are required")
        
        # Check if user exists
        existing_user = await self.get_user_by_username(username)
        if existing_user:
            raise ValueError(f"Username {username} already exists")
        
        # Create user
        user = User(
            id=self._generate_id(),
            username=username,
            email=email,
            created_at=datetime.now()
        )
        
        # Save to database (simulated)
        await self._save_user(user)
        
        self.logger.info(f"User created successfully: {user.id}")
        return user
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        # Database query simulation
        return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        # Database query simulation
        return None
    
    async def update_user(self, user_id: int, updates: Dict[str, Any]) -> User:
        """Update user information."""
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        await self._save_user(user)
        return user
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Soft delete
        user.is_active = False
        await self._save_user(user)
        
        self.logger.info(f"User {user_id} deleted")
        return True
    
    def _generate_id(self) -> int:
        """Generate unique user ID."""
        import random
        return random.randint(1000, 9999)
    
    async def _save_user(self, user: User) -> None:
        """Save user to database."""
        # Database save simulation
        pass
'''
    
    try:
        # Initialize components - use default model (qwen2.5-coder:1.5b)
        client = OllamaClient()
        config = SummarizationConfig()
        summarizer = FileSummarizer(client, config)
        
        # Extract heuristics for context
        extractor = HeuristicExtractor()
        heuristics = extractor.extract_file_metadata(test_file_content, "test_user_service.py")
        
        print(f"Heuristic metadata extracted:")
        if heuristics:
            print(f"   Classes: {len(heuristics.classes)}")
            print(f"   Functions: {len(heuristics.functions)}")
            print(f"   Methods: {len(heuristics.methods)}")
            print(f"   Imports: {len(heuristics.imports)}")
        else:
            print("   No heuristics extracted (None returned)")
            heuristics = None
        
        # Generate summary
        print(f"\nGenerating LLM summary...")
        summary = await summarizer.summarize_file(
            file_path="test_user_service.py",
            content=test_file_content,
            heuristic_metadata=heuristics
        )
        
        print(f"PASS: File Summary Generated:")
        print(f"   Purpose: {summary.purpose}")
        print(f"   Pattern: {summary.pattern}")
        print(f"   Key Exports: {summary.key_exports}")
        print(f"   Dependencies: {summary.dependencies}")
        print(f"   Domain: {summary.domain}")
        print(f"   Model: {summary.model_used}")
        print(f"   Tokens: {summary.tokens_used}")
        if summary.response_time_ms:
            print(f"   Time: {summary.response_time_ms:.1f}ms")
        else:
            print(f"   Time: N/A")
        print(f"   Is Skeleton: {summary.is_skeleton}")
        
        if summary.error:
            print(f"   Error: {summary.error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"FAIL: File summarization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await client.close()


async def test_large_file_skeleton():
    """Test skeleton extraction for large files."""
    print("\nTesting Large File Skeleton Extraction...")
    
    # Create a large file content (simulate 700+ lines)
    large_file_content = '''"""
Large service module with many functions.
This file exceeds the line limit and should trigger skeleton extraction.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

''' + '\n'.join([f'''
class Service{i}:
    """Service class {i}."""
    
    def __init__(self):
        """Initialize service {i}."""
        pass
    
    async def method_{i}_1(self, param: str) -> str:
        """Method {i}_1 with implementation."""
        # Lots of implementation code here
        result = param.upper()
        for j in range(10):
            result += f"_{{j}}"
        return result
    
    async def method_{i}_2(self, param: int) -> int:
        """Method {i}_2 with implementation."""
        # More implementation code
        result = param * 2
        for j in range(5):
            result += j
        return result
''' for i in range(1, 50)])  # Creates ~700+ lines
    
    try:
        # Use default model (qwen2.5-coder:1.5b)
        client = OllamaClient()
        config = SummarizationConfig(max_file_lines=600)  # Force skeleton extraction
        summarizer = FileSummarizer(client, config)
        
        print(f"Large file stats:")
        print(f"   Lines: {len(large_file_content.split(chr(10)))}")
        print(f"   Characters: {len(large_file_content)}")
        print(f"   Estimated tokens: {len(large_file_content.split())}")
        
        # Generate summary (should use skeleton)
        summary = await summarizer.summarize_file(
            file_path="large_service.py",
            content=large_file_content
        )
        
        print(f"PASS: Large File Summary:")
        print(f"   Purpose: {summary.purpose}")
        print(f"   Pattern: {summary.pattern}")
        print(f"   Is Skeleton: {summary.is_skeleton}")
        print(f"   Model: {summary.model_used}")
        print(f"   Time: {summary.response_time_ms:.1f}ms")
        
        if summary.error:
            print(f"   Error: {summary.error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"FAIL: Large file summarization failed: {e}")
        return False
    finally:
        await client.close()


async def main():
    """Run all Phase 3 tests."""
    print("Phase 3: Ollama Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Ollama Health Check", test_ollama_health),
        ("Simple LLM Generation", test_simple_generation),
        ("File Summarization", test_file_summarization),
        ("Large File Skeleton", test_large_file_skeleton),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"FAIL: {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("All Phase 3 tests passed! Ollama integration is working.")
    else:
        print("Some tests failed. Check Ollama setup and model availability.")
    
    return passed == len(results)


if __name__ == "__main__":
    asyncio.run(main())