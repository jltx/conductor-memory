#!/usr/bin/env python3
"""
Basic usage example for conductor-memory

This example demonstrates:
1. Creating a configuration
2. Initializing the memory service
3. Searching for relevant code
4. Storing and retrieving memories
"""

from pathlib import Path
from conductor_memory import MemoryService, ServerConfig, CodebaseConfig


def main():
    # Create configuration for your codebase
    config = ServerConfig(
        host="127.0.0.1",
        port=8000,
        persist_directory=str(Path.home() / ".conductor-memory" / "data"),
        codebases=[
            CodebaseConfig(
                name="my-project",
                path="/path/to/your/project",  # Change this!
                extensions=[".py", ".js", ".ts", ".md"],
                ignore_patterns=["__pycache__", ".git", "node_modules", "venv"]
            )
        ]
    )
    
    # Initialize the memory service (blocks until indexing completes)
    print("Initializing memory service...")
    service = MemoryService(config)
    service.initialize()
    print("Initialization complete!")
    
    # Check status
    status = service.get_status()
    print(f"Status: {status['status']}")
    print(f"Codebases: {list(status['codebases'].keys())}")
    
    # Search for code
    print("\n--- Searching for 'authentication' ---")
    results = service.search(
        query="how does authentication work",
        max_results=5,
        search_mode="auto"  # Let it auto-detect semantic vs keyword
    )
    
    print(f"Found {results['total_found']} results")
    print(f"Search mode used: {results.get('search_mode_used', 'unknown')}")
    
    for i, r in enumerate(results["results"], 1):
        print(f"\n{i}. Score: {r['relevance_score']:.3f}")
        print(f"   Tags: {', '.join(r['tags'][:3])}")
        print(f"   Content: {r['content'][:150]}...")
    
    # Store a memory (e.g., an architectural decision)
    print("\n--- Storing a decision ---")
    store_result = service.store(
        content="""
        DECISION: Use JWT tokens for API authentication
        CONTEXT: Need stateless auth for microservices
        ALTERNATIVES: Session cookies, OAuth2 only
        RATIONALE: JWTs are self-contained, no server state needed
        """,
        tags=["architecture", "auth", "api"],
        memory_type="decision"  # Auto-pinned
    )
    
    if store_result.get("success"):
        print(f"Stored decision with ID: {store_result['id']}")
    else:
        print(f"Failed to store: {store_result.get('error')}")
    
    # Search again - should find our new decision
    print("\n--- Searching again (should include our decision) ---")
    results = service.search("JWT authentication decision", max_results=3)
    
    for r in results["results"]:
        if "decision" in r["tags"]:
            print(f"Found decision: {r['content'][:100]}...")


if __name__ == "__main__":
    main()
