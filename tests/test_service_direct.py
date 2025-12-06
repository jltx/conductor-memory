"""Test MemoryService directly to isolate the issue"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from pathlib import Path
from conductor_memory.config.server import ServerConfig
from conductor_memory.service.memory_service import MemoryService

# Load the same config as the SSE server
config_path = Path.home() / ".conductor-memory" / "config.json"
print(f"Loading config from: {config_path}")

config = ServerConfig.from_file(str(config_path))
print(f"Codebases: {[cb.name for cb in config.codebases]}")

# Create memory service (this will load existing index, not reindex)
print("\nCreating MemoryService...")
service = MemoryService(config)
print(f"Vector stores: {list(service._vector_stores.keys())}")

# Get Conductor vector store
vs = service._vector_stores.get("Conductor")
print(f"\nConductor collection count: {vs.collection.count()}")

# Store a test memory
print("\n=== Storing test memory ===")
import asyncio

async def test():
    result = await service.store_async(
        content="DIRECT SERVICE TEST: Testing store via MemoryService directly",
        tags=["direct-test"],
        source="direct_test_script"
    )
    print(f"Store result: {result}")
    
    if result.get("success"):
        memory_id = result["id"]
        print(f"\nStored with ID: {memory_id}")
        
        # Immediately check via collection.get()
        print("\n=== Checking via collection.get() ===")
        get_result = vs.collection.get(ids=[memory_id])
        print(f"collection.get() result IDs: {get_result['ids']}")
        
        # Check via search
        print("\n=== Checking via search ===")
        search_result = await service.search_async(
            query="DIRECT SERVICE TEST",
            max_results=5
        )
        found_in_search = any(r["id"] == memory_id for r in search_result.get("results", []))
        print(f"Found in search: {found_in_search}")
        
        # Try delete
        print("\n=== Trying delete ===")
        delete_result = await service.delete_async(memory_id=memory_id)
        print(f"Delete result: {delete_result}")

asyncio.run(test())
