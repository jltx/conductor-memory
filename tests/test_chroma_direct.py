"""Test ChromaDB add/get/query/delete with same client"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import chromadb
from pathlib import Path
import uuid

persist_dir = str(Path.home() / ".conductor-memory")
print(f"Persist directory: {persist_dir}")

# Create client
client = chromadb.PersistentClient(path=persist_dir)
print(f"Client created")

# Get collection
col = client.get_collection("codebase_Conductor")
print(f"Collection: {col.name}, count: {col.count()}")

# Generate test data
test_id = str(uuid.uuid4())
test_embedding = [0.1] * 384  # Same dimension as all-MiniLM-L12-v2
test_doc = f"CHROMA DIRECT TEST {test_id}"

print(f"\n=== Adding document ===")
print(f"ID: {test_id}")
print(f"Doc: {test_doc}")

col.add(
    ids=[test_id],
    embeddings=[test_embedding],
    documents=[test_doc],
    metadatas=[{"source": "direct_test", "project_id": "test"}]
)
print("Add completed")

print(f"\n=== Immediate get by ID ===")
result = col.get(ids=[test_id])
print(f"Result IDs: {result['ids']}")
print(f"Found: {len(result['ids']) > 0}")

print(f"\n=== Query by embedding ===")
query_result = col.query(
    query_embeddings=[test_embedding],
    n_results=3
)
print(f"Query result IDs: {query_result['ids'][0]}")
found_in_query = test_id in query_result['ids'][0]
print(f"Found in query: {found_in_query}")

print(f"\n=== Delete by ID ===")
if result['ids']:
    col.delete(ids=[test_id])
    print("Delete completed")
    
    # Verify deletion
    verify = col.get(ids=[test_id])
    print(f"After delete, get returns: {verify['ids']}")
else:
    print("Skipping delete since get didn't find it")

print(f"\n=== Final count ===")
print(f"Collection count: {col.count()}")
