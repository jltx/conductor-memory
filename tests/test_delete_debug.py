"""Debug script to test memory store and delete operations"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import chromadb
from pathlib import Path

# Use the same persist directory as the MCP server
persist_dir = Path.home() / ".conductor-memory"
print(f"Persist directory: {persist_dir}")

# Connect to the same Chroma instance
client = chromadb.PersistentClient(path=str(persist_dir))

# List all collections
print("\n=== All Collections ===")
collections = client.list_collections()
for col in collections:
    print(f"  - {col.name}: {col.count()} documents")

# Let's check the Conductor codebase collection
print("\n=== Checking codebase_Conductor ===")
try:
    conductor_col = client.get_collection("codebase_Conductor")
    print(f"Collection found with {conductor_col.count()} documents")
    
    # Get recent entries (last 10)
    result = conductor_col.get(limit=10, include=["metadatas", "documents"])
    print(f"\nLast 10 document IDs:")
    for i, doc_id in enumerate(result["ids"]):
        meta = result["metadatas"][i] if result["metadatas"] else {}
        doc = result["documents"][i][:80] if result["documents"] else ""
        source = meta.get("source", "unknown")
        print(f"  {i+1}. ID: {doc_id[:40]}...")
        print(f"      Source: {source}")
        print(f"      Doc: {doc}...")
        print()
        
    # Search for our test entries by looking for "opencode" source
    print("\n=== Searching for opencode-sourced memories ===")
    result = conductor_col.get(
        where={"source": "opencode"},
        include=["metadatas", "documents"]
    )
    print(f"Found {len(result['ids'])} memories from opencode")
    for i, doc_id in enumerate(result["ids"][:5]):
        doc = result["documents"][i][:100] if result["documents"] else ""
        print(f"  ID: {doc_id}")
        print(f"  Doc: {doc}...")
        print()
    
    # Search for "TEST DECISION" in documents
    print("\n=== Searching for TEST in documents ===")
    all_docs = conductor_col.get(include=["documents", "metadatas"])
    for i, doc in enumerate(all_docs["documents"] or []):
        if "TEST" in doc.upper() or "MCP Memory Test" in doc:
            print(f"  Found: ID={all_docs['ids'][i]}")
            print(f"  Source: {all_docs['metadatas'][i].get('source', 'N/A')}")
            print(f"  Doc: {doc[:100]}...")
            print()
            
    # Try to get one of the specific IDs we stored
    print("\n=== Trying specific ID lookup ===")
    test_id = "709c7e01-8665-48f9-b99e-234e73674f30"  # From our test
    result = conductor_col.get(ids=[test_id])
    print(f"Lookup for {test_id}:")
    print(f"  Result IDs: {result['ids']}")
    
    # Test what happens with non-existent ID lookup
    print("\n=== Testing collection.get() behavior for non-existent ID ===")
    try:
        fake_id = "non-existent-uuid-12345"
        result = conductor_col.get(ids=[fake_id])
        print(f"Result for non-existent ID: {result}")
        print(f"  ids: {result['ids']}")
        print(f"  ids is truthy: {bool(result['ids'])}")
    except Exception as e:
        print(f"Exception thrown: {type(e).__name__}: {e}")
    
    # Now let's simulate what the MCP server does - store and retrieve
    print("\n=== Simulating MCP store/retrieve ===")
    import uuid
    test_uuid = str(uuid.uuid4())
    print(f"Storing with ID: {test_uuid}")
    
    # Need embeddings - let's use a simple approach
    # The embedding model creates 384-dimensional vectors for all-MiniLM-L12-v2
    fake_embedding = [0.1] * 384  
    
    conductor_col.add(
        ids=[test_uuid],
        embeddings=[fake_embedding],
        metadatas=[{"source": "test_script", "role": "user"}],
        documents=["This is a test document from debug script"]
    )
    print(f"Stored successfully")
    
    # Now try to retrieve it
    result = conductor_col.get(ids=[test_uuid])
    print(f"Immediate lookup result: {result['ids']}")
    
    # Check collection count
    print(f"Collection now has {conductor_col.count()} documents")
    
    # Try querying for the document
    query_result = conductor_col.query(
        query_embeddings=[fake_embedding],
        n_results=5
    )
    print(f"Query result IDs: {query_result['ids']}")
        
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
