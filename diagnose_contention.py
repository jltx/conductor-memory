"""
Quick diagnostic to measure ChromaDB query times during active summarization.
Run this while the summarizer is processing to see if SQLite contention is the issue.
"""
import time
import chromadb
from pathlib import Path

# Config from your settings
CHROMA_PATH = Path.home() / ".conductor-memory" / "chroma"

def measure_count_time(client, collection_name: str, iterations: int = 5):
    """Measure average count() time for a collection."""
    try:
        collection = client.get_collection(collection_name)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = collection.count()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            time.sleep(0.1)  # Small delay between measurements

        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        return avg, min_t, max_t
    except Exception as e:
        return None, None, str(e)

def main():
    print("ChromaDB Contention Diagnostic")
    print("=" * 50)
    print(f"Path: {CHROMA_PATH}")
    print()

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collections = client.list_collections()
    print(f"Found {len(collections)} collections")
    print()

    # Test all found collections
    test_collections = [c.name for c in collections]

    print("Measuring count() times (5 iterations each):")
    print("-" * 50)

    for name in test_collections:
        avg, min_t, max_t = measure_count_time(client, name)
        if avg:
            print(f"{name}:")
            print(f"  avg: {avg:.1f}ms, min: {min_t:.1f}ms, max: {max_t:.1f}ms")
            if max_t > avg * 2:
                print(f"  WARNING: High variance suggests contention!")
        else:
            print(f"{name}: {max_t}")  # max_t contains error message
        print()

    print()
    print("If times are >100ms or highly variable, SQLite contention")
    print("from the active summarizer is likely the cause.")
    print()
    print("Solutions:")
    print("1. Wait for summarization to complete")
    print("2. Switch to chroma_mode: 'http' in config.json")
    print("3. Pause summarization during UI usage")

if __name__ == "__main__":
    main()
