"""Debug script to check if MCP server is even getting our store requests"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Let's check the actual persist dir being used and look for recent changes
from pathlib import Path
import os
from datetime import datetime

persist_dir = Path.home() / ".conductor-memory"
print(f"Persist directory: {persist_dir}")
print(f"Exists: {persist_dir.exists()}")

# List contents with modification times
if persist_dir.exists():
    print("\n=== Directory contents (sorted by modification time) ===")
    items = []
    for item in persist_dir.iterdir():
        stat = item.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        items.append((item.name, mtime, stat.st_size if item.is_file() else "DIR"))
    
    items.sort(key=lambda x: x[1], reverse=True)
    for name, mtime, size in items[:20]:
        print(f"  {mtime.strftime('%Y-%m-%d %H:%M:%S')} {size:>10} {name}")
    
    # Look in subdirectories
    for subdir in persist_dir.iterdir():
        if subdir.is_dir():
            print(f"\n=== {subdir.name}/ contents ===")
            subitems = []
            for item in subdir.iterdir():
                stat = item.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                subitems.append((item.name, mtime, stat.st_size if item.is_file() else "DIR"))
            subitems.sort(key=lambda x: x[1], reverse=True)
            for name, mtime, size in subitems[:10]:
                print(f"  {mtime.strftime('%Y-%m-%d %H:%M:%S')} {size:>10} {name}")
