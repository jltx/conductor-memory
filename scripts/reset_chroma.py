#!/usr/bin/env python3
"""
Reset ChromaDB data to apply new HNSW settings.

This script:
1. Finds and deletes all ChromaDB data directories
2. The next server start will recreate collections with optimized settings

Usage:
    python scripts/reset_chroma.py [--dry-run]
    
Options:
    --dry-run    Show what would be deleted without actually deleting
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def find_chroma_dirs(base_path: Path) -> list[Path]:
    """Find all potential ChromaDB data directories."""
    chroma_dirs = []
    
    # Common locations
    candidates = [
        base_path / "data" / "chroma",
        base_path / "data",
        base_path / ".chroma",
        base_path / "chroma_data",
        Path.home() / ".conductor-memory",  # Default location - check root
        Path.home() / ".conductor-memory" / "chroma",
        Path.home() / ".conductor-memory" / "data",
        Path(os.environ.get("APPDATA", "")) / "conductor-memory" if os.name == "nt" else None,
        Path(os.environ.get("APPDATA", "")) / "conductor-memory" / "chroma" if os.name == "nt" else None,
        Path(os.environ.get("LOCALAPPDATA", "")) / "conductor-memory" if os.name == "nt" else None,
        Path(os.environ.get("LOCALAPPDATA", "")) / "conductor-memory" / "chroma" if os.name == "nt" else None,
    ]
    
    for candidate in candidates:
        if candidate and candidate.exists():
            # Check if it looks like a ChromaDB directory (has chroma.sqlite3 anywhere)
            if (candidate / "chroma.sqlite3").exists() or any(candidate.glob("*/chroma.sqlite3")) or any(candidate.glob("chroma.sqlite3")):
                chroma_dirs.append(candidate)
            elif candidate.name == "chroma" and candidate.is_dir():
                chroma_dirs.append(candidate)
    
    return chroma_dirs


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Reset ChromaDB data for conductor-memory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--data-dir", type=str, help="Explicit path to ChromaDB data directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ChromaDB Reset Script")
    print("=" * 60)
    print()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Find ChromaDB directories
    if args.data_dir:
        chroma_dirs = [Path(args.data_dir)] if Path(args.data_dir).exists() else []
    else:
        chroma_dirs = find_chroma_dirs(project_root)
    
    if not chroma_dirs:
        print("No ChromaDB data directories found.")
        print()
        print("Searched in:")
        print(f"  - {project_root / 'data' / 'chroma'}")
        print(f"  - {Path.home() / '.conductor-memory'}")
        if os.name == "nt":
            print(f"  - {Path(os.environ.get('APPDATA', '')) / 'conductor-memory'}")
        print()
        print("If your data is elsewhere, use: --data-dir <path>")
        return 0
    
    # Show what we found
    print("Found ChromaDB data directories:")
    print()
    total_size = 0
    for d in chroma_dirs:
        size = get_dir_size(d)
        total_size += size
        print(f"  {d}")
        print(f"    Size: {format_size(size)}")
        
        # Count collections if possible
        sqlite_files = list(d.rglob("chroma.sqlite3"))
        if sqlite_files:
            print(f"    SQLite files: {len(sqlite_files)}")
    
    print()
    print(f"Total size: {format_size(total_size)}")
    print()
    
    if args.dry_run:
        print("[DRY RUN] Would delete the above directories.")
        print("Run without --dry-run to actually delete.")
        return 0
    
    # Confirm deletion
    print("WARNING: This will delete all indexed data!")
    print("You will need to restart the server to reindex your codebases.")
    print()
    response = input("Are you sure you want to proceed? [y/N]: ").strip().lower()
    
    if response != "y":
        print("Aborted.")
        return 1
    
    # Delete directories (preserving config files)
    print()
    for d in chroma_dirs:
        print(f"Deleting ChromaDB data in {d}...")
        try:
            # Check if this directory contains config files we should preserve
            config_files = list(d.glob("config*.json"))
            
            if config_files:
                # Preserve config files - delete everything else
                print(f"  Preserving {len(config_files)} config file(s)...")
                for item in d.iterdir():
                    if item.name.startswith("config") and item.suffix == ".json":
                        continue  # Skip config files
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                print(f"  Deleted ChromaDB data (preserved config files).")
            else:
                # No config files - delete the whole directory
                shutil.rmtree(d)
                print(f"  Deleted successfully.")
        except Exception as e:
            print(f"  Error: {e}")
            return 1
    
    print()
    print("=" * 60)
    print("Done! ChromaDB data has been reset.")
    print()
    print("Next steps:")
    print("  1. Restart the conductor-memory server")
    print("  2. The server will automatically reindex your codebases")
    print("  3. New collections will use optimized HNSW settings")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
