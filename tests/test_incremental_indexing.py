"""
Tests for Incremental Indexing Features

This file tests the incremental codebase indexing functionality including:
- FileIndexMetadata CRUD operations
- ChromaVectorStore file tracking methods
- Content hash computation
- Incremental indexing logic (new/modified/deleted files)
- MemoryService indexing behavior
"""

import unittest
import tempfile
import shutil
import os
import time
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules under test
from conductor_memory.storage.chroma import ChromaVectorStore, FileIndexMetadata
from conductor_memory.core.models import MemoryChunk, RoleEnum
from conductor_memory.config.server import ServerConfig
from conductor_memory.service.memory_service import MemoryService


class TestContentHash(unittest.TestCase):
    """Test content hash computation"""
    
    def test_compute_content_hash_deterministic(self):
        """Hash should be deterministic for same content"""
        content = "def hello():\n    print('Hello, World!')"
        hash1 = ChromaVectorStore.compute_content_hash(content)
        hash2 = ChromaVectorStore.compute_content_hash(content)
        self.assertEqual(hash1, hash2)
    
    def test_compute_content_hash_different_content(self):
        """Different content should produce different hashes"""
        content1 = "def hello(): pass"
        content2 = "def goodbye(): pass"
        hash1 = ChromaVectorStore.compute_content_hash(content1)
        hash2 = ChromaVectorStore.compute_content_hash(content2)
        self.assertNotEqual(hash1, hash2)
    
    def test_compute_content_hash_empty_string(self):
        """Empty string should produce a valid hash"""
        hash_val = ChromaVectorStore.compute_content_hash("")
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)  # SHA-256 produces 64 hex chars
    
    def test_compute_content_hash_unicode(self):
        """Unicode content should be handled correctly"""
        content = "def greet(): print('Hëllö Wörld! 你好')"
        hash_val = ChromaVectorStore.compute_content_hash(content)
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)
    
    def test_compute_content_hash_whitespace_sensitive(self):
        """Hash should be sensitive to whitespace changes"""
        content1 = "def hello(): pass"
        content2 = "def hello():  pass"  # Extra space
        hash1 = ChromaVectorStore.compute_content_hash(content1)
        hash2 = ChromaVectorStore.compute_content_hash(content2)
        self.assertNotEqual(hash1, hash2)


class TestFileIndexMetadata(unittest.TestCase):
    """Test FileIndexMetadata CRUD operations"""
    
    def setUp(self):
        """Create a temporary directory for Chroma storage"""
        self.temp_dir = tempfile.mkdtemp()
        import chromadb
        self.client = chromadb.PersistentClient(path=self.temp_dir)
        self.file_index = FileIndexMetadata(self.client, "test_file_index")
    
    def tearDown(self):
        """Clean up temporary directory"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def test_get_file_info_nonexistent(self):
        """Getting info for non-existent file should return None"""
        result = self.file_index.get_file_info("nonexistent/file.py")
        self.assertIsNone(result)
    
    def test_update_and_get_file_info(self):
        """Should be able to store and retrieve file info"""
        file_path = "src/test.py"
        mtime = 1234567890.0
        content_hash = "abc123def456"
        chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
        
        self.file_index.update_file_info(file_path, mtime, content_hash, chunk_ids)
        
        result = self.file_index.get_file_info(file_path)
        self.assertIsNotNone(result)
        self.assertEqual(result["mtime"], mtime)
        self.assertEqual(result["content_hash"], content_hash)
        # chunk_ids are stored as comma-separated string
        self.assertEqual(result["chunk_ids"], ",".join(chunk_ids))
    
    def test_update_file_info_overwrites(self):
        """Updating file info should overwrite previous entry"""
        file_path = "test.py"
        
        # First update
        self.file_index.update_file_info(file_path, 1000.0, "hash1", ["chunk-1"])
        
        # Second update
        self.file_index.update_file_info(file_path, 2000.0, "hash2", ["chunk-2", "chunk-3"])
        
        result = self.file_index.get_file_info(file_path)
        self.assertEqual(result["mtime"], 2000.0)
        self.assertEqual(result["content_hash"], "hash2")
        # chunk_ids are stored as comma-separated string
        self.assertEqual(result["chunk_ids"], "chunk-2,chunk-3")
    
    def test_remove_file_info(self):
        """Should be able to remove file info"""
        file_path = "test.py"
        self.file_index.update_file_info(file_path, 1000.0, "hash", ["chunk-1"])
        
        # Verify it exists
        self.assertIsNotNone(self.file_index.get_file_info(file_path))
        
        # Remove it
        self.file_index.delete_file_info(file_path)
        
        # Verify it's gone
        self.assertIsNone(self.file_index.get_file_info(file_path))
    
    def test_get_all_indexed_files_empty(self):
        """Should return empty dict when no files indexed"""
        result = self.file_index.get_all_indexed_files()
        self.assertEqual(result, {})
    
    def test_get_all_indexed_files_multiple(self):
        """Should return all indexed files"""
        self.file_index.update_file_info("file1.py", 1000.0, "h1", ["c1"])
        self.file_index.update_file_info("file2.py", 2000.0, "h2", ["c2"])
        self.file_index.update_file_info("dir/file3.py", 3000.0, "h3", ["c3"])
        
        result = self.file_index.get_all_indexed_files()
        self.assertEqual(len(result), 3)
        self.assertIn("file1.py", result)
        self.assertIn("file2.py", result)
        self.assertIn("dir/file3.py", result)
    
    def test_indexed_at_timestamp(self):
        """Should record indexed_at timestamp"""
        file_path = "test.py"
        before = datetime.now().isoformat()
        
        self.file_index.update_file_info(file_path, 1000.0, "hash", ["chunk-1"])
        
        after = datetime.now().isoformat()
        result = self.file_index.get_file_info(file_path)
        
        # indexed_at should be between before and after
        self.assertIsNotNone(result.get("indexed_at"))
        self.assertGreaterEqual(result["indexed_at"], before)
        self.assertLessEqual(result["indexed_at"], after)


class TestChromaVectorStoreFileTracking(unittest.TestCase):
    """Test ChromaVectorStore file tracking methods"""
    
    def setUp(self):
        """Create a temporary directory for Chroma storage"""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up temporary directory"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def test_get_indexed_files_initially_empty(self):
        """Should return empty dict initially"""
        result = self.vector_store.get_indexed_files()
        self.assertEqual(result, {})
    
    def test_needs_reindex_new_file(self):
        """New file should need indexing"""
        result = self.vector_store.needs_reindex("new_file.py", 1000.0, "hash123")
        self.assertTrue(result)
    
    def test_needs_reindex_unchanged_file(self):
        """Unchanged file should not need re-indexing"""
        file_path = "test.py"
        mtime = 1000.0
        content_hash = "hash123"
        
        # Index the file
        self.vector_store.update_file_index(file_path, mtime, content_hash, ["chunk-1"])
        
        # Check if needs re-indexing with same mtime and hash
        result = self.vector_store.needs_reindex(file_path, mtime, content_hash)
        self.assertFalse(result)
    
    def test_needs_reindex_modified_file(self):
        """Modified file should need re-indexing"""
        file_path = "test.py"
        
        # Index the file
        self.vector_store.update_file_index(file_path, 1000.0, "hash1", ["chunk-1"])
        
        # Check with different mtime and hash
        result = self.vector_store.needs_reindex(file_path, 2000.0, "hash2")
        self.assertTrue(result)
    
    def test_needs_reindex_touched_but_unchanged(self):
        """File with different mtime but same hash should not need re-indexing"""
        file_path = "test.py"
        content_hash = "hash123"
        
        # Index the file
        self.vector_store.update_file_index(file_path, 1000.0, content_hash, ["chunk-1"])
        
        # Check with different mtime but same hash
        result = self.vector_store.needs_reindex(file_path, 2000.0, content_hash)
        self.assertFalse(result)
    
    def test_update_file_index(self):
        """Should store file index metadata"""
        file_path = "src/module.py"
        mtime = 1234567890.0
        content_hash = "abc123"
        chunk_ids = ["chunk-1", "chunk-2"]
        
        self.vector_store.update_file_index(file_path, mtime, content_hash, chunk_ids)
        
        indexed_files = self.vector_store.get_indexed_files()
        self.assertIn(file_path, indexed_files)
        self.assertEqual(indexed_files[file_path]["mtime"], mtime)
        self.assertEqual(indexed_files[file_path]["content_hash"], content_hash)
    
    def test_remove_file_chunks(self):
        """Should remove all chunks for a file"""
        file_path = "test.py"
        
        # Create and add some chunks
        chunk1 = MemoryChunk(
            id="chunk-1",
            project_id="test",
            role=RoleEnum.SYSTEM,
            prompt="",
            response="",
            doc_text="def hello(): pass",
            embedding_id="",
            tags=[f"file:{file_path}"],
            pin=False,
            relevance_score=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="codebase_indexing"
        )
        chunk2 = MemoryChunk(
            id="chunk-2",
            project_id="test",
            role=RoleEnum.SYSTEM,
            prompt="",
            response="",
            doc_text="def world(): pass",
            embedding_id="",
            tags=[f"file:{file_path}"],
            pin=False,
            relevance_score=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="codebase_indexing"
        )
        
        # Add chunks with dummy embeddings
        dummy_embedding = [0.1] * 384
        self.vector_store.add(chunk1, dummy_embedding)
        self.vector_store.add(chunk2, dummy_embedding)
        
        # Update file index
        self.vector_store.update_file_index(file_path, 1000.0, "hash", ["chunk-1", "chunk-2"])
        
        # Remove chunks
        removed_count = self.vector_store.remove_file_chunks(file_path)
        self.assertEqual(removed_count, 2)
        
        # File should no longer be in index
        indexed_files = self.vector_store.get_indexed_files()
        self.assertNotIn(file_path, indexed_files)
    
    def test_get_indexed_files_empty(self):
        """Should return empty dict when no files indexed"""
        result = self.vector_store.get_indexed_files()
        self.assertEqual(result, {})
    
    def test_get_indexed_files_multiple(self):
        """Should return all indexed files"""
        self.vector_store.update_file_index("file1.py", 1000.0, "h1", ["c1"])
        self.vector_store.update_file_index("file2.py", 2000.0, "h2", ["c2"])
        
        result = self.vector_store.get_indexed_files()
        self.assertEqual(len(result), 2)
        self.assertIn("file1.py", result)
        self.assertIn("file2.py", result)


class TestMemoryServiceIndexing(unittest.TestCase):
    """Test MemoryService incremental indexing behavior"""
    
    def setUp(self):
        """Create temporary directories for testing"""
        self.temp_codebase = tempfile.mkdtemp()
        self.temp_persist = tempfile.mkdtemp()
        
        # Create some test files with enough content to pass the 50 char minimum
        self._create_file("module1.py", "def hello():\n    '''A greeting function that prints hello'''\n    print('Hello')\n")
        self._create_file("module2.py", "def world():\n    '''A function that prints world to the console'''\n    print('World')\n")
        self._create_file("subdir/nested.py", "class Nested:\n    '''A nested class for testing purposes'''\n    pass\n")
    
    def tearDown(self):
        """Clean up temporary directories"""
        try:
            shutil.rmtree(self.temp_codebase)
        except Exception:
            pass
        try:
            shutil.rmtree(self.temp_persist)
        except Exception:
            pass
    
    def _create_file(self, relative_path: str, content: str):
        """Helper to create a file in the temp codebase"""
        full_path = Path(self.temp_codebase) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path
    
    def _modify_file(self, relative_path: str, content: str):
        """Helper to modify a file in the temp codebase"""
        full_path = Path(self.temp_codebase) / relative_path
        full_path.write_text(content)
        return full_path
    
    def _delete_file(self, relative_path: str):
        """Helper to delete a file from the temp codebase"""
        full_path = Path(self.temp_codebase) / relative_path
        if full_path.exists():
            full_path.unlink()
    
    def _create_service(self) -> MemoryService:
        """Helper to create a MemoryService with test config"""
        config = ServerConfig.create_default(
            codebase_path=self.temp_codebase,
            codebase_name="test"
        )
        config.persist_directory = self.temp_persist
        config.enable_file_watcher = False  # Disable for tests
        return MemoryService(config)
    
    def test_service_initialization(self):
        """Service should initialize with ChromaVectorStore"""
        service = self._create_service()
        
        vector_store = service.get_vector_store("test")
        self.assertIsNotNone(vector_store)
        self.assertEqual(vector_store.__class__.__name__, "ChromaVectorStore")
    
    def test_full_codebase_indexing(self):
        """Should index all code files on first run"""
        service = self._create_service()
        
        # Run indexing (sync version blocks until complete)
        service.initialize()
        
        # Check all files are indexed
        vector_store = service.get_vector_store("test")
        indexed_files = vector_store.get_indexed_files()
        self.assertIn("module1.py", indexed_files)
        self.assertIn("module2.py", indexed_files)
        # Note: subdir/nested.py path format may vary by OS
        nested_found = any("nested.py" in path for path in indexed_files.keys())
        self.assertTrue(nested_found, f"nested.py not found in {indexed_files.keys()}")
    
    def test_incremental_indexing_new_file(self):
        """Should only index new files on subsequent runs"""
        # First indexing
        service1 = self._create_service()
        service1.initialize()
        
        initial_count = len(service1.get_vector_store("test").get_indexed_files())
        
        # Add a new file with enough content
        self._create_file("new_module.py", "def new_function():\n    '''A new function for testing incremental indexing'''\n    pass\n")
        
        # Second indexing (simulating restart) - use same persist dir
        service2 = self._create_service()
        service2.initialize()
        
        # Should have one more file
        final_count = len(service2.get_vector_store("test").get_indexed_files())
        self.assertEqual(final_count, initial_count + 1)
        
        indexed_files = service2.get_vector_store("test").get_indexed_files()
        self.assertIn("new_module.py", indexed_files)
    
    def test_incremental_indexing_modified_file(self):
        """Should re-index modified files"""
        # First indexing
        service1 = self._create_service()
        service1.initialize()
        
        original_hash = service1.get_vector_store("test").get_indexed_files()["module1.py"]["content_hash"]
        
        # Modify a file (need small delay to ensure different mtime)
        time.sleep(0.1)
        self._modify_file("module1.py", "def hello():\n    '''A modified greeting function'''\n    print('Hello Modified')\n")
        
        # Second indexing - use same persist dir
        service2 = self._create_service()
        service2.initialize()
        
        # Hash should be different
        new_hash = service2.get_vector_store("test").get_indexed_files()["module1.py"]["content_hash"]
        self.assertNotEqual(original_hash, new_hash)
    
    def test_incremental_indexing_deleted_file(self):
        """Should remove index for deleted files"""
        # First indexing
        service1 = self._create_service()
        service1.initialize()
        
        # Verify file is indexed
        self.assertIn("module2.py", service1.get_vector_store("test").get_indexed_files())
        
        # Delete a file
        self._delete_file("module2.py")
        
        # Second indexing - use same persist dir
        service2 = self._create_service()
        service2.initialize()
        
        # File should no longer be indexed
        self.assertNotIn("module2.py", service2.get_vector_store("test").get_indexed_files())
    
    def test_no_reindex_unchanged_files(self):
        """Should not re-index unchanged files"""
        # First indexing
        service1 = self._create_service()
        service1.initialize()
        
        # Get the indexed_at timestamp for a file
        first_indexed = service1.get_vector_store("test").get_indexed_files()["module1.py"]["indexed_at"]
        
        # Wait a moment
        time.sleep(0.1)
        
        # Second indexing without changes - use same persist dir
        service2 = self._create_service()
        service2.initialize()
        
        # indexed_at should be the same (file wasn't re-indexed)
        second_indexed = service2.get_vector_store("test").get_indexed_files()["module1.py"]["indexed_at"]
        self.assertEqual(first_indexed, second_indexed)


class TestIgnorePatterns(unittest.TestCase):
    """Test that ignore patterns are respected during indexing"""
    
    def setUp(self):
        """Create temporary directories with files that should be ignored"""
        self.temp_codebase = tempfile.mkdtemp()
        self.temp_persist = tempfile.mkdtemp()
        
        # Create files that should be indexed (with enough content)
        self._create_file("main.py", "def main():\n    '''Main entry point for the application'''\n    pass\n")
        
        # Create files that should be ignored (also with enough content to be sure they're skipped for the right reason)
        self._create_file("__pycache__/cached.py", "def cached():\n    '''This should not be indexed'''\n    pass\n")
        self._create_file(".git/hooks/pre-commit.py", "def hook():\n    '''This should not be indexed'''\n    pass\n")
        self._create_file("node_modules/package/index.py", "def node():\n    '''This should not be indexed'''\n    pass\n")
        self._create_file("venv/lib/site.py", "def venv():\n    '''This should not be indexed'''\n    pass\n")
        self._create_file(".idea/workspace.py", "def idea():\n    '''This should not be indexed'''\n    pass\n")
    
    def tearDown(self):
        """Clean up temporary directories"""
        try:
            shutil.rmtree(self.temp_codebase)
        except Exception:
            pass
        try:
            shutil.rmtree(self.temp_persist)
        except Exception:
            pass
    
    def _create_file(self, relative_path: str, content: str):
        """Helper to create a file in the temp codebase"""
        full_path = Path(self.temp_codebase) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path
    
    def _create_service(self) -> MemoryService:
        """Helper to create a MemoryService with test config"""
        config = ServerConfig.create_default(
            codebase_path=self.temp_codebase,
            codebase_name="test"
        )
        config.persist_directory = self.temp_persist
        config.enable_file_watcher = False
        return MemoryService(config)
    
    def test_ignore_patterns_respected(self):
        """Files matching ignore patterns should not be indexed"""
        service = self._create_service()
        service.initialize()
        
        indexed_files = service.get_vector_store("test").get_indexed_files()
        
        # main.py should be indexed
        self.assertIn("main.py", indexed_files)
        
        # Ignored directories should not have any files indexed
        for path in indexed_files.keys():
            self.assertNotIn("__pycache__", path)
            self.assertNotIn(".git", path)
            self.assertNotIn("node_modules", path)
            self.assertNotIn("venv", path)
            self.assertNotIn(".idea", path)


if __name__ == "__main__":
    unittest.main()
