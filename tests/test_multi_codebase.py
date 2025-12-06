"""
Tests for Multi-Codebase Support

This file tests the multi-codebase indexing functionality including:
- ServerConfig and CodebaseConfig classes
- Multi-codebase MemoryService initialization
- Per-codebase and cross-codebase search
- Config file loading/saving
"""

import unittest
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path

from conductor_memory.config.server import ServerConfig, CodebaseConfig, generate_example_config
from conductor_memory.storage.chroma import ChromaVectorStore
from conductor_memory.service.memory_service import MemoryService


class TestCodebaseConfig(unittest.TestCase):
    """Test CodebaseConfig class"""
    
    def test_create_with_defaults(self):
        """Should create config with default values"""
        config = CodebaseConfig(name="test", path="/path/to/code")
        
        self.assertEqual(config.name, "test")
        self.assertTrue(config.enabled)
        self.assertIn('.py', config.extensions)
        self.assertIn('.js', config.extensions)
        self.assertIn('__pycache__', config.ignore_patterns)
        self.assertIn('.git', config.ignore_patterns)
    
    def test_extension_normalization(self):
        """Extensions should be normalized to start with dot"""
        config = CodebaseConfig(
            name="test",
            path="/path",
            extensions=['py', '.js', 'ts']
        )
        
        self.assertEqual(config.extensions, ['.py', '.js', '.ts'])
    
    def test_should_ignore(self):
        """Should correctly identify paths to ignore"""
        config = CodebaseConfig(
            name="test",
            path="/path",
            ignore_patterns=['__pycache__', '.git', '*.pyc']
        )
        
        self.assertTrue(config.should_ignore('/path/__pycache__/module.py'))
        self.assertTrue(config.should_ignore('/path/.git/config'))
        self.assertTrue(config.should_ignore('/path/module.pyc'))
        self.assertFalse(config.should_ignore('/path/src/module.py'))
    
    def test_to_dict_and_from_dict(self):
        """Should serialize and deserialize correctly"""
        original = CodebaseConfig(
            name="test",
            path="/path/to/code",
            extensions=['.py', '.js'],
            ignore_patterns=['__pycache__'],
            enabled=True,
            description="Test codebase"
        )
        
        data = original.to_dict()
        restored = CodebaseConfig.from_dict(data)
        
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.extensions, original.extensions)
        self.assertEqual(restored.ignore_patterns, original.ignore_patterns)
        self.assertEqual(restored.enabled, original.enabled)
        self.assertEqual(restored.description, original.description)


class TestServerConfig(unittest.TestCase):
    """Test ServerConfig class"""
    
    def test_create_default(self):
        """Should create default config"""
        config = ServerConfig.create_default()
        
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8000)
        self.assertEqual(config.codebases, [])
    
    def test_create_default_with_codebase(self):
        """Should create default config with a single codebase"""
        config = ServerConfig.create_default(
            codebase_path="/path/to/code",
            codebase_name="myproject"
        )
        
        self.assertEqual(len(config.codebases), 1)
        self.assertEqual(config.codebases[0].name, "myproject")
        self.assertEqual(config.codebases[0].path, str(Path("/path/to/code").absolute()))
    
    def test_get_enabled_codebases(self):
        """Should filter enabled codebases"""
        config = ServerConfig(
            codebases=[
                CodebaseConfig(name="enabled1", path="/path1", enabled=True),
                CodebaseConfig(name="disabled", path="/path2", enabled=False),
                CodebaseConfig(name="enabled2", path="/path3", enabled=True),
            ]
        )
        
        enabled = config.get_enabled_codebases()
        self.assertEqual(len(enabled), 2)
        self.assertEqual(enabled[0].name, "enabled1")
        self.assertEqual(enabled[1].name, "enabled2")
    
    def test_get_codebase_by_name(self):
        """Should find codebase by name"""
        config = ServerConfig(
            codebases=[
                CodebaseConfig(name="project1", path="/path1"),
                CodebaseConfig(name="project2", path="/path2"),
            ]
        )
        
        codebase = config.get_codebase_by_name("project2")
        self.assertIsNotNone(codebase)
        self.assertEqual(codebase.name, "project2")
        
        missing = config.get_codebase_by_name("nonexistent")
        self.assertIsNone(missing)
    
    def test_save_and_load_json(self):
        """Should save and load JSON config file"""
        config = ServerConfig(
            host="localhost",
            port=9000,
            codebases=[
                CodebaseConfig(name="test", path="/path/to/test")
            ]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            config.save_to_file(config_path)
            loaded = ServerConfig.from_file(config_path)
            
            self.assertEqual(loaded.host, config.host)
            self.assertEqual(loaded.port, config.port)
            self.assertEqual(len(loaded.codebases), 1)
            self.assertEqual(loaded.codebases[0].name, "test")
        finally:
            Path(config_path).unlink()
    
    def test_generate_example_config(self):
        """Should generate example config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            generate_example_config(config_path)
            
            # Should be valid JSON
            with open(config_path) as f:
                data = json.load(f)
            
            self.assertIn('host', data)
            self.assertIn('port', data)
            self.assertIn('codebases', data)
            self.assertEqual(len(data['codebases']), 2)
        finally:
            Path(config_path).unlink()


class TestMultiCodebaseService(unittest.TestCase):
    """Test MemoryService with multiple codebases"""
    
    def setUp(self):
        """Create temporary directories for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.persist_dir = tempfile.mkdtemp()
        
        # Create two separate codebase directories
        self.codebase1_dir = Path(self.temp_dir) / "project1"
        self.codebase2_dir = Path(self.temp_dir) / "project2"
        self.codebase1_dir.mkdir()
        self.codebase2_dir.mkdir()
        
        # Create files in each codebase
        self._create_file(self.codebase1_dir, "module1.py", 
            "def project1_function():\n    '''Function in project 1'''\n    return 'project1'\n")
        self._create_file(self.codebase1_dir, "utils.py",
            "def project1_util():\n    '''Utility function in project 1'''\n    pass\n")
        
        self._create_file(self.codebase2_dir, "module2.py",
            "def project2_function():\n    '''Function in project 2'''\n    return 'project2'\n")
        self._create_file(self.codebase2_dir, "helpers.py",
            "def project2_helper():\n    '''Helper function in project 2'''\n    pass\n")
    
    def tearDown(self):
        """Clean up temporary directories"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        try:
            shutil.rmtree(self.persist_dir)
        except Exception:
            pass
    
    def _create_file(self, base_dir: Path, filename: str, content: str):
        """Helper to create a file"""
        filepath = base_dir / filename
        filepath.write_text(content)
        return filepath
    
    def _create_config(self) -> ServerConfig:
        """Helper to create test config"""
        config = ServerConfig(
            persist_directory=self.persist_dir,
            enable_file_watcher=False,
            codebases=[
                CodebaseConfig(name="project1", path=str(self.codebase1_dir)),
                CodebaseConfig(name="project2", path=str(self.codebase2_dir)),
            ]
        )
        return config
    
    def test_multi_codebase_initialization(self):
        """Should initialize with multiple codebases"""
        service = MemoryService(self._create_config())
        
        self.assertIsNotNone(service.get_vector_store("project1"))
        self.assertIsNotNone(service.get_vector_store("project2"))
    
    def test_multi_codebase_indexing(self):
        """Should index multiple codebases independently"""
        service = MemoryService(self._create_config())
        service.initialize()  # Blocks until complete
        
        # Check both codebases are indexed
        project1_files = service.get_vector_store("project1").get_indexed_files()
        project2_files = service.get_vector_store("project2").get_indexed_files()
        
        self.assertIn("module1.py", project1_files)
        self.assertIn("utils.py", project1_files)
        self.assertIn("module2.py", project2_files)
        self.assertIn("helpers.py", project2_files)
        
        # Files should not be mixed between codebases
        self.assertNotIn("module2.py", project1_files)
        self.assertNotIn("module1.py", project2_files)
    
    def test_per_codebase_search(self):
        """Should search within a specific codebase"""
        service = MemoryService(self._create_config())
        service.initialize()
        
        # Search in project1 only
        result = service.search(
            query="function",
            codebase="project1",
            max_results=10
        )
        
        # All results should be from project1
        for r in result["results"]:
            self.assertEqual(r["project_id"], "project1")
    
    def test_cross_codebase_search(self):
        """Should search across all codebases when no codebase specified"""
        service = MemoryService(self._create_config())
        service.initialize()
        
        # Search across all codebases
        result = service.search(
            query="function",
            codebase=None,  # Search all
            max_results=20
        )
        
        # Should have results from both codebases
        project_ids = set(r["project_id"] for r in result["results"])
        self.assertIn("project1", project_ids)
        self.assertIn("project2", project_ids)
    
    def test_disabled_codebase_not_indexed(self):
        """Disabled codebases should not be indexed"""
        config = ServerConfig(
            persist_directory=self.persist_dir,
            enable_file_watcher=False,
            codebases=[
                CodebaseConfig(name="project1", path=str(self.codebase1_dir), enabled=True),
                CodebaseConfig(name="project2", path=str(self.codebase2_dir), enabled=False),
            ]
        )
        
        service = MemoryService(config)
        
        # Only enabled codebases should have vector stores
        self.assertIsNotNone(service.get_vector_store("project1"))
        self.assertIsNone(service.get_vector_store("project2"))
    
    def test_backward_compatibility_single_codebase(self):
        """Should work with single codebase path (backward compatibility)"""
        config = ServerConfig.create_default(
            codebase_path=str(self.codebase1_dir),
            codebase_name="default"
        )
        config.persist_directory = self.persist_dir
        config.enable_file_watcher = False
        
        service = MemoryService(config)
        
        # Should have created a default codebase config
        self.assertEqual(len(service.config.codebases), 1)
        self.assertEqual(service.config.codebases[0].name, "default")
        self.assertIsNotNone(service.get_vector_store("default"))


class TestConfigFilePersistence(unittest.TestCase):
    """Test config file persistence across service restarts"""
    
    def setUp(self):
        """Create temporary directories"""
        self.temp_dir = tempfile.mkdtemp()
        self.persist_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "config.json"
        
        # Create a codebase directory
        self.codebase_dir = Path(self.temp_dir) / "codebase"
        self.codebase_dir.mkdir()
        (self.codebase_dir / "test.py").write_text(
            "def test_function():\n    '''Test function for persistence test'''\n    pass\n"
        )
    
    def tearDown(self):
        """Clean up"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        try:
            shutil.rmtree(self.persist_dir)
        except Exception:
            pass
    
    def test_index_persists_across_restarts(self):
        """Index should persist and be loaded on restart"""
        config = ServerConfig(
            persist_directory=self.persist_dir,
            enable_file_watcher=False,
            codebases=[
                CodebaseConfig(name="test", path=str(self.codebase_dir))
            ]
        )
        
        # First service instance - index the codebase
        service1 = MemoryService(config)
        service1.initialize()
        
        files1 = service1.get_vector_store("test").get_indexed_files()
        self.assertIn("test.py", files1)
        
        # Second service instance - should load existing index
        service2 = MemoryService(config)
        
        files2 = service2.get_vector_store("test").get_indexed_files()
        self.assertIn("test.py", files2)
        self.assertEqual(files1["test.py"]["content_hash"], files2["test.py"]["content_hash"])


if __name__ == "__main__":
    unittest.main()
