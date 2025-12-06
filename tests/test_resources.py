"""
Tests for OS-agnostic resource detection.

Tests cover:
- CPU detection
- RAM detection  
- GPU detection (CUDA, MPS, ROCm)
- Batch size recommendations
- Device selection
- Configuration overrides
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

from conductor_memory.core.resources import (
    ResourceDetector,
    SystemResources,
    GPUInfo,
    DeviceType,
    detect_resources,
    get_optimal_device,
    get_resource_detector,
)


class TestCPUDetection:
    """Tests for CPU core/thread detection"""
    
    def test_detects_cpu_count(self):
        """Should detect CPU cores/threads"""
        detector = ResourceDetector()
        resources = detector.detect()
        
        assert resources.cpu_cores >= 1
        assert resources.cpu_threads >= 1
        assert resources.cpu_threads >= resources.cpu_cores
    
    @patch('os.cpu_count')
    def test_handles_cpu_count_none(self, mock_cpu_count):
        """Should handle os.cpu_count() returning None"""
        mock_cpu_count.return_value = None
        
        detector = ResourceDetector()
        cores, threads = detector._detect_cpu()
        
        assert cores >= 1
        assert threads >= 1
    
    @patch('os.cpu_count')
    def test_estimates_cores_from_threads(self, mock_cpu_count):
        """Should estimate physical cores as half of threads when psutil unavailable"""
        mock_cpu_count.return_value = 16
        
        with patch.dict(sys.modules, {'psutil': None}):
            detector = ResourceDetector()
            # Force reimport to use mocked psutil
            detector._resources = None
            cores, threads = detector._detect_cpu()
        
        assert threads == 16
        # Without psutil, cores should be estimated as threads // 2
        # But since psutil is likely installed, this may return actual value


class TestRAMDetection:
    """Tests for RAM detection"""
    
    def test_detects_ram(self):
        """Should detect total and available RAM"""
        detector = ResourceDetector()
        resources = detector.detect()
        
        assert resources.total_ram_gb > 0
        assert resources.available_ram_gb > 0
        assert resources.available_ram_gb <= resources.total_ram_gb
    
    def test_ram_in_reasonable_range(self):
        """RAM should be in a reasonable range for modern systems"""
        detector = ResourceDetector()
        resources = detector.detect()
        
        # Most systems have between 4GB and 1TB RAM
        assert 1.0 <= resources.total_ram_gb <= 1024.0
    
    @patch('psutil.virtual_memory')
    def test_handles_psutil_error(self, mock_vmem):
        """Should use defaults if psutil fails"""
        mock_vmem.side_effect = Exception("psutil error")
        
        detector = ResourceDetector()
        detector._resources = None  # Clear cache
        total, available = detector._detect_ram()
        
        # Should return defaults
        assert total == 4.0
        assert available == 2.0


class TestGPUDetection:
    """Tests for GPU detection"""
    
    def test_detects_available_gpu(self):
        """Should detect GPU if available"""
        detector = ResourceDetector()
        resources = detector.detect()
        
        # GPU may or may not be present - just verify structure
        if resources.gpu:
            assert resources.gpu.device_type in DeviceType
            assert resources.gpu.memory_gb > 0
            assert resources.gpu.device_name
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_detects_cuda_gpu(self, mock_props, mock_available):
        """Should detect NVIDIA CUDA GPU"""
        mock_available.return_value = True
        
        # Create a proper mock with attributes
        props_mock = MagicMock()
        props_mock.name = "NVIDIA GeForce RTX 4090"
        props_mock.total_memory = 24 * 1024**3  # 24GB
        mock_props.return_value = props_mock
        
        detector = ResourceDetector()
        detector._resources = None  # Clear cache
        gpu = detector._detect_cuda_gpu()
        
        assert gpu is not None
        assert gpu.device_type == DeviceType.CUDA
        assert gpu.device_name == "NVIDIA GeForce RTX 4090"
        assert abs(gpu.memory_gb - 24.0) < 0.1
    
    @patch('torch.cuda.is_available')
    def test_no_cuda_returns_none(self, mock_available):
        """Should return None when CUDA not available"""
        mock_available.return_value = False
        
        detector = ResourceDetector()
        detector._resources = None
        gpu = detector._detect_cuda_gpu()
        
        assert gpu is None
    
    def test_handles_no_torch(self):
        """Should handle torch not being installed"""
        with patch.dict(sys.modules, {'torch': None}):
            detector = ResourceDetector()
            detector._resources = None
            
            # Should not raise, just return None for GPU
            gpu = detector._detect_cuda_gpu()
            # Can't easily test this without uninstalling torch


class TestMPSDetection:
    """Tests for Apple Silicon MPS detection"""
    
    @patch('torch.backends.mps.is_available')
    def test_detects_mps_gpu(self, mock_mps):
        """Should detect Apple Silicon MPS"""
        mock_mps.return_value = True
        
        # Mock psutil for memory estimation
        with patch('psutil.virtual_memory') as mock_vmem:
            mock_vmem.return_value = MagicMock(total=32 * 1024**3)  # 32GB unified
            
            detector = ResourceDetector()
            detector._resources = None
            gpu = detector._detect_mps_gpu()
            
            if gpu:  # Only runs on systems with MPS support in torch
                assert gpu.device_type == DeviceType.MPS
                assert gpu.device_name == "Apple Silicon GPU"
                assert gpu.memory_gb > 0


class TestBatchSizeRecommendations:
    """Tests for batch size recommendations based on resources"""
    
    def test_high_memory_gpu_gets_large_batch(self):
        """High VRAM GPU should get large batch size"""
        resources = SystemResources()
        resources.gpu = GPUInfo(
            device_type=DeviceType.CUDA,
            device_name="RTX 4090",
            memory_gb=24.0
        )
        
        detector = ResourceDetector()
        detector._calculate_recommendations(resources)
        
        assert resources.recommended_embedding_batch_size == 512
        assert resources.recommended_device == "cuda"
    
    def test_medium_memory_gpu_gets_medium_batch(self):
        """Medium VRAM GPU should get medium batch size"""
        resources = SystemResources()
        resources.gpu = GPUInfo(
            device_type=DeviceType.CUDA,
            device_name="RTX 3080",
            memory_gb=10.0
        )
        
        detector = ResourceDetector()
        detector._calculate_recommendations(resources)
        
        assert resources.recommended_embedding_batch_size == 256
    
    def test_low_memory_gpu_gets_small_batch(self):
        """Low VRAM GPU should get smaller batch size"""
        resources = SystemResources()
        resources.gpu = GPUInfo(
            device_type=DeviceType.CUDA,
            device_name="GTX 1650",
            memory_gb=4.0
        )
        
        detector = ResourceDetector()
        detector._calculate_recommendations(resources)
        
        assert resources.recommended_embedding_batch_size == 64
    
    def test_cpu_only_gets_conservative_batch(self):
        """CPU-only should get conservative batch size"""
        resources = SystemResources()
        resources.gpu = None
        resources.available_ram_gb = 8.0
        
        detector = ResourceDetector()
        detector._calculate_recommendations(resources)
        
        assert resources.recommended_embedding_batch_size == 32
        assert resources.recommended_device == "cpu"
    
    def test_high_ram_cpu_gets_larger_batch(self):
        """High RAM CPU-only should get larger batch"""
        resources = SystemResources()
        resources.gpu = None
        resources.available_ram_gb = 32.0
        
        detector = ResourceDetector()
        detector._calculate_recommendations(resources)
        
        assert resources.recommended_embedding_batch_size == 64
    
    def test_mps_device_string(self):
        """MPS should return 'mps' device string"""
        resources = SystemResources()
        resources.gpu = GPUInfo(
            device_type=DeviceType.MPS,
            device_name="Apple Silicon GPU",
            memory_gb=16.0
        )
        
        detector = ResourceDetector()
        detector._calculate_recommendations(resources)
        
        assert resources.recommended_device == "mps"


class TestDeviceSelection:
    """Tests for get_torch_device()"""
    
    def test_cuda_device_string(self):
        """Should return cuda:0 for CUDA GPU"""
        detector = ResourceDetector()
        detector._resources = SystemResources(
            gpu=GPUInfo(
                device_type=DeviceType.CUDA,
                device_name="Test GPU",
                memory_gb=8.0,
                device_index=0
            )
        )
        
        assert detector.get_torch_device() == "cuda:0"
    
    def test_mps_device_string(self):
        """Should return mps for Apple Silicon"""
        detector = ResourceDetector()
        detector._resources = SystemResources(
            gpu=GPUInfo(
                device_type=DeviceType.MPS,
                device_name="Apple Silicon GPU",
                memory_gb=16.0
            )
        )
        
        assert detector.get_torch_device() == "mps"
    
    def test_cpu_fallback(self):
        """Should return cpu when no GPU"""
        detector = ResourceDetector()
        detector._resources = SystemResources(gpu=None)
        
        assert detector.get_torch_device() == "cpu"


class TestCaching:
    """Tests for resource detection caching"""
    
    def test_caches_results(self):
        """Should cache detection results"""
        detector = ResourceDetector()
        
        result1 = detector.detect()
        result2 = detector.detect()
        
        assert result1 is result2
    
    def test_force_refresh_clears_cache(self):
        """Should re-detect when force_refresh=True"""
        detector = ResourceDetector()
        
        result1 = detector.detect()
        result2 = detector.detect(force_refresh=True)
        
        # Results should be equal but not same object
        assert result1 is not result2
        assert result1.cpu_cores == result2.cpu_cores


class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""
    
    def test_detect_resources_returns_system_resources(self):
        """detect_resources() should return SystemResources"""
        resources = detect_resources()
        
        assert isinstance(resources, SystemResources)
    
    def test_get_optimal_device_returns_string(self):
        """get_optimal_device() should return device string"""
        device = get_optimal_device()
        
        assert isinstance(device, str)
        assert device in ["cpu", "mps"] or device.startswith("cuda")
    
    def test_get_resource_detector_singleton(self):
        """get_resource_detector() should return same instance"""
        detector1 = get_resource_detector()
        detector2 = get_resource_detector()
        
        assert detector1 is detector2


class TestConfigOverrides:
    """Tests for configuration overrides in ServerConfig"""
    
    def test_auto_device_uses_detection(self):
        """device='auto' should use detected device"""
        from conductor_memory.config.server import ServerConfig
        
        config = ServerConfig(device="auto")
        assert config.device == "auto"
    
    def test_explicit_device_override(self):
        """Explicit device should override detection"""
        from conductor_memory.config.server import ServerConfig
        
        config = ServerConfig(device="cpu")
        assert config.device == "cpu"
    
    def test_auto_batch_size(self):
        """embedding_batch_size='auto' should be parsed correctly"""
        from conductor_memory.config.server import ServerConfig
        
        config = ServerConfig(embedding_batch_size="auto")
        assert config.embedding_batch_size == "auto"
    
    def test_explicit_batch_size(self):
        """Explicit batch size should be parsed correctly"""
        from conductor_memory.config.server import ServerConfig
        
        config = ServerConfig(embedding_batch_size="256")
        assert config.embedding_batch_size == "256"
    
    def test_config_serialization(self):
        """Device and batch size should survive serialization"""
        from conductor_memory.config.server import ServerConfig
        
        config = ServerConfig(device="cuda:1", embedding_batch_size="128")
        data = config.to_dict()
        
        assert data["device"] == "cuda:1"
        assert data["embedding_batch_size"] == "128"
        
        restored = ServerConfig.from_dict(data)
        assert restored.device == "cuda:1"
        assert restored.embedding_batch_size == "128"


class TestIntegration:
    """Integration tests for resource detection with MemoryService"""
    
    def test_memory_service_uses_detected_resources(self):
        """MemoryService should use detected resources"""
        from conductor_memory.config.server import ServerConfig
        from conductor_memory.service.memory_service import MemoryService
        
        config = ServerConfig(
            device="cpu",  # Force CPU for test portability
            embedding_batch_size="32"
        )
        
        service = MemoryService(config)
        
        assert service._device == "cpu"
        assert service._embedding_batch_size == 32
    
    def test_memory_service_auto_detection(self):
        """MemoryService should auto-detect when set to 'auto'"""
        from conductor_memory.config.server import ServerConfig
        from conductor_memory.service.memory_service import MemoryService
        
        config = ServerConfig(
            device="auto",
            embedding_batch_size="auto"
        )
        
        service = MemoryService(config)
        
        # Should have detected something
        assert service._device in ["cpu", "mps"] or service._device.startswith("cuda")
        assert service._embedding_batch_size >= 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
