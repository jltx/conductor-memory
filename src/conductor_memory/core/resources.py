"""
OS-agnostic resource detection for conductor-memory.

Detects available compute resources (GPU, CPU, RAM) across different platforms
and provides optimal configuration recommendations.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Available compute device types"""
    CUDA = "cuda"       # NVIDIA GPU
    MPS = "mps"         # Apple Silicon GPU
    ROCm = "rocm"       # AMD GPU (via ROCm)
    CPU = "cpu"         # CPU fallback


@dataclass
class GPUInfo:
    """Information about an available GPU"""
    device_type: DeviceType
    device_name: str
    memory_gb: float
    device_index: int = 0


@dataclass
class SystemResources:
    """Detected system resources"""
    # GPU info (None if no GPU available)
    gpu: Optional[GPUInfo] = None
    
    # CPU info
    cpu_cores: int = 1
    cpu_threads: int = 1
    
    # Memory info (in GB)
    total_ram_gb: float = 4.0
    available_ram_gb: float = 2.0
    
    # Recommended settings based on detected resources
    recommended_device: str = "cpu"
    recommended_embedding_batch_size: int = 32
    recommended_index_batch_size: int = 10  # Files to process before yielding


class ResourceDetector:
    """
    Detects available system resources in an OS-agnostic way.
    
    Supports:
    - NVIDIA GPUs (via PyTorch CUDA)
    - Apple Silicon (via PyTorch MPS)
    - AMD GPUs (via ROCm, experimental)
    - CPU fallback with RAM/core detection
    """
    
    def __init__(self):
        self._resources: Optional[SystemResources] = None
    
    def detect(self, force_refresh: bool = False) -> SystemResources:
        """
        Detect available system resources.
        
        Args:
            force_refresh: If True, re-detect even if cached
            
        Returns:
            SystemResources with detected capabilities
        """
        if self._resources is not None and not force_refresh:
            return self._resources
        
        resources = SystemResources()
        
        # Detect CPU and RAM
        resources.cpu_cores, resources.cpu_threads = self._detect_cpu()
        resources.total_ram_gb, resources.available_ram_gb = self._detect_ram()
        
        # Detect GPU (try each backend in order of preference)
        resources.gpu = (
            self._detect_cuda_gpu() or
            self._detect_mps_gpu() or
            self._detect_rocm_gpu()
        )
        
        # Calculate recommendations based on detected resources
        self._calculate_recommendations(resources)
        
        self._resources = resources
        self._log_detection_results(resources)
        
        return resources
    
    def _detect_cpu(self) -> Tuple[int, int]:
        """Detect CPU cores and threads"""
        cores = 1
        threads = 1
        
        try:
            # os.cpu_count() returns logical processors (threads)
            threads = os.cpu_count() or 1
            
            # Try to get physical cores via psutil
            try:
                import psutil
                cores = psutil.cpu_count(logical=False) or threads
            except ImportError:
                # Estimate physical cores as half of threads (common for hyperthreading)
                cores = max(1, threads // 2)
        except Exception as e:
            logger.debug(f"Error detecting CPU: {e}")
        
        return cores, threads
    
    def _detect_ram(self) -> Tuple[float, float]:
        """Detect total and available RAM in GB"""
        total_gb = 4.0  # Conservative default
        available_gb = 2.0
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)
        except ImportError:
            logger.debug("psutil not available, using default RAM estimates")
        except Exception as e:
            logger.debug(f"Error detecting RAM: {e}")
        
        return total_gb, available_gb
    
    def _detect_cuda_gpu(self) -> Optional[GPUInfo]:
        """Detect NVIDIA GPU via PyTorch CUDA"""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return GPUInfo(
                    device_type=DeviceType.CUDA,
                    device_name=props.name,
                    memory_gb=props.total_memory / (1024 ** 3),
                    device_index=0
                )
            else:
                # Check if NVIDIA GPU might be present but PyTorch lacks CUDA support
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_name = result.stdout.strip().split('\n')[0]
                        logger.warning(
                            f"NVIDIA GPU detected ({gpu_name}) but PyTorch CUDA is not available. "
                            f"Install PyTorch with CUDA support for GPU acceleration: "
                            f"pip install torch --index-url https://download.pytorch.org/whl/cu121"
                        )
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                    pass  # nvidia-smi not available
        except ImportError:
            logger.debug("PyTorch not available for CUDA detection")
        except Exception as e:
            logger.debug(f"Error detecting CUDA GPU: {e}")
        
        return None
    
    def _detect_mps_gpu(self) -> Optional[GPUInfo]:
        """Detect Apple Silicon GPU via PyTorch MPS"""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't expose memory info directly
                # Estimate based on unified memory (typically 50-75% available for GPU)
                try:
                    import psutil
                    total_ram = psutil.virtual_memory().total / (1024 ** 3)
                    # Apple Silicon uses unified memory, estimate GPU portion
                    estimated_gpu_mem = total_ram * 0.6
                except ImportError:
                    estimated_gpu_mem = 8.0  # Conservative default
                
                return GPUInfo(
                    device_type=DeviceType.MPS,
                    device_name="Apple Silicon GPU",
                    memory_gb=estimated_gpu_mem,
                    device_index=0
                )
        except ImportError:
            logger.debug("PyTorch not available for MPS detection")
        except Exception as e:
            logger.debug(f"Error detecting MPS GPU: {e}")
        
        return None
    
    def _detect_rocm_gpu(self) -> Optional[GPUInfo]:
        """Detect AMD GPU via ROCm/HIP"""
        try:
            import torch
            # ROCm uses the same CUDA API in PyTorch when built with ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    return GPUInfo(
                        device_type=DeviceType.ROCm,
                        device_name=props.name,
                        memory_gb=props.total_memory / (1024 ** 3),
                        device_index=0
                    )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error detecting ROCm GPU: {e}")
        
        return None
    
    def _calculate_recommendations(self, resources: SystemResources) -> None:
        """Calculate recommended settings based on detected resources"""
        
        if resources.gpu:
            resources.recommended_device = resources.gpu.device_type.value
            gpu_mem = resources.gpu.memory_gb
            
            # Embedding batch size based on GPU memory
            # MiniLM models are small, can handle large batches
            if gpu_mem >= 20:      # RTX 4090, A100, etc.
                resources.recommended_embedding_batch_size = 512
            elif gpu_mem >= 10:    # RTX 3080, etc.
                resources.recommended_embedding_batch_size = 256
            elif gpu_mem >= 6:     # RTX 3060, GTX 1660, etc.
                resources.recommended_embedding_batch_size = 128
            elif gpu_mem >= 4:     # GTX 1650, Apple M1, etc.
                resources.recommended_embedding_batch_size = 64
            else:
                resources.recommended_embedding_batch_size = 32
            
            # Index batch size (files per batch) - GPU can handle more
            resources.recommended_index_batch_size = 50
        
        else:
            # CPU-only mode
            resources.recommended_device = "cpu"
            
            # Base on available RAM and CPU cores
            if resources.available_ram_gb >= 16:
                resources.recommended_embedding_batch_size = 64
            elif resources.available_ram_gb >= 8:
                resources.recommended_embedding_batch_size = 32
            else:
                resources.recommended_embedding_batch_size = 16
            
            # Index batch size based on CPU cores
            resources.recommended_index_batch_size = max(5, resources.cpu_cores * 2)
    
    def _log_detection_results(self, resources: SystemResources) -> None:
        """Log the detection results"""
        if resources.gpu:
            logger.info(
                f"Detected {resources.gpu.device_type.value.upper()} GPU: "
                f"{resources.gpu.device_name} ({resources.gpu.memory_gb:.1f}GB)"
            )
        else:
            logger.info("No GPU detected, using CPU")
        
        logger.info(
            f"System: {resources.cpu_cores} cores, {resources.cpu_threads} threads, "
            f"{resources.total_ram_gb:.1f}GB RAM ({resources.available_ram_gb:.1f}GB available)"
        )
        logger.info(
            f"Recommended settings: device={resources.recommended_device}, "
            f"embedding_batch_size={resources.recommended_embedding_batch_size}"
        )
    
    def get_torch_device(self) -> str:
        """
        Get the appropriate PyTorch device string.
        
        Returns:
            Device string for torch.device() or SentenceTransformer
        """
        resources = self.detect()
        
        if resources.gpu:
            if resources.gpu.device_type == DeviceType.CUDA:
                return f"cuda:{resources.gpu.device_index}"
            elif resources.gpu.device_type == DeviceType.MPS:
                return "mps"
            elif resources.gpu.device_type == DeviceType.ROCm:
                return f"cuda:{resources.gpu.device_index}"  # ROCm uses CUDA API
        
        return "cpu"


# Global singleton for resource detection
_detector: Optional[ResourceDetector] = None


def get_resource_detector() -> ResourceDetector:
    """Get the global resource detector instance"""
    global _detector
    if _detector is None:
        _detector = ResourceDetector()
    return _detector


def detect_resources(force_refresh: bool = False) -> SystemResources:
    """
    Convenience function to detect system resources.
    
    Args:
        force_refresh: If True, re-detect even if cached
        
    Returns:
        SystemResources with detected capabilities
    """
    return get_resource_detector().detect(force_refresh)


def get_optimal_device() -> str:
    """
    Get the optimal PyTorch device string for this system.
    
    Returns:
        Device string (e.g., "cuda:0", "mps", "cpu")
    """
    return get_resource_detector().get_torch_device()
