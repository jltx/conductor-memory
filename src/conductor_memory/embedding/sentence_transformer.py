"""
Sentence Transformer Embedder for the Hybrid Local/Cloud LLM Orchestrator

Provides high-quality text embeddings using sentence-transformers library.
"""

import logging
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from ..core.embedder import Embedder

logger = logging.getLogger(__name__)


def _check_mps_stability() -> bool:
    """
    Check if MPS (Apple Silicon GPU) is stable for embedding operations.
    
    MPS can cause segfaults with certain PyTorch/sentence-transformers versions,
    particularly due to:
    - Race conditions in the MPS backend (pytorch/pytorch#167541)
    - OpenMP conflicts on M4 chips (pytorch/pytorch#161865)
    - Threading issues with Metal Performance Shaders
    
    Returns:
        True if MPS appears stable, False otherwise
    """
    try:
        import torch
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            return False
        
        # Check PyTorch version - MPS is more stable in 2.1+
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version < (2, 1):
            logger.warning(f"PyTorch {torch.__version__} has known MPS stability issues. Consider upgrading to 2.1+")
        
        # Try a small tensor operation on MPS to check stability
        test_tensor = torch.randn(10, 10, device='mps')
        _ = test_tensor @ test_tensor.T
        
        # Synchronize to ensure operation completed
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        
        del test_tensor
        
        # Clear MPS cache
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        return True
    except Exception as e:
        logger.warning(f"MPS stability check failed: {e}")
        return False


def _get_mps_safe_batch_size(requested_size: int) -> int:
    """
    Get a safe batch size for MPS operations.
    
    MPS can be unstable with large batch sizes due to memory pressure
    and threading issues. This function caps the batch size to a safe value.
    
    Args:
        requested_size: The requested batch size
        
    Returns:
        A safe batch size for MPS
    """
    # MPS is more stable with smaller batches due to memory management
    # and potential race conditions in the Metal backend
    MPS_MAX_SAFE_BATCH = 64
    return min(requested_size, MPS_MAX_SAFE_BATCH)


class SentenceTransformerEmbedder(Embedder):
    """
    SentenceTransformer-based implementation of Embedder for high-quality text embeddings.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L12-v2",
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None):
        """
        Initialize the sentence transformer embedder

        Args:
            model_name: Name of the sentence transformer model
            device: Device to run on:
                - 'cpu': CPU only
                - 'cuda' or 'cuda:0': NVIDIA GPU
                - 'mps': Apple Silicon GPU (M1/M2/M3)
                - None: Auto-detect best available
            cache_folder: Folder to cache downloaded models
        """
        self.model_name = model_name
        self.cache_folder = cache_folder
        
        # Handle MPS stability issues
        # MPS can cause segfaults on some Apple Silicon configurations
        original_device = device
        if device == 'mps':
            # Check environment variable to force CPU
            if os.environ.get('CONDUCTOR_MEMORY_FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
                logger.info("CONDUCTOR_MEMORY_FORCE_CPU is set, using CPU instead of MPS")
                device = 'cpu'
            elif not _check_mps_stability():
                logger.warning(
                    "MPS stability check failed. Falling back to CPU. "
                    "Set CONDUCTOR_MEMORY_FORCE_CPU=1 to always use CPU, or "
                    "try updating PyTorch: pip install --upgrade torch"
                )
                device = 'cpu'
        
        self.device = device

        logger.info(f"Loading sentence transformer model: {model_name} on device: {device or 'auto'}")
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder
            )
            actual_device = str(self.model.device)
            logger.info(f"Model loaded on {actual_device}. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
            # For MPS, do a warmup embedding to catch any issues early
            if 'mps' in actual_device.lower():
                logger.info("Performing MPS warmup embedding...")
                try:
                    _ = self.model.encode("warmup test", convert_to_numpy=True, show_progress_bar=False)
                    logger.info("MPS warmup successful")
                except Exception as e:
                    logger.error(f"MPS warmup failed: {e}. Falling back to CPU.")
                    # Reload model on CPU
                    self.device = 'cpu'
                    self.model = SentenceTransformer(
                        model_name,
                        device='cpu',
                        cache_folder=cache_folder
                    )
                    logger.info(f"Model reloaded on CPU. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            dimension = self.model.get_sentence_embedding_dimension()
            return [0.0] * dimension

        try:
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            
            # Synchronize MPS to prevent race conditions
            self._sync_mps_if_needed()
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            # Return zero vector as fallback
            dimension = self.model.get_sentence_embedding_dimension()
            return [0.0] * dimension
    
    def _sync_mps_if_needed(self) -> None:
        """
        Synchronize MPS device if we're using it.
        
        This helps prevent race conditions and segfaults on Apple Silicon
        by ensuring all MPS operations complete before continuing.
        """
        if 'mps' in str(self.model.device).lower():
            try:
                import torch
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            except Exception:
                pass  # Ignore sync errors

    def generate_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size for GPU processing (auto-detected if None)

        Returns:
            List of embeddings (one per input text)
        """
        if not texts:
            return []

        # Filter out empty texts but keep track of indices
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            # All texts were empty
            dimension = self.model.get_sentence_embedding_dimension()
            return [[0.0] * dimension] * len(texts)

        # Use provided batch_size or auto-detect based on device
        device_str = str(self.model.device).lower()
        if batch_size is None:
            if 'cuda' in device_str:
                batch_size = 256  # Larger batches for NVIDIA GPU
            elif 'mps' in device_str:
                # MPS needs smaller batches to avoid race conditions and memory issues
                # See pytorch/pytorch#167541 for details on MPS threading issues
                batch_size = 64
            else:
                batch_size = 32   # Smaller for CPU
        elif 'mps' in device_str:
            # Cap MPS batch size even if user specified larger
            batch_size = _get_mps_safe_batch_size(batch_size)

        try:
            # Generate embeddings for valid texts
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
            
            # Synchronize MPS to prevent race conditions
            self._sync_mps_if_needed()

            # Reconstruct full result list with zero vectors for empty texts
            result = []
            embedding_idx = 0

            for i in range(len(texts)):
                if i in valid_indices:
                    result.append(embeddings[embedding_idx].tolist())
                    embedding_idx += 1
                else:
                    # Empty text - zero vector
                    dimension = self.model.get_sentence_embedding_dimension()
                    result.append([0.0] * dimension)

            return result

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors as fallback
            dimension = self.model.get_sentence_embedding_dimension()
            return [[0.0] * dimension] * len(texts)

    def get_dimension(self) -> int:
        """
        Get the embedding dimension for this model

        Returns:
            Integer dimension of embeddings
        """
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.get_dimension(),
            "device": str(self.model.device),
            "max_seq_length": getattr(self.model, 'max_seq_length', None)
        }