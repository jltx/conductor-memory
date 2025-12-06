"""
Sentence Transformer Embedder for the Hybrid Local/Cloud LLM Orchestrator

Provides high-quality text embeddings using sentence-transformers library.
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from ..core.embedder import Embedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(Embedder):
    """
    SentenceTransformer-based implementation of Embedder for high-quality text embeddings.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None):
        """
        Initialize the sentence transformer embedder

        Args:
            model_name: Name of the sentence transformer model
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
            cache_folder: Folder to cache downloaded models
        """
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder

        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder
            )
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
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
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            # Return zero vector as fallback
            dimension = self.model.get_sentence_embedding_dimension()
            return [0.0] * dimension

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of texts to embed

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

        try:
            # Generate embeddings for valid texts
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)

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