"""Sentence embedding module for KATM."""

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    """Embeds texts using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """Initialize SentenceEmbedder.

        Args:
            model_name: Name of the sentence-transformer model.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to dense embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.

        Returns:
            numpy.ndarray of shape (N, D) where D is embedding dimension.
        """
        if not texts:
            return np.array([])

        # Encode returns a numpy array directly
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for better clustering
        )

        return embeddings