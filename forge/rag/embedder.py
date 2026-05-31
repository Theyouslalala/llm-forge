import os
from typing import Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Text embedding using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using fallback embedder")
                self._model = "fallback"
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.model == "fallback":
            return self._fallback_embed(texts)
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def _fallback_embed(self, texts: list[str]) -> np.ndarray:
        """TF-IDF-like fallback embedding using character n-gram hashing."""
        dim = 384
        embeddings = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            # Character trigram hashing for basic semantic signal
            text_lower = text.lower().strip()
            for i in range(len(text_lower) - 2):
                trigram = text_lower[i:i+3]
                h = hash(trigram) % dim
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self.model == "fallback":
            return 384
        return self.model.get_sentence_embedding_dimension()
