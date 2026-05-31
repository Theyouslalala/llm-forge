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
        import hashlib
        dim = 384
        embeddings = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), dim*2), 2)])
            if len(vec) < dim:
                vec = np.pad(vec, (0, dim - len(vec)))
            else:
                vec = vec[:dim]
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self.model == "fallback":
            return 384
        return self.model.get_sentence_embedding_dimension()
