import os
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """FAISS-based vector store for document embeddings."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        store_path: Optional[str] = None,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.store_path = store_path
        self.index = None
        self.documents: list[dict] = []
        self._embeddings: list[np.ndarray] = []
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                self._ivf_trained = False
            elif self.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized FAISS index: {self.index_type}, dim={self.dimension}")
        except ImportError:
            logger.warning("FAISS not installed, using numpy fallback")
            self.index = None

    def add(self, embeddings: np.ndarray, documents: list[dict]):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if self.index is not None:
            import faiss
            embeddings = embeddings.astype("float32")
            faiss.normalize_L2(embeddings)
            if self.index_type == "ivf" and not getattr(self, "_ivf_trained", True):
                self.index.train(embeddings)
                self._ivf_trained = True
            self.index.add(embeddings)
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            for vec in normalized:
                self._embeddings.append(vec)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if len(self.documents) == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.index is not None:
            import faiss
            query_embedding = query_embedding.astype("float32")
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc["score"] = float(score)
                    results.append(doc)
            return results
        else:
            all_embeddings = self._get_numpy_embeddings()
            if all_embeddings is None:
                return []
            scores = np.dot(all_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                results.append(doc)
            return results

    def _get_numpy_embeddings(self) -> Optional[np.ndarray]:
        if not self._embeddings:
            return None
        return np.array(self._embeddings, dtype=np.float32)

    def save(self, path: Optional[str] = None):
        save_path = path or self.store_path
        if not save_path:
            return
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))
        elif self._embeddings:
            np.save(os.path.join(save_path, "embeddings.npy"), np.array(self._embeddings, dtype=np.float32))
        with open(os.path.join(save_path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        logger.info(f"Vector store saved to {save_path}")

    def load(self, path: str):
        if os.path.exists(os.path.join(path, "index.faiss")):
            import faiss
            self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        elif os.path.exists(os.path.join(path, "embeddings.npy")):
            self._embeddings = list(np.load(os.path.join(path, "embeddings.npy")))
        doc_path = os.path.join(path, "documents.json")
        if os.path.exists(doc_path):
            with open(doc_path, encoding="utf-8") as f:
                self.documents = json.load(f)
        logger.info(f"Vector store loaded from {path}")

    def __len__(self) -> int:
        return len(self.documents)
