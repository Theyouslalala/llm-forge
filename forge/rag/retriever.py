import numpy as np
from typing import Optional

from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """Retrieve relevant documents using semantic search."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(query)

        if self.use_mmr:
            results = self._mmr_search(query_embedding, k)
        else:
            results = self.vector_store.search(query_embedding, top_k=k)

        results = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
        return results

    def _mmr_search(self, query_embedding: np.ndarray, k: int) -> list[dict]:
        candidates = self.vector_store.search(query_embedding, top_k=k * 3)
        if len(candidates) <= k:
            return candidates

        candidate_contents = [c["content"] for c in candidates]
        candidate_embeddings = self.embedder.embed(candidate_contents)

        selected = []
        remaining = list(range(len(candidates)))

        for _ in range(k):
            if not remaining:
                break

            best_score = -float("inf")
            best_idx = -1

            for idx in remaining:
                relevance = candidates[idx].get("score", 0)
                diversity = 0.0
                if selected:
                    for s in selected:
                        sim = self._cosine_sim(candidate_embeddings[idx], candidate_embeddings[s])
                        diversity = max(diversity, sim)

                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
