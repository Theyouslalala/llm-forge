import json
from typing import Optional
from pathlib import Path

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentMemory:
    """Agent memory system with short-term (conversation) and long-term (vector) memory."""

    def __init__(self, max_short_term: int = 20, embedding_dim: int = 384):
        self.max_short_term = max_short_term
        self.embedding_dim = embedding_dim
        self.short_term: list[dict] = []
        self.long_term: list[dict] = []
        self.long_term_embeddings: list[np.ndarray] = []

    def add_short_term(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]

    def add_long_term(self, content: str, metadata: Optional[dict] = None):
        entry = {
            "content": content,
            "metadata": metadata or {},
        }
        self.long_term.append(entry)
        embedding = self._simple_embed(content)
        self.long_term_embeddings.append(embedding)
        logger.info(f"Added to long-term memory: {content[:50]}...")

    def search_long_term(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.long_term_embeddings:
            return []

        query_embed = self._simple_embed(query)
        scores = []
        for emb in self.long_term_embeddings:
            score = np.dot(query_embed, emb) / (np.linalg.norm(query_embed) * np.linalg.norm(emb) + 1e-8)
            scores.append(score)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            entry = self.long_term[idx].copy()
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results

    def get_context(self, include_short_term: bool = True, query: Optional[str] = None, long_term_k: int = 2) -> str:
        context_parts = []

        if include_short_term and self.short_term:
            context_parts.append("对话历史:")
            for msg in self.short_term[-10:]:
                context_parts.append(f"  {msg['role']}: {msg['content']}")

        if query and self.long_term:
            relevant = self.search_long_term(query, top_k=long_term_k)
            if relevant:
                context_parts.append("\n相关记忆:")
                for mem in relevant:
                    context_parts.append(f"  - {mem['content']}")

        return "\n".join(context_parts)

    def _simple_embed(self, text: str) -> np.ndarray:
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        vec = np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), self.embedding_dim * 2), 2)])
        if len(vec) < self.embedding_dim:
            vec = np.pad(vec, (0, self.embedding_dim - len(vec)))
        else:
            vec = vec[:self.embedding_dim]
        return vec / (np.linalg.norm(vec) + 1e-8)

    def save(self, path: str):
        data = {
            "short_term": self.short_term,
            "long_term": self.long_term,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.short_term = data.get("short_term", [])
        self.long_term = data.get("long_term", [])
        self.long_term_embeddings = [
            self._simple_embed(entry["content"]) for entry in self.long_term
        ]

    def clear_short_term(self):
        self.short_term.clear()

    def clear_all(self):
        self.short_term.clear()
        self.long_term.clear()
        self.long_term_embeddings.clear()
