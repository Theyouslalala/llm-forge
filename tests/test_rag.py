"""Tests for RAG components: TextSplitter, Embedder, VectorStore, Retriever."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.rag.text_splitter import TextSplitter
from forge.rag.embedder import Embedder
from forge.rag.vector_store import VectorStore
from forge.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# TextSplitter
# ---------------------------------------------------------------------------

class TestTextSplitter:
    def test_short_text_not_split(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        text = "Hello world"
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_text("")
        assert chunks == []

    def test_whitespace_only(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_text("   \n\n  ")
        assert chunks == []

    def test_long_text_split(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        text = "A" * 50
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 20

    def test_split_by_newline(self):
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0, separators=["\n", " "])
        text = "Line one\nLine two\nLine three"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_overlap(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        text = "A" * 40
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_documents(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        docs = [
            {"content": "First document content.", "source": "doc1.txt"},
            {"content": "Second document content here.", "source": "doc2.txt"},
        ]
        chunks = splitter.split_documents(docs)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "content" in chunk
            assert "source" in chunk
            assert "chunk_index" in chunk

    def test_split_documents_preserves_source(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        docs = [{"content": "Short doc.", "source": "test.md"}]
        chunks = splitter.split_documents(docs)
        assert chunks[0]["source"] == "test.md"

    def test_chinese_text_split(self):
        splitter = TextSplitter(chunk_size=15, chunk_overlap=0)
        text = "这是一个测试文本，用于验证中文分句功能。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_recursive_split_deep(self):
        splitter = TextSplitter(chunk_size=10, chunk_overlap=0,
                                separators=["\n\n", "\n", " ", ""])
        text = "A" * 25
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 10


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_fallback_embed_shape(self):
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "test"
        embedder.device = "cpu"
        embedder._model = "fallback"

        embeddings = embedder.embed(["hello", "world"])
        assert embeddings.shape == (2, 384)

    def test_fallback_embed_normalized(self):
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "test"
        embedder.device = "cpu"
        embedder._model = "fallback"

        embeddings = embedder.embed(["test text"])
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_fallback_embed_deterministic(self):
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "test"
        embedder.device = "cpu"
        embedder._model = "fallback"

        e1 = embedder.embed(["same text"])
        e2 = embedder.embed(["same text"])
        np.testing.assert_array_equal(e1, e2)

    def test_fallback_embed_different_texts(self):
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "test"
        embedder.device = "cpu"
        embedder._model = "fallback"

        e1 = embedder.embed(["cat"])
        e2 = embedder.embed(["dog"])
        # Different texts should produce different embeddings
        assert not np.allclose(e1, e2)

    def test_embed_query(self):
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "test"
        embedder.device = "cpu"
        embedder._model = "fallback"

        q = embedder.embed_query("test query")
        assert q.shape == (384,)

    def test_dimension_fallback(self):
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "test"
        embedder.device = "cpu"
        embedder._model = "fallback"
        assert embedder.dimension == 384


# ---------------------------------------------------------------------------
# VectorStore (numpy fallback, no faiss)
# ---------------------------------------------------------------------------

class TestVectorStore:
    def _make_store(self, dim=384):
        """Create a VectorStore with numpy fallback (no FAISS)."""
        store = VectorStore.__new__(VectorStore)
        store.dimension = dim
        store.index_type = "flat"
        store.store_path = None
        store.index = None  # Force numpy fallback
        store.documents = []
        store._embeddings = []
        return store

    def test_add_single(self):
        store = self._make_store(dim=4)
        emb = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        store.add(emb, [{"content": "doc1"}])
        assert len(store) == 1

    def test_add_multiple(self):
        store = self._make_store(dim=4)
        embs = np.random.randn(3, 4).astype(np.float32)
        docs = [{"content": f"doc{i}"} for i in range(3)]
        store.add(embs, docs)
        assert len(store) == 3

    def test_add_1d_embedding(self):
        store = self._make_store(dim=4)
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        store.add(emb, [{"content": "single"}])
        assert len(store) == 1

    def test_search_empty(self):
        store = self._make_store(dim=4)
        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
        assert results == []

    def test_search_returns_scores(self):
        store = self._make_store(dim=4)
        embs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        docs = [{"content": "x-axis"}, {"content": "y-axis"}]
        store.add(embs, docs)

        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, top_k=2)
        assert len(results) == 2
        assert all("score" in r for r in results)
        # First result should be x-axis (higher similarity)
        assert results[0]["content"] == "x-axis"

    def test_search_top_k(self):
        store = self._make_store(dim=4)
        embs = np.random.randn(10, 4).astype(np.float32)
        docs = [{"content": f"doc{i}"} for i in range(10)]
        store.add(embs, docs)

        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=3)
        assert len(results) == 3

    def test_len(self):
        store = self._make_store()
        assert len(store) == 0
        store.add(np.random.randn(2, 4).astype(np.float32), [{"content": "a"}, {"content": "b"}])
        assert len(store) == 2


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def _make_retriever(self, dim=384, threshold=0.0):
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "test"
        embedder.device = "cpu"
        embedder._model = "fallback"

        store = VectorStore.__new__(VectorStore)
        store.dimension = dim
        store.index_type = "flat"
        store.store_path = None
        store.index = None
        store.documents = []
        store._embeddings = []

        return Retriever(
            embedder=embedder,
            vector_store=store,
            top_k=3,
            similarity_threshold=threshold,
        )

    def test_retrieve_basic(self):
        r = self._make_retriever(threshold=-1.0)

        # Add documents
        texts = ["cat says meow", "dog says woow", "bird tweets"]
        for text in texts:
            emb = r.embedder.embed([text])
            r.vector_store.add(emb, [{"content": text}])

        results = r.retrieve("cat")
        assert len(results) >= 1
        assert all("content" in doc for doc in results)
        assert all("score" in doc for doc in results)

    def test_retrieve_top_k(self):
        r = self._make_retriever(threshold=-1.0)

        for i in range(10):
            emb = r.embedder.embed([f"document number {i}"])
            r.vector_store.add(emb, [{"content": f"doc{i}"}])

        results = r.retrieve("query", top_k=2)
        assert len(results) <= 2

    def test_retrieve_threshold_filter(self):
        r = self._make_retriever(threshold=0.99)

        emb = r.embedder.embed(["completely different text"])
        r.vector_store.add(emb, [{"content": "unrelated"}])

        # With high threshold, likely nothing passes
        results = r.retrieve("something else entirely different query xyz")
        # At least the filter is applied (results may be 0 or 1)
        for doc in results:
            assert doc["score"] >= 0.99

    def test_mmr_retrieval(self):
        r = self._make_retriever(threshold=-1.0)
        r.use_mmr = True
        r.mmr_lambda = 0.5

        texts = ["Python programming", "Java programming", "Cooking recipes"]
        for text in texts:
            emb = r.embedder.embed([text])
            r.vector_store.add(emb, [{"content": text}])

        results = r.retrieve("programming", top_k=2)
        assert len(results) >= 1

    def test_cosine_similarity(self):
        r = self._make_retriever()
        import numpy as np
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        sim_same = r._cosine_sim(a, b)
        sim_diff = r._cosine_sim(a, c)
        assert sim_same > sim_diff
