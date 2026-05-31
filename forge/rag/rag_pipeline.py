import os
from typing import Optional

import yaml
import torch

from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Complete RAG pipeline: document loading -> splitting -> embedding -> retrieval -> generation."""

    def __init__(self, config_path: str = "configs/rag.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=self.config["text_splitter"]["chunk_size"],
            chunk_overlap=self.config["text_splitter"]["chunk_overlap"],
            separators=self.config["text_splitter"]["separators"],
        )
        self.embedder = Embedder(
            model_name=self.config["embedder"]["model_name"],
            device=self.config["embedder"].get("device", "cuda"),
        )
        self.vector_store = VectorStore(
            dimension=self.config["vector_store"]["dimension"],
            index_type=self.config["vector_store"]["index_type"],
            store_path=self.config["vector_store"].get("store_path"),
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.config["retriever"]["top_k"],
            similarity_threshold=self.config["retriever"]["similarity_threshold"],
            use_mmr=self.config["retriever"].get("use_mmr", False),
            mmr_lambda=self.config["retriever"].get("mmr_lambda", 0.5),
        )
        self._generator = None

    def ingest(self, path: str):
        """Load, split, and index documents."""
        logger.info(f"Loading documents from {path}")
        documents = self.loader.load(path)
        logger.info(f"Loaded {len(documents)} documents")

        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        texts = [c["content"] for c in chunks]
        logger.info("Computing embeddings...")
        embeddings = self.embedder.embed(texts, batch_size=self.config["embedder"].get("batch_size", 32))

        self.vector_store.add(embeddings, chunks)
        logger.info(f"Indexed {len(chunks)} chunks into vector store")

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """Retrieve relevant documents and generate answer."""
        results = self.retriever.retrieve(question, top_k=top_k)

        context = "\n\n".join([r["content"] for r in results])

        prompt = self._build_prompt(question, context)

        answer = self._generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "context": context,
        }

    def _build_prompt(self, question: str, context: str) -> str:
        system_prompt = self.config["generator"].get("system_prompt", "")
        prompt = f"""{system_prompt}

参考资料：
{context}

用户问题：{question}

请根据以上参考资料回答问题。如果参考资料中没有相关信息，请说明你不确定。

回答："""
        return prompt

    def _load_generator(self):
        """Load and cache the generator model and tokenizer."""
        if self._generator is not None:
            return self._generator

        gen_config = self.config["generator"]
        model_path = gen_config["model_path"]

        if not os.path.exists(model_path):
            return None

        from ..model.transformer import GPTModel, GPTConfig
        from ..tokenizer.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(os.path.join(model_path, "tokenizer.json"))
        model = GPTModel(GPTConfig())
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location="cpu", weights_only=True))
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        self._generator = {"model": model, "tokenizer": tokenizer, "device": device, "config": gen_config}
        return self._generator

    def _generate(self, prompt: str) -> str:
        gen = self._load_generator()
        if gen is None:
            model_path = self.config["generator"]["model_path"]
            return f"[模型未找到: {model_path}] 请先完成模型训练。基于检索到的上下文，以下是相关信息：\n{prompt[:500]}"

        try:
            model = gen["model"]
            tokenizer = gen["tokenizer"]
            device = gen["device"]
            gen_config = gen["config"]

            input_ids = tokenizer.encode(prompt, add_special=True)
            input_tensor = torch.tensor([input_ids], device=device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    max_new_tokens=gen_config.get("max_new_tokens", 512),
                    temperature=gen_config.get("temperature", 0.7),
                    top_p=gen_config.get("top_p", 0.9),
                    eos_token_id=tokenizer.eos_token_id,
                )

            answer = tokenizer.decode(output_ids[0].tolist(), skip_special=True)
            return answer[len(prompt):] if answer.startswith(prompt) else answer

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[生成失败: {e}]"

    def save(self):
        self.vector_store.save()

    def load(self):
        store_path = self.config["vector_store"].get("store_path")
        if store_path:
            self.vector_store.load(store_path)
