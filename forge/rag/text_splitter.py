import re
from typing import Optional


class TextSplitter:
    """Recursive character text splitter with overlap support."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[list[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", "，", " "]

    def split_text(self, text: str) -> list[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0] if separators else ""
        new_separators = separators[1:] if separators else []

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            piece = split + separator if separator and i < len(splits) - 1 else split
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(piece) > self.chunk_size and new_separators:
                    sub_chunks = self._recursive_split(piece, new_separators)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = piece

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return [c for c in chunks if c]

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            result.append(overlap_text + " " + chunks[i])
        return result

    def split_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "source": doc.get("source", ""),
                    "chunk_index": i,
                })
        return all_chunks
