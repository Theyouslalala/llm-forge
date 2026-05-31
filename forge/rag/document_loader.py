import os
from pathlib import Path
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Load documents from various formats."""

    def load(self, path: str) -> list[dict]:
        path = Path(path)
        if path.is_dir():
            return self._load_directory(path)
        return [self._load_file(path)]

    def _load_directory(self, dir_path: Path) -> list[dict]:
        documents = []
        for file_path in dir_path.rglob("*"):
            if file_path.suffix in (".txt", ".md", ".pdf"):
                try:
                    doc = self._load_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        return documents

    def _load_file(self, file_path: Path) -> dict:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix in (".txt", ".md"):
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_text(self, file_path: Path) -> dict:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return {
            "content": content,
            "source": str(file_path),
            "type": file_path.suffix.lstrip("."),
        }

    def _load_pdf(self, file_path: Path) -> dict:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(file_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            content = "\n".join(pages)
        except ImportError:
            logger.warning("PyPDF2 not installed, falling back to basic loader")
            content = ""

        return {
            "content": content,
            "source": str(file_path),
            "type": "pdf",
        }
