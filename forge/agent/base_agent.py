import json
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from ..model.transformer import GPTModel, GPTConfig
from ..tokenizer.bpe_tokenizer import BPETokenizer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    """Base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def run(self, input_text: str) -> str:
        pass

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
        }


class BaseAgent(ABC):
    """Base agent with LLM backbone and tool integration."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_turns: int = 10,
    ):
        self.device = device
        self.max_turns = max_turns
        self.tools: dict[str, BaseTool] = {}
        self.model = None
        self.tokenizer = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        import os
        try:
            self.tokenizer = BPETokenizer.load(os.path.join(model_path, "tokenizer.json"))
            self.model = GPTModel(GPTConfig())
            self.model.load_state_dict(
                torch.load(os.path.join(model_path, "model.pt"), map_location="cpu", weights_only=True)
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Agent model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Agent will use template responses.")

    def register_tool(self, tool: BaseTool):
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.model is None or self.tokenizer is None:
            return "[模型未加载]"

        input_ids = self.tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(output_ids[0].tolist(), skip_special=True)
        return response[len(prompt):] if response.startswith(prompt) else response

    @abstractmethod
    def run(self, query: str) -> str:
        pass
