import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for linear layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge(self):
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    def unmerge(self):
        self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target linear layers with LoRA layers."""
    for name, module in model.named_modules():
        for target in config.target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                attr_name = name.split(".")[-1]
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                    bias=module.bias is not None,
                )
                lora_layer.linear = module
                setattr(parent, attr_name, lora_layer)
    return model


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Extract only LoRA trainable parameters."""
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
