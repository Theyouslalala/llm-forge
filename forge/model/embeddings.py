import math
import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        # Duplicate to match head_dim: (seq_len, head_dim/2) -> (seq_len, head_dim)
        cos_cached = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin_cached = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
