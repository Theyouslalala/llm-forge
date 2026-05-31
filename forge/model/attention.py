import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embeddings import RotaryPositionalEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE and optional GQA support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, rope_theta)

        self._build_causal_mask(max_seq_len)

    def _build_causal_mask(self, max_seq_len: int):
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bs, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bs, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(bs, self.n_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bs, seq_len, _ = x.shape

        q = self.q_proj(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if use_cache:
            present_kv = (k, v)
        else:
            present_kv = None

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scale = math.sqrt(self.head_dim) if hasattr(self, "head_dim") else 1.0
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        kv_len = k.shape[2]
        causal = self.causal_mask[:seq_len, :kv_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = attn_weights + causal

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        out = self.o_proj(out)
        out = self.resid_dropout(out)

        return out, present_kv
