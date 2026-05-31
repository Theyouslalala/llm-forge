import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .feedforward import SwiGLU
from .embeddings import TokenEmbedding, RMSNorm


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    tie_weights: bool = True
    rope_theta: float = 10000.0
    n_kv_heads: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.attention_norm(x)
        attn_out, present_kv = self.attention(
            x, attention_mask=attention_mask, use_cache=use_cache, past_kv=past_kv
        )
        x = residual + attn_out

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x, present_kv


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
    ) -> dict:
        x = self.token_embedding(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].float()
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min

        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, present_kv = layer(
                x, attention_mask=attention_mask, use_cache=use_cache, past_kv=past_kv
            )
            if use_cache:
                present_key_values.append(present_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values if use_cache else None,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int = 3,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_key_values is not None:
                    input_for_model = generated[:, -1:]
                else:
                    input_for_model = generated

                outputs = self.forward(
                    input_for_model, use_cache=True, past_key_values=past_key_values
                )
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]

                if temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, 1)
                    next_token = sorted_indices.gather(-1, next_token)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)

                if (next_token == eos_token_id).all():
                    break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
