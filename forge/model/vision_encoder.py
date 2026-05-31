import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embeddings import RMSNorm


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings using Conv2d."""

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, d_model: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, d_model = x.shape
        qkv = self.qkv(x).reshape(bs, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(bs, seq_len, d_model)
        out = self.proj_dropout(self.proj(out))
        return out


class ViTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = ViTAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image feature extraction."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [ViTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(bs, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]
