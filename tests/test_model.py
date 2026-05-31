"""Tests for model architecture components."""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.model.transformer import GPTConfig, GPTModel, TransformerBlock
from forge.model.attention import MultiHeadAttention
from forge.model.feedforward import SwiGLU, FeedForward
from forge.model.embeddings import TokenEmbedding, RotaryPositionalEmbedding, RMSNorm
from forge.model.lora import LoRALinear, LoRAConfig, apply_lora_to_model, get_lora_parameters
from forge.model.vision_encoder import VisionTransformer, PatchEmbedding
from forge.model.multimodal import MultimodalModel, MultimodalProjector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """Minimal GPT config for fast CPU tests."""
    return GPTConfig(
        vocab_size=256,
        max_seq_len=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        tie_weights=True,
        rope_theta=10000.0,
        n_kv_heads=None,
    )


@pytest.fixture
def small_model(small_config):
    """Small GPTModel on CPU with no dropout."""
    model = GPTModel(small_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GPTConfig
# ---------------------------------------------------------------------------

class TestGPTConfig:
    def test_default_values(self):
        cfg = GPTConfig()
        assert cfg.vocab_size == 32000
        assert cfg.max_seq_len == 1024
        assert cfg.d_model == 768
        assert cfg.n_heads == 12
        assert cfg.n_layers == 12
        assert cfg.d_ff == 3072
        assert cfg.dropout == 0.1
        assert cfg.tie_weights is True
        assert cfg.rope_theta == 10000.0
        assert cfg.n_kv_heads is None

    def test_from_dict(self):
        d = {"vocab_size": 1000, "d_model": 128, "n_heads": 4, "unknown_key": "ignored"}
        cfg = GPTConfig.from_dict(d)
        assert cfg.vocab_size == 1000
        assert cfg.d_model == 128
        assert cfg.n_heads == 4
        # unknown keys are silently ignored
        assert not hasattr(cfg, "unknown_key")

    def test_from_dict_partial(self):
        d = {"vocab_size": 512}
        cfg = GPTConfig.from_dict(d)
        assert cfg.vocab_size == 512
        assert cfg.d_model == 768  # default


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_output_dtype_preserved(self):
        norm = RMSNorm(32)
        x = torch.randn(1, 5, 32)
        y = norm(x)
        assert y.dtype == x.dtype

    def test_learnable_weight(self):
        norm = RMSNorm(16)
        assert norm.weight.requires_grad
        assert torch.allclose(norm.weight.data, torch.ones(16))


# ---------------------------------------------------------------------------
# TokenEmbedding
# ---------------------------------------------------------------------------

class TestTokenEmbedding:
    def test_output_shape(self):
        emb = TokenEmbedding(256, 64)
        ids = torch.randint(0, 256, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, 64)

    def test_padding_idx(self):
        emb = TokenEmbedding(100, 32, padding_idx=0)
        ids = torch.tensor([[0, 1, 2]])
        out = emb(ids)
        assert torch.all(out[0, 0] == 0)


# ---------------------------------------------------------------------------
# RotaryPositionalEmbedding
# ---------------------------------------------------------------------------

class TestRotaryPositionalEmbedding:
    def test_output_shapes(self):
        rope = RotaryPositionalEmbedding(d_model=32, max_seq_len=64)
        dummy = torch.zeros(1, 1, 32)
        cos, sin = rope(dummy, seq_len=10)
        assert cos.shape == (10, 32)
        assert sin.shape == (10, 32)

    def test_values_bounded(self):
        rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=32)
        dummy = torch.zeros(1, 1, 64)
        cos, sin = rope(dummy, seq_len=32)
        assert cos.abs().max() <= 1.0 + 1e-5
        assert sin.abs().max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0, max_seq_len=32)
        x = torch.randn(2, 10, 64)
        out, kv = attn(x)
        assert out.shape == (2, 10, 64)
        assert kv is None

    def test_causal_masking(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0, max_seq_len=16)
        x = torch.randn(1, 8, 32)
        # Changing future tokens should not affect earlier positions
        out1, _ = attn(x)
        x2 = x.clone()
        x2[0, 7, :] = 999.0
        out2, _ = attn(x2)
        # First position should be identical under causal masking
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-4)

    def test_kv_cache(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0, max_seq_len=32)
        x = torch.randn(1, 5, 32)
        out1, kv = attn(x, use_cache=True)
        assert kv is not None
        k, v = kv
        assert k.shape[2] == 5

        # Feed one more token with cache
        x2 = torch.randn(1, 1, 32)
        out2, kv2 = attn(x2, use_cache=True, past_kv=kv)
        assert kv2[0].shape[2] == 6  # 5 + 1

    def test_gqa(self):
        attn = MultiHeadAttention(d_model=64, n_heads=8, n_kv_heads=2, dropout=0.0, max_seq_len=16)
        x = torch.randn(1, 8, 64)
        out, _ = attn(x)
        assert out.shape == (1, 8, 64)


# ---------------------------------------------------------------------------
# FeedForward / SwiGLU
# ---------------------------------------------------------------------------

class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(d_model=64, d_ff=128)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_no_bias(self):
        ffn = SwiGLU(32, 64)
        assert ffn.gate_proj.bias is None
        assert ffn.up_proj.bias is None
        assert ffn.down_proj.bias is None


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(d_model=64, d_ff=128)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# GPTModel
# ---------------------------------------------------------------------------

class TestGPTModel:
    def test_forward_shape(self, small_model):
        ids = torch.randint(0, 256, (2, 16))
        out = small_model(ids)
        assert out["logits"].shape == (2, 16, 256)
        assert out["loss"] is None
        assert out["past_key_values"] is None

    def test_forward_with_labels(self, small_model):
        ids = torch.randint(0, 256, (1, 16))
        labels = torch.randint(0, 256, (1, 16))
        out = small_model(ids, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].dim() == 0  # scalar

    def test_forward_with_attention_mask(self, small_model):
        ids = torch.randint(0, 256, (1, 16))
        mask = torch.ones(1, 16)
        mask[0, 10:] = 0
        out = small_model(ids, attention_mask=mask)
        assert out["logits"].shape == (1, 16, 256)

    def test_count_parameters(self, small_model):
        count = small_model.count_parameters()
        assert count > 0
        # With tied weights, trainable == total
        assert count == small_model.count_trainable_parameters()

    def test_generate(self, small_model):
        ids = torch.randint(0, 256, (1, 4))
        out = small_model.generate(ids, max_new_tokens=5, temperature=0.0)
        assert out.shape[0] == 1
        assert out.shape[1] >= 4 + 1  # at least one new token

    def test_generate_deterministic(self, small_model):
        ids = torch.randint(0, 256, (1, 4))
        out1 = small_model.generate(ids, max_new_tokens=3, temperature=0.0)
        out2 = small_model.generate(ids, max_new_tokens=3, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_untied_weights(self):
        cfg = GPTConfig(vocab_size=128, d_model=32, n_heads=2, n_layers=1, d_ff=64,
                        max_seq_len=16, dropout=0.0, tie_weights=False)
        model = GPTModel(cfg)
        assert model.lm_head.weight is not model.token_embedding.embedding.weight

    def test_gqa_model(self):
        cfg = GPTConfig(vocab_size=128, d_model=32, n_heads=4, n_kv_heads=2,
                        n_layers=1, d_ff=64, max_seq_len=16, dropout=0.0)
        model = GPTModel(cfg)
        model.eval()
        ids = torch.randint(0, 128, (1, 8))
        out = model(ids)
        assert out["logits"].shape == (1, 8, 128)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_output_shape(self):
        lora = LoRALinear(64, 32, rank=4, alpha=8.0)
        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 32)

    def test_base_weights_frozen(self):
        lora = LoRALinear(64, 32, rank=4)
        assert not lora.linear.weight.requires_grad

    def test_lora_params_trainable(self):
        lora = LoRALinear(64, 32, rank=4)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_merge_unmerge(self):
        lora = LoRALinear(32, 32, rank=4)
        x = torch.randn(1, 5, 32)
        out_before = lora(x)

        lora.merge()
        out_merged = lora(x)
        # After merge, output should be similar (base weights changed)
        # Unmerge should restore original weights
        lora.unmerge()
        out_after = lora(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_scaling(self):
        lora = LoRALinear(16, 16, rank=4, alpha=8.0)
        assert lora.scaling == 2.0  # alpha / rank

    def test_dropout(self):
        lora = LoRALinear(32, 32, rank=4, dropout=0.5)
        lora.train()
        x = torch.randn(100, 10, 32)
        out = lora(x)
        assert out.shape == (100, 10, 32)


# ---------------------------------------------------------------------------
# apply_lora_to_model / get_lora_parameters
# ---------------------------------------------------------------------------

class TestApplyLoRA:
    def test_apply_to_gpt(self, small_config):
        model = GPTModel(small_config)
        total_before = model.count_trainable_parameters()

        lora_cfg = LoRAConfig(rank=4, alpha=8.0, dropout=0.0,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        apply_lora_to_model(model, lora_cfg)

        lora_params = get_lora_parameters(model)
        assert len(lora_params) > 0
        # Total trainable should be less than full model (base weights frozen)
        trainable_after = model.count_trainable_parameters()
        assert trainable_after < total_before

    def test_forward_after_lora(self, small_config):
        model = GPTModel(small_config)
        lora_cfg = LoRAConfig(rank=4, alpha=8.0)
        apply_lora_to_model(model, lora_cfg)
        model.eval()

        ids = torch.randint(0, 256, (1, 8))
        out = model(ids)
        assert out["logits"].shape == (1, 8, 256)


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------

class TestPatchEmbedding:
    def test_output_shape(self):
        pe = PatchEmbedding(image_size=32, patch_size=8, in_channels=3, d_model=64)
        assert pe.num_patches == (32 // 8) ** 2  # 16
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        assert out.shape == (2, 16, 64)


# ---------------------------------------------------------------------------
# VisionTransformer
# ---------------------------------------------------------------------------

class TestVisionTransformer:
    def test_output_shape(self):
        vit = VisionTransformer(
            image_size=32, patch_size=8, in_channels=3,
            d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0,
        )
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        # Output is CLS token: (batch, d_model)
        assert out.shape == (2, 64)

    def test_single_image(self):
        vit = VisionTransformer(
            image_size=32, patch_size=8, d_model=32, n_heads=2,
            n_layers=1, d_ff=64, dropout=0.0,
        )
        x = torch.randn(1, 3, 32, 32)
        out = vit(x)
        assert out.shape == (1, 32)


# ---------------------------------------------------------------------------
# MultimodalModel
# ---------------------------------------------------------------------------

class TestMultimodalModel:
    def test_text_only_forward(self):
        vision_cfg = {"image_size": 32, "patch_size": 8, "d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64}
        text_cfg = {"vocab_size": 128, "d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64,
                    "max_seq_len": 32, "dropout": 0.0}
        model = MultimodalModel(vision_config=vision_cfg, text_config=text_cfg)
        model.eval()

        ids = torch.randint(0, 128, (1, 10))
        out = model(input_ids=ids)
        assert out["logits"].shape == (1, 10, 128)

    def test_multimodal_forward(self):
        vision_cfg = {"image_size": 32, "patch_size": 8, "d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64}
        text_cfg = {"vocab_size": 128, "d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64,
                    "max_seq_len": 32, "dropout": 0.0}
        model = MultimodalModel(vision_config=vision_cfg, text_config=text_cfg)
        model.eval()

        # Sequence with one image token placeholder (-200)
        ids = torch.tensor([[1, 2, -200, 3, 4]])
        pixels = torch.randn(1, 3, 32, 32)
        out = model(input_ids=ids, pixel_values=pixels)
        assert out["logits"].shape == (1, 5, 128)

    def test_projector_shape(self):
        proj = MultimodalProjector(vision_dim=64, text_dim=32)
        x = torch.randn(2, 10, 64)
        out = proj(x)
        assert out.shape == (2, 10, 32)
