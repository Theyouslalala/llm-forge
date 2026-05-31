import torch
import torch.nn as nn
from typing import Optional

from .transformer import GPTModel, GPTConfig
from .vision_encoder import VisionTransformer


class MultimodalProjector(nn.Module):
    """Project vision features into language model space."""

    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultimodalModel(nn.Module):
    """Vision-Language Model combining ViT and GPT."""

    def __init__(
        self,
        vision_config: Optional[dict] = None,
        text_config: Optional[dict] = None,
    ):
        super().__init__()

        vision_cfg = vision_config or {}
        text_cfg = text_config or {}

        self.vision_encoder = VisionTransformer(
            image_size=vision_cfg.get("image_size", 224),
            patch_size=vision_cfg.get("patch_size", 16),
            d_model=vision_cfg.get("d_model", 768),
            n_heads=vision_cfg.get("n_heads", 12),
            n_layers=vision_cfg.get("n_layers", 12),
            d_ff=vision_cfg.get("d_ff", 3072),
            dropout=vision_cfg.get("dropout", 0.1),
        )

        gpt_config = GPTConfig.from_dict(text_cfg) if text_cfg else GPTConfig()
        self.language_model = GPTModel(gpt_config)

        self.projector = MultimodalProjector(
            vision_dim=vision_cfg.get("d_model", 768),
            text_dim=gpt_config.d_model,
        )

        self.image_token_id = -200

    def _merge_visual(
        self, input_ids: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        bs = input_ids.shape[0]
        text_embeds = self.language_model.token_embedding(input_ids)

        image_mask = (input_ids == self.image_token_id)
        if image_mask.any():
            projected_images = self.projector(image_features)
            text_embeds[image_mask] = projected_images.reshape(-1, projected_images.shape[-1])

        return text_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        if pixel_values is not None:
            image_features = self.vision_encoder(pixel_values)
            inputs_embeds = self._merge_visual(input_ids, image_features)

            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :].float()
                attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

            x = inputs_embeds
            for layer in self.language_model.layers:
                x, _ = layer(x, attention_mask=attention_mask)
            x = self.language_model.norm(x)
            logits = self.language_model.lm_head(x)

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            return {"loss": loss, "logits": logits}
        else:
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
