import os
from typing import Optional

import torch
import yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.transformer import GPTModel, GPTConfig
from ..model.lora import LoRAConfig, apply_lora_to_model, get_lora_parameters
from ..tokenizer.bpe_tokenizer import BPETokenizer
from ..data.dataset import DPODataset
from .trainer import Trainer
from .data_collator import DPODataCollator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DPOTrainer(Trainer):
    """Trainer with DPO-specific loss computation."""

    def __init__(self, beta: float = 0.1, loss_type: str = "sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.loss_type = loss_type

    def _train_step(self, batch: dict) -> float:
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.fp16):
            chosen_outputs = self.model(input_ids=chosen_ids, attention_mask=chosen_mask, labels=chosen_ids)
            rejected_outputs = self.model(input_ids=rejected_ids, attention_mask=rejected_mask, labels=rejected_ids)

            chosen_logps = self._compute_log_probs(chosen_outputs["logits"], chosen_ids, chosen_mask)
            rejected_logps = self._compute_log_probs(rejected_outputs["logits"], rejected_ids, rejected_mask)

            logits = chosen_logps - rejected_logps
            if self.loss_type == "sigmoid":
                loss = -F.logsigmoid(self.beta * logits).mean()
            elif self.loss_type == "hinge":
                loss = torch.relu(1 - self.beta * logits).mean()
            else:
                loss = -F.logsigmoid(self.beta * logits).mean()

            loss = loss / self.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        self._step_counter += 1
        if self._step_counter % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps

    def _compute_log_probs(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * shift_mask.float()
        return token_log_probs.sum(-1)


def dpo_train(config_path: str = "configs/dpo.yaml"):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model_cfg = config["model"]
    sft_model_path = model_cfg["sft_model"]

    tokenizer_path = os.path.join(sft_model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "./outputs/tokenizer/tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)

    model = GPTModel(GPTConfig())
    model_path = os.path.join(sft_model_path, "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        logger.info(f"Loaded SFT model from {model_path}")

    if model_cfg.get("use_lora", False):
        lora_config = LoRAConfig(
            rank=model_cfg["lora_rank"],
            alpha=model_cfg["lora_alpha"],
            dropout=model_cfg.get("lora_dropout", 0.0),
        )
        model = apply_lora_to_model(model, lora_config)
        for param in model.parameters():
            param.requires_grad = False
        for param in get_lora_parameters(model):
            param.requires_grad = True

    train_cfg = config["training"]
    data_cfg = config["data"]

    train_dataset = DPODataset(data_cfg["train_file"], tokenizer, data_cfg["max_length"])
    eval_dataset = DPODataset(data_cfg["eval_file"], tokenizer, data_cfg["max_length"])

    collator = DPODataCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=train_cfg["batch_size"], collate_fn=collator)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])

    total_steps = len(train_loader) * train_cfg["num_epochs"] // train_cfg["gradient_accumulation_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    trainer = DPOTrainer(
        beta=train_cfg["beta"],
        loss_type=train_cfg.get("loss_type", "sigmoid"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=train_cfg["output_dir"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg.get("fp16", True),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        log_steps=train_cfg["log_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
    )

    logger.info("Starting DPO training...")
    history = trainer.train(train_loader, num_epochs=train_cfg["num_epochs"], eval_loader=eval_loader)

    save_path = os.path.join(train_cfg["output_dir"], "best_model")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    logger.info(f"DPO model saved to {save_path}")

    return history


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dpo.yaml"
    dpo_train(config_path)
