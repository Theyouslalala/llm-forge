import math
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..model.transformer import GPTModel, GPTConfig
from ..model.lora import LoRAConfig, apply_lora_to_model, get_lora_parameters
from ..tokenizer.bpe_tokenizer import BPETokenizer
from ..data.dataset import SFTDataset
from .trainer import Trainer
from .data_collator import DataCollator
from ..utils.logger import get_logger

logger = get_logger(__name__)


def sft_train(config_path: str = "configs/sft.yaml"):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model_cfg = config["model"]
    base_model_path = model_cfg["base_model"]

    tokenizer_path = os.path.join(base_model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "./outputs/tokenizer/tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)

    gpt_config = GPTConfig.from_dict(model_cfg)
    model = GPTModel(gpt_config)
    model_path = os.path.join(base_model_path, "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        logger.info(f"Loaded pretrained model from {model_path}")

    if model_cfg.get("use_lora", False):
        lora_config = LoRAConfig(
            rank=model_cfg["lora_rank"],
            alpha=model_cfg["lora_alpha"],
            dropout=model_cfg.get("lora_dropout", 0.0),
            target_modules=model_cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        model = apply_lora_to_model(model, lora_config)

        for param in model.parameters():
            param.requires_grad = False
        for param in get_lora_parameters(model):
            param.requires_grad = True

        trainable = model.count_trainable_parameters()
        total = model.count_parameters()
        logger.info(f"LoRA applied. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    train_cfg = config["training"]
    data_cfg = config["data"]

    train_dataset = SFTDataset(data_cfg["train_file"], tokenizer, data_cfg["max_length"])
    eval_dataset = SFTDataset(data_cfg["eval_file"], tokenizer, data_cfg["max_length"])

    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=train_cfg["batch_size"], collate_fn=collator)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = math.ceil(len(train_loader) * train_cfg["num_epochs"] / train_cfg["gradient_accumulation_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=train_cfg["learning_rate"] * 0.1
    )

    trainer = Trainer(
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

    logger.info("Starting SFT training...")
    history = trainer.train(
        train_loader,
        num_epochs=train_cfg["num_epochs"],
        eval_loader=eval_loader,
    )

    save_path = os.path.join(train_cfg["output_dir"], "best_model")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    logger.info(f"SFT model saved to {save_path}")

    return history


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sft.yaml"
    sft_train(config_path)
