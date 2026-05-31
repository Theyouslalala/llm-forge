import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..model.transformer import GPTModel, GPTConfig
from ..tokenizer.bpe_tokenizer import BPETokenizer
from ..data.dataset import PretrainDataset
from ..data.sample_data import generate_sample_data
from .trainer import Trainer
from .data_collator import DataCollator
from ..utils.logger import get_logger

logger = get_logger(__name__)


def pretrain(config_path: str = "configs/pretrain.yaml"):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    data_cfg = config["data"]
    if not os.path.exists(data_cfg["train_file"]):
        logger.info("Generating sample data...")
        generate_sample_data()

    tokenizer_path = "./outputs/tokenizer/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        logger.info("Training tokenizer...")
        from ..tokenizer.train_tokenizer import train_tokenizer
        tokenizer = train_tokenizer(
            data_path=data_cfg["train_file"],
            vocab_size=config["tokenizer"]["vocab_size"],
            output_path=tokenizer_path,
        )

    model_config = GPTConfig.from_dict(config["model"])
    model = GPTModel(model_config)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    train_dataset = PretrainDataset(
        data_cfg["train_file"], tokenizer, data_cfg["max_length"]
    )
    eval_dataset = PretrainDataset(
        data_cfg["eval_file"], tokenizer, data_cfg["max_length"]
    )

    train_cfg = config["training"]
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=data_cfg.get("preprocessing_num_workers", 0),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=train_cfg["batch_size"], collate_fn=collator)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * train_cfg["num_epochs"] // train_cfg["gradient_accumulation_steps"]
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

    logger.info("Starting pretraining...")
    history = trainer.train(
        train_loader,
        num_epochs=train_cfg["num_epochs"],
        eval_loader=eval_loader,
    )

    save_path = os.path.join(train_cfg["output_dir"], "best_model")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    logger.info(f"Model saved to {save_path}")

    return history


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/pretrain.yaml"
    pretrain(config_path)
