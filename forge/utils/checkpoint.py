import os
import json
import shutil
from pathlib import Path
from typing import Optional

import torch

from .logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    def __init__(self, output_dir: str, max_keep: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints: list[str] = []

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        step: int,
        epoch: int,
        loss: float,
        extra: Optional[dict] = None,
    ) -> str:
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        meta = {"step": step, "epoch": epoch, "loss": loss}
        if extra:
            meta.update(extra)
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.checkpoints.append(str(ckpt_dir))
        if len(self.checkpoints) > self.max_keep:
            old = self.checkpoints.pop(0)
            if os.path.exists(old):
                shutil.rmtree(old)
                logger.info(f"Removed old checkpoint: {old}")

        logger.info(f"Saved checkpoint at step {step} to {ckpt_dir}")
        return str(ckpt_dir)

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
    ) -> dict:
        ckpt_dir = Path(checkpoint_path)
        model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=device)
        )
        if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(ckpt_dir / "optimizer.pt", map_location=device)
            )
        if scheduler is not None and (ckpt_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(ckpt_dir / "scheduler.pt", map_location=device)
            )

        meta_path = ckpt_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return meta

    def get_latest(self) -> Optional[str]:
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]
