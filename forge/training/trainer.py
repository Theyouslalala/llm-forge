import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..utils.logger import get_logger
from ..utils.checkpoint import CheckpointManager

logger = get_logger(__name__)


class Trainer:
    """Universal trainer supporting pretraining, SFT, and DPO."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        gradient_accumulation_steps: int = 1,
        log_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 500,
        max_keep: int = 5,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps

        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16)
        self.ckpt_manager = CheckpointManager(output_dir, max_keep=max_keep)
        self.global_step = 0
        self._step_counter = 0
        self.best_eval_loss = float("inf")

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ) -> dict:
        if resume_from:
            meta = self.ckpt_manager.load(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            self.global_step = meta.get("step", 0)

        self.model.train()
        history = {"train_loss": [], "eval_loss": [], "lr": []}

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            start_time = time.time()

            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self._train_step(batch)
                epoch_loss += loss
                epoch_steps += 1

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.global_step += 1

                    if self.global_step % self.log_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs} | Step {self.global_step} | "
                            f"Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
                        )
                        history["train_loss"].append(avg_loss)
                        history["lr"].append(lr)

                    if eval_dataloader and self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        history["eval_loss"].append(eval_loss)
                        logger.info(f"Eval Loss: {eval_loss:.4f}")
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.ckpt_manager.save(
                                self.model, self.optimizer, self.scheduler,
                                self.global_step, epoch, eval_loss,
                                {"best": True},
                            )
                        self.model.train()

                    if self.global_step % self.save_steps == 0:
                        self.ckpt_manager.save(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, loss,
                        )

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        return history

    def _train_step(self, batch: dict) -> float:
        with torch.amp.autocast("cuda", enabled=self.fp16):
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
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

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=self.fp16):
                outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            total_loss += loss.item()
            total_steps += 1

        return total_loss / max(total_steps, 1)
