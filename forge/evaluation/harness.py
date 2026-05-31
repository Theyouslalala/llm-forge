import os
import json
import time
from pathlib import Path
from typing import Optional, Callable

import torch

from .metrics import compute_metrics, compute_perplexity
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EvalHarness:
    """Unified evaluation framework for LLM assessment."""

    def __init__(self, output_dir: str = "./outputs/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict] = {}

    def evaluate_generation(
        self,
        model: torch.nn.Module,
        tokenizer,
        test_data: list[dict],
        task_name: str = "text_generation",
        max_new_tokens: int = 128,
        device: str = "cuda",
        metrics: Optional[list[str]] = None,
    ) -> dict:
        logger.info(f"Evaluating task: {task_name} with {len(test_data)} samples")

        predictions = []
        references = []
        latencies = []

        model.eval()
        model = model.to(device)

        for item in test_data:
            prompt = item.get("prompt", "")
            reference = item.get("reference", "")

            input_ids = tokenizer.encode(prompt, add_special=True)
            input_tensor = torch.tensor([input_ids], device=device)

            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            latency = time.time() - start_time

            prediction = tokenizer.decode(output_ids[0].tolist(), skip_special=True)
            if prediction.startswith(prompt):
                prediction = prediction[len(prompt):]

            predictions.append(prediction.strip())
            references.append(reference.strip())
            latencies.append(latency)

        eval_metrics = compute_metrics(predictions, references, metrics)
        eval_metrics["avg_latency"] = sum(latencies) / len(latencies)
        eval_metrics["total_samples"] = len(test_data)

        self.results[task_name] = {
            "metrics": eval_metrics,
            "predictions": predictions[:5],
            "references": references[:5],
            "num_samples": len(test_data),
        }

        logger.info(f"Results for {task_name}: {eval_metrics}")
        return eval_metrics

    def evaluate_perplexity(
        self,
        model: torch.nn.Module,
        dataloader,
        device: str = "cuda",
    ) -> dict:
        model.eval()
        model = model.to(device)
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                total_loss += loss.item() * batch["input_ids"].numel()
                total_tokens += batch["input_ids"].numel()

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = compute_perplexity(avg_loss)

        result = {"loss": avg_loss, "perplexity": ppl, "total_tokens": total_tokens}
        self.results["perplexity"] = result
        logger.info(f"Perplexity: {ppl:.2f} (loss: {avg_loss:.4f})")
        return result

    def evaluate_qa(
        self,
        model: torch.nn.Module,
        tokenizer,
        qa_pairs: list[dict],
        device: str = "cuda",
    ) -> dict:
        return self.evaluate_generation(
            model, tokenizer, qa_pairs,
            task_name="question_answering",
            max_new_tokens=256,
            device=device,
            metrics=["exact_match", "f1", "rouge_l"],
        )

    def save_results(self, filename: str = "eval_results.json"):
        path = self.output_dir / filename
        serializable = {}
        for task, result in self.results.items():
            serializable[task] = {}
            for k, v in result.items():
                if isinstance(v, (int, float, str, list)):
                    serializable[task][k] = v
                elif isinstance(v, dict):
                    serializable[task][k] = {
                        sk: sv for sk, sv in v.items() if isinstance(sv, (int, float, str))
                    }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {path}")

    def summary(self) -> str:
        lines = ["=" * 50, "Evaluation Summary", "=" * 50]
        for task, result in self.results.items():
            lines.append(f"\nTask: {task}")
            metrics = result.get("metrics", result)
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                elif isinstance(v, int):
                    lines.append(f"  {k}: {v}")
        return "\n".join(lines)
