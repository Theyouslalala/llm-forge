from typing import Optional

import torch

from .harness import EvalHarness
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """High-level evaluator that orchestrates all evaluation tasks."""

    def __init__(self, output_dir: str = "./outputs/evaluation"):
        self.harness = EvalHarness(output_dir)

    def full_evaluation(
        self,
        model: torch.nn.Module,
        tokenizer,
        test_data: Optional[dict] = None,
        device: str = "cuda",
    ) -> dict:
        logger.info("Starting full evaluation...")
        all_results = {}

        if test_data and "generation" in test_data:
            gen_results = self.harness.evaluate_generation(
                model, tokenizer, test_data["generation"],
                task_name="text_generation", device=device,
            )
            all_results["generation"] = gen_results

        if test_data and "qa" in test_data:
            qa_results = self.harness.evaluate_qa(
                model, tokenizer, test_data["qa"], device=device,
            )
            all_results["qa"] = qa_results

        if test_data and "perplexity_loader" in test_data:
            ppl_results = self.harness.evaluate_perplexity(
                model, test_data["perplexity_loader"], device=device,
            )
            all_results["perplexity"] = ppl_results

        self.harness.save_results()
        summary = self.harness.summary()
        logger.info(summary)

        return all_results
