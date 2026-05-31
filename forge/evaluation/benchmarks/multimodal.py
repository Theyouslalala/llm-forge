MULTIMODAL_BENCHMARK = [
    {
        "prompt": "请描述这张图片中的内容。",
        "reference": "图片中显示了一个场景",
        "image_required": True,
    },
    {
        "prompt": "图片中的主要颜色是什么？",
        "reference": "图片中包含多种颜色",
        "image_required": True,
    },
]


class MultimodalBenchmark:
    """Multimodal understanding benchmark."""

    def __init__(self):
        self.data = MULTIMODAL_BENCHMARK

    def evaluate(self, model, tokenizer, images=None, device: str = "cuda") -> dict:
        if images is None:
            return {"note": "Multimodal evaluation requires image inputs", "num_samples": len(self.data)}

        from ..harness import EvalHarness
        harness = EvalHarness()
        # Inject images into each data item for multimodal evaluation
        eval_data = []
        for item in self.data:
            eval_item = {**item, "images": images}
            eval_data.append(eval_item)
        results = harness.evaluate_generation(
            model, tokenizer, eval_data,
            task_name="multimodal_benchmark",
            device=device,
        )
        return results

    def get_data(self) -> list[dict]:
        return self.data
