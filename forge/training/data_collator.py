import torch
from typing import Optional


class DataCollator:
    """Collate batches for different training stages."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        keys = batch[0].keys()
        result = {}
        for key in keys:
            result[key] = torch.stack([item[key] for item in batch])
        return result


class DPODataCollator:
    """Collate batches for DPO training."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        return {
            "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
            "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
            "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
            "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch]),
        }
