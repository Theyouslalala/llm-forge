"""Tests for data processing: datasets, collators, and preprocessor."""

import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.data.dataset import PretrainDataset, SFTDataset, DPODataset
from forge.data.preprocessor import DataPreprocessor
from forge.training.data_collator import DataCollator, DPODataCollator


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer stub for dataset tests."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    @property
    def pad_token_id(self):
        return 0

    @property
    def bos_token_id(self):
        return 2

    @property
    def eos_token_id(self):
        return 3

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = []
        if add_special:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append((ord(ch) % (self.vocab_size - 4)) + 4)
        if add_special:
            ids.append(self.eos_token_id)
        return ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer():
    return MockTokenizer(vocab_size=100)


@pytest.fixture
def pretrain_file(tokenizer, tmp_path):
    """Create a temporary pretrain text file."""
    lines = [
        "This is a test sentence for pretraining.",
        "Another line of sample text data.",
        "Third line here.",
    ]
    path = tmp_path / "pretrain.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def sft_file(tmp_path):
    """Create a temporary SFT jsonl file."""
    data = [
        {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
        {"instruction": "Explain ML", "input": "", "output": "Machine learning is a subset of AI."},
    ]
    path = tmp_path / "sft.jsonl"
    path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in data),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def dpo_file(tmp_path):
    """Create a temporary DPO jsonl file."""
    data = [
        {"prompt": "What is Python?", "chosen": "A programming language.", "rejected": "A snake."},
        {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"},
    ]
    path = tmp_path / "dpo.jsonl"
    path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in data),
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# PretrainDataset
# ---------------------------------------------------------------------------

class TestPretrainDataset:
    def test_length(self, pretrain_file, tokenizer):
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=32)
        assert len(ds) == 3

    def test_item_keys(self, pretrain_file, tokenizer):
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_tensor_shapes(self, pretrain_file, tokenizer):
        max_len = 32
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=max_len)
        item = ds[0]
        assert item["input_ids"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)
        assert item["labels"].shape == (max_len,)

    def test_padding(self, pretrain_file, tokenizer):
        max_len = 128
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=max_len)
        item = ds[0]
        # Attention mask should have zeros for padded positions
        real_len = len(tokenizer.encode(ds.samples[0], add_special=True))
        assert item["attention_mask"][:real_len].sum() == real_len
        assert item["attention_mask"][real_len:].sum() == 0

    def test_labels_padding_is_ignore_index(self, pretrain_file, tokenizer):
        max_len = 128
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=max_len)
        item = ds[0]
        real_len = len(tokenizer.encode(ds.samples[0], add_special=True))
        assert (item["labels"][real_len:] == -100).all()

    def test_truncation(self, pretrain_file, tokenizer):
        max_len = 5
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=max_len)
        item = ds[0]
        assert item["input_ids"].shape == (max_len,)


# ---------------------------------------------------------------------------
# SFTDataset
# ---------------------------------------------------------------------------

class TestSFTDataset:
    def test_length(self, sft_file, tokenizer):
        ds = SFTDataset(sft_file, tokenizer, max_length=64)
        assert len(ds) == 2

    def test_prompt_masked_in_labels(self, sft_file, tokenizer):
        ds = SFTDataset(sft_file, tokenizer, max_length=128)
        item = ds[0]
        # Prompt tokens should be masked with -100
        prompt = json.loads(ds.samples[0].rstrip() if hasattr(ds, 'samples') else '{}')
        # Just check that some -100 labels exist (prompt portion)
        assert (item["labels"] == -100).any()

    def test_output_present_in_labels(self, sft_file, tokenizer):
        ds = SFTDataset(sft_file, tokenizer, max_length=128)
        item = ds[0]
        # Some labels should not be -100 (the output portion)
        assert (item["labels"] != -100).any()

    def test_tensor_dtypes(self, sft_file, tokenizer):
        ds = SFTDataset(sft_file, tokenizer, max_length=64)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["labels"].dtype == torch.long


# ---------------------------------------------------------------------------
# DPODataset
# ---------------------------------------------------------------------------

class TestDPODataset:
    def test_length(self, dpo_file, tokenizer):
        ds = DPODataset(dpo_file, tokenizer, max_length=64)
        assert len(ds) == 2

    def test_item_keys(self, dpo_file, tokenizer):
        ds = DPODataset(dpo_file, tokenizer, max_length=64)
        item = ds[0]
        assert "chosen_input_ids" in item
        assert "chosen_attention_mask" in item
        assert "rejected_input_ids" in item
        assert "rejected_attention_mask" in item

    def test_tensor_shapes(self, dpo_file, tokenizer):
        max_len = 64
        ds = DPODataset(dpo_file, tokenizer, max_length=max_len)
        item = ds[0]
        assert item["chosen_input_ids"].shape == (max_len,)
        assert item["rejected_input_ids"].shape == (max_len,)

    def test_chosen_rejected_different(self, dpo_file, tokenizer):
        ds = DPODataset(dpo_file, tokenizer, max_length=128)
        item = ds[0]
        # chosen and rejected have different content so ids should differ
        assert not torch.equal(item["chosen_input_ids"], item["rejected_input_ids"])


# ---------------------------------------------------------------------------
# DataCollator
# ---------------------------------------------------------------------------

class TestDataCollator:
    def test_basic_collation(self, pretrain_file, tokenizer):
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=32)
        collator = DataCollator(pad_token_id=0)
        batch = collator([ds[0], ds[1]])
        assert batch["input_ids"].shape == (2, 32)
        assert batch["attention_mask"].shape == (2, 32)
        assert batch["labels"].shape == (2, 32)

    def test_single_item(self, pretrain_file, tokenizer):
        ds = PretrainDataset(pretrain_file, tokenizer, max_length=32)
        collator = DataCollator()
        batch = collator([ds[0]])
        assert batch["input_ids"].shape[0] == 1


class TestDPODataCollator:
    def test_dpo_collation(self, dpo_file, tokenizer):
        ds = DPODataset(dpo_file, tokenizer, max_length=64)
        collator = DPODataCollator()
        batch = collator([ds[0], ds[1]])
        assert batch["chosen_input_ids"].shape == (2, 64)
        assert batch["rejected_input_ids"].shape == (2, 64)
        assert batch["chosen_attention_mask"].shape == (2, 64)
        assert batch["rejected_attention_mask"].shape == (2, 64)


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------

class TestDataPreprocessor:
    def test_clean_text(self):
        dp = DataPreprocessor()
        assert dp.clean_text("  hello   world  ") == "hello world"
        assert dp.clean_text("\n\tfoo\n\t") == "foo"
        assert dp.clean_text("") == ""

    def test_prepare_pretrain_data(self, tmp_path):
        input_file = tmp_path / "raw.txt"
        output_file = tmp_path / "clean.txt"
        input_file.write_text("  Short\n\n  This is a longer line that should be kept.\n", encoding="utf-8")

        dp = DataPreprocessor()
        dp.prepare_pretrain_data(str(input_file), str(output_file))

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        # "Short" has <= 10 chars, should be filtered
        assert len(lines) == 1
        assert "longer line" in lines[0]

    def test_prepare_sft_data(self, tmp_path):
        input_file = tmp_path / "raw.json"
        output_file = tmp_path / "sft.jsonl"
        data = [
            {"instruction": "Do something", "input": "", "output": "Done."},
            {"instruction": "", "input": "", "output": "No instruction"},  # skipped
        ]
        input_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        dp = DataPreprocessor()
        dp.prepare_sft_data(str(input_file), str(output_file), template="alpaca")

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        result = json.loads(lines[0])
        assert "prompt" in result
        assert "### 指令" in result["prompt"]

    def test_prepare_sft_data_raw_template(self, tmp_path):
        input_file = tmp_path / "raw.jsonl"
        output_file = tmp_path / "sft.jsonl"
        data = [{"instruction": "Test", "input": "some input", "output": "Result"}]
        input_file.write_text(
            "\n".join(json.dumps(d) for d in data), encoding="utf-8",
        )

        dp = DataPreprocessor()
        dp.prepare_sft_data(str(input_file), str(output_file), template="raw")

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        result = json.loads(lines[0])
        assert result["prompt"] == "Test"

    def test_prepare_dpo_data(self, tmp_path):
        input_file = tmp_path / "raw.json"
        output_file = tmp_path / "dpo.jsonl"
        data = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "answer": "missing fields"},  # skipped
        ]
        input_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        dp = DataPreprocessor()
        dp.prepare_dpo_data(str(input_file), str(output_file))

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        result = json.loads(lines[0])
        assert result["prompt"] == "Q1"

    def test_format_sft_with_input(self):
        dp = DataPreprocessor()
        item = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        result = dp._format_sft(item, "alpaca")
        assert result is not None
        assert "### 输入:" in result["prompt"]
        assert "Hello" in result["prompt"]

    def test_format_sft_without_input(self):
        dp = DataPreprocessor()
        item = {"instruction": "Summarize", "input": "", "output": "Summary"}
        result = dp._format_sft(item, "alpaca")
        assert result is not None
        assert "### 输入:" not in result["prompt"]

    def test_format_sft_missing_instruction(self):
        dp = DataPreprocessor()
        item = {"instruction": "", "input": "", "output": "Something"}
        result = dp._format_sft(item, "alpaca")
        assert result is None
