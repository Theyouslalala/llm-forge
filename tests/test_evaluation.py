"""Tests for evaluation metrics: BLEU, ROUGE-L, exact match, F1, perplexity."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.evaluation.metrics import (
    compute_bleu,
    compute_rouge_l,
    compute_exact_match,
    compute_f1,
    compute_perplexity,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# compute_perplexity
# ---------------------------------------------------------------------------

class TestPerplexity:
    def test_zero_loss(self):
        assert compute_perplexity(0.0) == 1.0

    def test_positive_loss(self):
        ppl = compute_perplexity(1.0)
        assert abs(ppl - math.e) < 1e-6

    def test_large_loss_capped(self):
        # Loss > 20 is capped to prevent overflow
        ppl = compute_perplexity(100.0)
        assert ppl == math.exp(20)

    def test_negative_loss(self):
        # Negative loss should still work (exp of negative is < 1)
        ppl = compute_perplexity(-1.0)
        assert ppl < 1.0

    def test_typical_loss(self):
        ppl = compute_perplexity(3.0)
        assert abs(ppl - math.exp(3.0)) < 1e-6


# ---------------------------------------------------------------------------
# compute_bleu
# ---------------------------------------------------------------------------

class TestBLEU:
    def test_perfect_match(self):
        score = compute_bleu("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_completely_wrong(self):
        score = compute_bleu("abc def", "xyz uvw")
        assert score == 0.0

    def test_empty_hypothesis(self):
        score = compute_bleu("hello", "")
        assert score == 0.0

    def test_partial_match(self):
        score = compute_bleu("the cat sat on the mat", "the cat is on the mat")
        assert 0.0 < score < 1.0

    def test_single_char_match(self):
        score = compute_bleu("a", "a")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_chinese_text(self):
        score = compute_bleu("机器学习是人工智能", "机器学习是人工智能")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_bleu_with_ngrams(self):
        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quick brown fox jumps over the lazy cat"
        score = compute_bleu(ref, hyp, max_n=4)
        assert score > 0.8  # Very similar

    def test_ordering_matters(self):
        score1 = compute_bleu("a b c", "a b c")
        score2 = compute_bleu("a b c", "c b a")
        assert score1 > score2

    def test_brevity_penalty(self):
        # Short hypothesis vs long reference should penalize
        score = compute_bleu("the quick brown fox jumps over the lazy dog", "the")
        assert score < 0.5


# ---------------------------------------------------------------------------
# compute_rouge_l
# ---------------------------------------------------------------------------

class TestROUGEL:
    def test_perfect_match(self):
        result = compute_rouge_l("hello world", "hello world")
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_completely_wrong(self):
        result = compute_rouge_l("abc", "xyz")
        assert result["f1"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        result = compute_rouge_l("the cat sat", "the dog sat")
        assert result["f1"] > 0.5  # "the" and "sat" match

    def test_subset_hypothesis(self):
        result = compute_rouge_l("a b c d e", "b c")
        # LCS is "b c" (2), precision = 2/2 = 1.0, recall = 2/5 = 0.4
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.4)

    def test_empty_strings(self):
        result = compute_rouge_l("", "")
        assert result["f1"] == pytest.approx(0.0)

    def test_chinese_text(self):
        result = compute_rouge_l("机器学习", "机器学习")
        assert result["f1"] == pytest.approx(1.0)

    def test_f1_is_harmonic_mean(self):
        result = compute_rouge_l("a b c", "a b d")
        p, r, f1 = result["precision"], result["recall"], result["f1"]
        expected_f1 = 2 * p * r / (p + r + 1e-8)
        assert f1 == pytest.approx(expected_f1, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_match(self):
        assert compute_exact_match("hello", "hello") is True

    def test_case_insensitive(self):
        assert compute_exact_match("Hello", "hello") is True
        assert compute_exact_match("HELLO", "hello") is True

    def test_whitespace_stripped(self):
        assert compute_exact_match("  hello  ", "hello") is True

    def test_no_match(self):
        assert compute_exact_match("hello", "world") is False

    def test_empty_strings(self):
        assert compute_exact_match("", "") is True

    def test_partial_not_match(self):
        assert compute_exact_match("hello world", "hello") is False


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------

class TestF1:
    def test_perfect_match(self):
        score = compute_f1("hello world", "hello world")
        assert score == pytest.approx(1.0)

    def test_no_overlap(self):
        score = compute_f1("abc def", "xyz uvw")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = compute_f1("the cat sat", "the dog sat")
        # "the" and "sat" overlap: precision = 2/3, recall = 2/3
        assert 0.5 < score < 1.0

    def test_empty_reference(self):
        score = compute_f1("", "hello")
        assert score == 0.0

    def test_empty_hypothesis(self):
        score = compute_f1("hello", "")
        assert score == 0.0

    def test_subset(self):
        score = compute_f1("a b c d", "a b")
        # precision = 2/2 = 1.0, recall = 2/4 = 0.5
        expected = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert score == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_metrics (batch)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_all_metrics(self):
        refs = ["hello world", "foo bar"]
        preds = ["hello world", "foo baz"]
        results = compute_metrics(preds, refs)
        assert "bleu" in results
        assert "rouge_l" in results
        assert "exact_match" in results
        assert "f1" in results

    def test_specific_metrics(self):
        refs = ["test"]
        preds = ["test"]
        results = compute_metrics(preds, refs, metrics=["bleu"])
        assert "bleu" in results
        assert "rouge_l" not in results

    def test_single_metric(self):
        refs = ["a b c"]
        preds = ["a b c"]
        results = compute_metrics(preds, refs, metrics=["exact_match"])
        assert results["exact_match"] == 1.0

    def test_empty_predictions(self):
        refs = ["hello"]
        preds = [""]
        results = compute_metrics(preds, refs, metrics=["exact_match", "f1"])
        assert results["exact_match"] == 0.0
        assert results["f1"] == 0.0

    def test_multiple_samples_averaged(self):
        refs = ["a", "b", "c"]
        preds = ["a", "b", "x"]
        results = compute_metrics(preds, refs, metrics=["exact_match"])
        assert results["exact_match"] == pytest.approx(2.0 / 3.0)

    def test_perfect_scores(self):
        refs = ["the cat sat on the mat"]
        preds = ["the cat sat on the mat"]
        results = compute_metrics(preds, refs)
        assert results["exact_match"] == 1.0
        assert results["rouge_l"] == pytest.approx(1.0, abs=1e-4)
        assert results["f1"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# EvalHarness (unit-level, no real model)
# ---------------------------------------------------------------------------

class TestEvalHarness:
    def test_summary_format(self):
        from forge.evaluation.harness import EvalHarness
        harness = EvalHarness(output_dir="./test_outputs_tmp")
        harness.results["test_task"] = {
            "metrics": {"bleu": 0.85, "rouge_l": 0.72},
            "num_samples": 10,
        }
        summary = harness.summary()
        assert "test_task" in summary
        assert "0.85" in summary
        assert "0.72" in summary

    def test_save_results(self, tmp_path):
        from forge.evaluation.harness import EvalHarness
        output_dir = str(tmp_path / "eval_output")
        harness = EvalHarness(output_dir=output_dir)
        harness.results["task1"] = {
            "metrics": {"bleu": 0.9, "f1": 0.8},
            "predictions": ["pred1"],
            "references": ["ref1"],
            "num_samples": 1,
        }
        harness.save_results("test_results.json")

        import json
        with open(os.path.join(output_dir, "test_results.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert "task1" in data
        assert data["task1"]["metrics"]["bleu"] == 0.9
