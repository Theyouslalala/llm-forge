import math
import re
from collections import Counter
from typing import Optional

import numpy as np


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if not hyp_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        clipped = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
        precision = clipped / sum(hyp_ngrams.values())
        precisions.append(precision)

    if all(p > 0 for p in precisions):
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
        return bp * math.exp(log_avg)
    return 0.0


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_rouge_l(reference: str, hypothesis: str) -> dict:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_len / max(len(hyp_tokens), 1)
    recall = lcs_len / max(len(ref_tokens), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    if m > 500 or n > 500:
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if seq1[i-1] == seq2[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = temp
        return dp[n]

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_exact_match(reference: str, hypothesis: str) -> bool:
    return reference.strip().lower() == hypothesis.strip().lower()


def _normalize_text(text: str) -> list[str]:
    """Normalize text for F1: lowercase and strip punctuation."""
    import string
    return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]


def compute_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = set(_normalize_text(reference))
    hyp_tokens = set(_normalize_text(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return 0.0

    common = ref_tokens & hyp_tokens
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def compute_metrics(
    predictions: list[str],
    references: list[str],
    metrics: Optional[list[str]] = None,
) -> dict:
    if metrics is None:
        metrics = ["bleu", "rouge_l", "exact_match", "f1"]

    results = {}
    for metric in metrics:
        if metric == "bleu":
            scores = [compute_bleu(r, p) for r, p in zip(references, predictions)]
            results["bleu"] = np.mean(scores)
        elif metric == "rouge_l":
            scores = [compute_rouge_l(r, p)["f1"] for r, p in zip(references, predictions)]
            results["rouge_l"] = np.mean(scores)
        elif metric == "exact_match":
            scores = [compute_exact_match(r, p) for r, p in zip(references, predictions)]
            results["exact_match"] = np.mean(scores)
        elif metric == "f1":
            scores = [compute_f1(r, p) for r, p in zip(references, predictions)]
            results["f1"] = np.mean(scores)

    return results
