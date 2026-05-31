import json
import re
from pathlib import Path
from typing import Optional


class BPETokenizer:
    """Byte Pair Encoding tokenizer implemented from scratch."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        self._pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[一-鿿]|[a-zA-Z]+|[0-9]+|[^\s\w一-鿿]""")

    def _get_word_tokens(self, text: str) -> list[list[str]]:
        words = self._pattern.findall(text)
        return [[c for c in w] for w in words]

    def _get_stats(self, word_freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        pairs: dict[tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_pair(
        self, word_freqs: dict[tuple[str, ...], int], pair: tuple[str, str]
    ) -> dict[tuple[str, ...], int]:
        new_freqs = {}
        bigram = pair
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def train(self, texts: list[str], verbose: bool = False):
        word_freqs: dict[tuple[str, ...], int] = {}
        for text in texts:
            for word_tokens in self._get_word_tokens(text):
                key = tuple(word_tokens)
                word_freqs[key] = word_freqs.get(key, 0) + 1

        all_tokens = set()
        for word in word_freqs:
            for token in word:
                all_tokens.add(token)

        self.token_to_id = dict(self.special_tokens)
        idx = len(self.special_tokens)
        for token in sorted(all_tokens):
            self.token_to_id[token] = idx
            idx += 1

        num_merges = self.vocab_size - len(self.token_to_id)
        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            word_freqs = self._merge_pair(word_freqs, best_pair)
            merged = best_pair[0] + best_pair[1]
            if merged not in self.token_to_id:
                self.token_to_id[merged] = idx
                idx += 1

            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {merged}")

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = []
        if add_special:
            ids.append(self.special_tokens["<bos>"])

        for word_tokens in self._get_word_tokens(text):
            merged = self._apply_merges(word_tokens)
            for token in merged:
                ids.append(self.token_to_id.get(token, self.special_tokens["<unk>"]))

        if add_special:
            ids.append(self.special_tokens["<eos>"])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special_ids = set(self.special_tokens.values())
        tokens = []
        for id_ in ids:
            if skip_special and id_ in special_ids:
                continue
            tokens.append(self.id_to_token.get(id_, "<unk>"))
        return "".join(tokens)

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.special_tokens = data["special_tokens"]
        return tokenizer

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<eos>"]

    def __len__(self) -> int:
        return len(self.token_to_id)
