import argparse
from pathlib import Path

from .bpe_tokenizer import BPETokenizer
from ..utils.logger import get_logger

logger = get_logger(__name__)


def train_tokenizer(
    data_path: str,
    vocab_size: int = 32000,
    output_path: str = "./outputs/tokenizer/tokenizer.json",
    file_pattern: str = "*.txt",
) -> BPETokenizer:
    logger.info(f"Loading training data from {data_path}")

    texts = []
    data_dir = Path(data_path)
    if data_dir.is_file():
        with open(data_dir, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        for file_path in data_dir.glob(file_pattern):
            with open(file_path, encoding="utf-8") as f:
                texts.extend(line.strip() for line in f if line.strip())

    logger.info(f"Loaded {len(texts)} text samples")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}")
    tokenizer.train(texts, verbose=True)

    tokenizer.save(output_path)
    logger.info(f"Tokenizer saved to {output_path}")
    logger.info(f"Final vocab size: {len(tokenizer)}")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--output_path", type=str, default="./outputs/tokenizer/tokenizer.json")
    args = parser.parse_args()

    train_tokenizer(args.data_path, args.vocab_size, args.output_path)
