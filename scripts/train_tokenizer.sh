#!/usr/bin/env bash
# =============================================================================
# LLM-Forge: Train BPE Tokenizer
# =============================================================================
# This script trains a Byte-Pair Encoding (BPE) tokenizer on the provided
# text corpus. The trained tokenizer is saved to the specified output path
# and will be used by downstream training stages.
#
# Usage:
#   ./scripts/train_tokenizer.sh [OPTIONS]
#
# Options:
#   --data_path     Path to training text data (default: ./data/pretrain/train.txt)
#   --vocab_size    Vocabulary size for the tokenizer (default: 32000)
#   --output_path   Path to save the trained tokenizer (default: ./outputs/tokenizer/tokenizer.json)
#   --help          Show this help message
# =============================================================================

set -e  # Exit immediately on error

# ---- Default arguments ----
DATA_PATH="./data/pretrain/train.txt"
VOCAB_SIZE=32000
OUTPUT_PATH="./outputs/tokenizer/tokenizer.json"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --vocab_size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --help)
            head -20 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---- Activate conda environment ----
echo "========================================="
echo " LLM-Forge: Train BPE Tokenizer"
echo "========================================="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-forge
echo "[INFO] Conda environment 'llm-forge' activated"

# ---- Validate inputs ----
if [ ! -f "$DATA_PATH" ] && [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data path does not exist: $DATA_PATH"
    echo "[INFO]  Please provide a valid --data_path argument"
    exit 1
fi

# ---- Create output directory ----
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"
echo "[INFO] Output directory: $OUTPUT_DIR"

# ---- Run tokenizer training ----
echo "[INFO] Starting BPE tokenizer training..."
echo "[INFO]   Data path:   $DATA_PATH"
echo "[INFO]   Vocab size:  $VOCAB_SIZE"
echo "[INFO]   Output path: $OUTPUT_PATH"

python -m forge.tokenizer.train_tokenizer \
    --data_path "$DATA_PATH" \
    --vocab_size "$VOCAB_SIZE" \
    --output_path "$OUTPUT_PATH"

echo "========================================="
echo "[INFO] Tokenizer training completed!"
echo "[INFO] Tokenizer saved to: $OUTPUT_PATH"
echo "========================================="
