#!/usr/bin/env bash
# =============================================================================
# LLM-Forge: Pretraining
# =============================================================================
# This script runs the GPT-style causal language model pretraining pipeline.
# It loads the configuration from a YAML file, trains a transformer model
# on a text corpus, and saves the pretrained model checkpoint.
#
# If no tokenizer is found, it will automatically train one first.
#
# Usage:
#   ./scripts/pretrain.sh [OPTIONS]
#
# Options:
#   --config    Path to pretrain config YAML (default: configs/pretrain.yaml)
#   --help      Show this help message
# =============================================================================

set -e  # Exit immediately on error

# ---- Default arguments ----
CONFIG="configs/pretrain.yaml"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            head -22 "$0" | tail -18
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
echo " LLM-Forge: Pretraining"
echo "========================================="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-forge
echo "[INFO] Conda environment 'llm-forge' activated"

# ---- Validate config ----
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
fi
echo "[INFO] Using config: $CONFIG"

# ---- Check for GPU ----
python -c "import torch; print(f'[INFO] CUDA available: {torch.cuda.is_available()}'); print(f'[INFO] GPU count: {torch.cuda.device_count()}')" 2>/dev/null || true

# ---- Run pretraining ----
echo "[INFO] Starting pretraining..."
python -m forge.training.pretrain "$CONFIG"

echo "========================================="
echo "[INFO] Pretraining completed!"
echo "[INFO] Check outputs in ./outputs/pretrain/"
echo "========================================="
