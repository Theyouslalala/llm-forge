#!/usr/bin/env bash
# =============================================================================
# LLM-Forge: Supervised Fine-Tuning (SFT)
# =============================================================================
# This script runs supervised fine-tuning on a pretrained model using
# instruction-response pairs. Supports LoRA for parameter-efficient training.
#
# Usage:
#   ./scripts/sft.sh [OPTIONS]
#
# Options:
#   --config    Path to SFT config YAML (default: configs/sft.yaml)
#   --help      Show this help message
# =============================================================================

set -e  # Exit immediately on error

# ---- Default arguments ----
CONFIG="configs/sft.yaml"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
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
echo " LLM-Forge: Supervised Fine-Tuning (SFT)"
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

# ---- Run SFT training ----
echo "[INFO] Starting SFT fine-tuning..."
python -m forge.training.sft "$CONFIG"

echo "========================================="
echo "[INFO] SFT fine-tuning completed!"
echo "[INFO] Check outputs in ./outputs/sft/"
echo "========================================="
