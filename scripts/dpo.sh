#!/usr/bin/env bash
# =============================================================================
# LLM-Forge: Direct Preference Optimization (DPO) Alignment
# =============================================================================
# This script runs DPO alignment training on an SFT model. DPO uses
# preference pairs (chosen/rejected) to align the model with human preferences
# without requiring a separate reward model.
#
# Usage:
#   ./scripts/dpo.sh [OPTIONS]
#
# Options:
#   --config    Path to DPO config YAML (default: configs/dpo.yaml)
#   --help      Show this help message
# =============================================================================

set -e  # Exit immediately on error

# ---- Default arguments ----
CONFIG="configs/dpo.yaml"

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
echo " LLM-Forge: DPO Alignment"
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

# ---- Run DPO training ----
echo "[INFO] Starting DPO alignment training..."
python -m forge.training.dpo "$CONFIG"

echo "========================================="
echo "[INFO] DPO alignment completed!"
echo "[INFO] Check outputs in ./outputs/dpo/"
echo "========================================="
