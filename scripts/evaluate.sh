#!/usr/bin/env bash
# =============================================================================
# LLM-Forge: Model Evaluation
# =============================================================================
# This script runs the evaluation harness on a trained model. It supports
# multiple evaluation tasks including text generation, question answering,
# and perplexity measurement.
#
# Usage:
#   ./scripts/evaluate.sh [OPTIONS]
#
# Options:
#   --model_path    Path to the trained model directory (default: ./outputs/sft/best_model)
#   --output_dir    Directory to save evaluation results (default: ./outputs/evaluation)
#   --tasks         Comma-separated evaluation tasks: generation,qa,perplexity (default: all)
#   --device        Device to run on: cuda or cpu (default: cuda)
#   --help          Show this help message
# =============================================================================

set -e  # Exit immediately on error

# ---- Default arguments ----
MODEL_PATH="./outputs/sft/best_model"
OUTPUT_DIR="./outputs/evaluation"
TASKS="generation,qa,perplexity"
DEVICE="cuda"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            head -24 "$0" | tail -20
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
echo " LLM-Forge: Model Evaluation"
echo "========================================="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-forge
echo "[INFO] Conda environment 'llm-forge' activated"

# ---- Validate model path ----
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path not found: $MODEL_PATH"
    echo "[INFO]  Please train a model first or provide --model_path"
    exit 1
fi
echo "[INFO] Model path:    $MODEL_PATH"
echo "[INFO] Output dir:    $OUTPUT_DIR"
echo "[INFO] Tasks:         $TASKS"
echo "[INFO] Device:        $DEVICE"

# ---- Create output directory ----
mkdir -p "$OUTPUT_DIR"

# ---- Run evaluation ----
echo "[INFO] Starting evaluation..."
python -c "
import sys
import os
import torch
sys.path.insert(0, '.')

from forge.model.transformer import GPTModel, GPTConfig
from forge.tokenizer.bpe_tokenizer import BPETokenizer
from forge.evaluation.evaluator import Evaluator
from forge.evaluation.harness import EvalHarness
from forge.data.sample_data import generate_sample_data

model_path = '$MODEL_PATH'
output_dir = '$OUTPUT_DIR'
tasks = '$TASKS'.split(',')
device = '$DEVICE'

# Load tokenizer
tokenizer_path = os.path.join(model_path, 'tokenizer.json')
if not os.path.exists(tokenizer_path):
    tokenizer_path = './outputs/tokenizer/tokenizer.json'
print(f'[INFO] Loading tokenizer from: {tokenizer_path}')
tokenizer = BPETokenizer.load(tokenizer_path)

# Load model
print(f'[INFO] Loading model from: {model_path}')
model = GPTModel(GPTConfig())
model_path_pt = os.path.join(model_path, 'model.pt')
if os.path.exists(model_path_pt):
    model.load_state_dict(torch.load(model_path_pt, map_location='cpu'))
    print(f'[INFO] Model loaded successfully')
else:
    print(f'[WARN] Model file not found, using random weights')

# Initialize evaluator
evaluator = Evaluator(output_dir=output_dir)

# Prepare test data
test_data = {}

if 'generation' in tasks or 'qa' in tasks:
    # Generate sample test data if needed
    sample_data_path = './data/pretrain/eval.txt'
    if not os.path.exists(sample_data_path):
        print('[INFO] Generating sample test data...')
        generate_sample_data()

    if 'generation' in tasks:
        test_data['generation'] = [
            {'prompt': 'The future of artificial intelligence is', 'reference': 'The future of artificial intelligence is bright and full of possibilities.'},
            {'prompt': 'Machine learning algorithms can be categorized into', 'reference': 'Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning.'},
            {'prompt': 'Natural language processing enables computers to', 'reference': 'Natural language processing enables computers to understand and generate human language.'},
            {'prompt': 'Deep learning has revolutionized the field of', 'reference': 'Deep learning has revolutionized the field of computer vision and natural language processing.'},
            {'prompt': 'The transformer architecture introduced', 'reference': 'The transformer architecture introduced self-attention mechanisms for sequence modeling.'},
        ]

    if 'qa' in tasks:
        test_data['qa'] = [
            {'prompt': 'Question: What is deep learning? Answer:', 'reference': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers.'},
            {'prompt': 'Question: What is a transformer model? Answer:', 'reference': 'A transformer model is an architecture based on self-attention mechanisms.'},
            {'prompt': 'Question: What is backpropagation? Answer:', 'reference': 'Backpropagation is an algorithm for training neural networks by computing gradients.'},
        ]

# Run evaluation
results = evaluator.full_evaluation(
    model=model,
    tokenizer=tokenizer,
    test_data=test_data if test_data else None,
    device=device,
)

# Print summary
print()
print('=' * 60)
print(' EVALUATION RESULTS')
print('=' * 60)
for task_name, task_results in results.items():
    print(f'\n  Task: {task_name}')
    if isinstance(task_results, dict):
        for k, v in task_results.items():
            if isinstance(v, float):
                print(f'    {k}: {v:.4f}')
            elif isinstance(v, (int, str)):
                print(f'    {k}: {v}')
print()
print('=' * 60)
print(f' Results saved to: $OUTPUT_DIR')
print('=' * 60)
"

echo "========================================="
echo "[INFO] Evaluation completed!"
echo "[INFO] Results saved to: $OUTPUT_DIR"
echo "========================================="
