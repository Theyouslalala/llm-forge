#!/usr/bin/env bash
# =============================================================================
# LLM-Forge: Agent Demo
# =============================================================================
# This script runs the ReAct (Reasoning + Acting) agent demo. The agent can
# use tools (calculator, code executor, file reader, web search) to answer
# complex queries through multi-step reasoning.
#
# Usage:
#   ./scripts/run_agent.sh [OPTIONS]
#
# Options:
#   --model_path    Path to the trained model directory (default: ./outputs/sft/best_model)
#   --query         Query for the agent (default: interactive mode)
#   --max_turns     Maximum reasoning turns (default: 5)
#   --device        Device to run on: cuda or cpu (default: cuda)
#   --help          Show this help message
# =============================================================================

set -e  # Exit immediately on error

# ---- Default arguments ----
MODEL_PATH="./outputs/sft/best_model"
QUERY=""
MAX_TURNS=5
DEVICE="cuda"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --query)
            QUERY="$2"
            shift 2
            ;;
        --max_turns)
            MAX_TURNS="$2"
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
echo " LLM-Forge: ReAct Agent Demo"
echo "========================================="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-forge
echo "[INFO] Conda environment 'llm-forge' activated"

# ---- Validate model path ----
if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARN] Model path not found: $MODEL_PATH"
    echo "[INFO]  Agent will run without LLM backbone (template responses only)"
fi

echo "[INFO] Model path: $MODEL_PATH"
echo "[INFO] Device:      $DEVICE"
echo "[INFO] Max turns:    $MAX_TURNS"

# ---- Run agent demo ----
echo "[INFO] Starting ReAct agent..."
echo ""

python -c "
import sys
sys.path.insert(0, '.')
from forge.agent.react_agent import ReActAgent
from forge.agent.tools.calculator import CalculatorTool
from forge.agent.tools.code_executor import CodeExecutorTool
from forge.agent.tools.file_reader import FileReaderTool

# Initialize agent
agent = ReActAgent(
    model_path='$MODEL_PATH',
    device='$DEVICE',
    max_turns=$MAX_TURNS,
)

# Register tools
agent.register_tool(CalculatorTool())
agent.register_tool(CodeExecutorTool())
agent.register_tool(FileReaderTool())

query = '$QUERY'
if not query:
    # Interactive mode
    print('=' * 50)
    print(' ReAct Agent Interactive Mode')
    print(' Available tools: calculator, code_executor, file_reader')
    print(' Type your question, or \"quit\" to exit')
    print('=' * 50)
    while True:
        try:
            query = input('\nQuery: ').strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if not query:
                continue
            answer = agent.run(query)
            print(f'\nAnswer: {answer}')
        except (EOFError, KeyboardInterrupt):
            break
    print('\n[INFO] Agent demo finished.')
else:
    answer = agent.run(query)
    print(f'Query:  {query}')
    print(f'Answer: {answer}')
"

echo "========================================="
echo "[INFO] Agent demo completed!"
echo "========================================="
