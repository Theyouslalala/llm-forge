#!/usr/bin/env bash
# =============================================================================
# LLM-Forge: RAG Pipeline Demo
# =============================================================================
# This script runs the Retrieval-Augmented Generation (RAG) pipeline demo.
# It loads documents, builds a vector index, and answers queries using
# retrieved context augmented with LLM generation.
#
# Usage:
#   ./scripts/run_rag.sh [OPTIONS]
#
# Options:
#   --config        Path to RAG config YAML (default: configs/rag.yaml)
#   --docs_path     Path to documents to ingest (default: ./data/rag/)
#   --query         Question to ask the RAG pipeline (default: interactive mode)
#   --ingest_only   Only ingest documents, skip querying (flag)
#   --help          Show this help message
# =============================================================================

set -e  # Exit immediately on error

# ---- Default arguments ----
CONFIG="configs/rag.yaml"
DOCS_PATH="./data/rag/"
QUERY=""
INGEST_ONLY=false

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --docs_path)
            DOCS_PATH="$2"
            shift 2
            ;;
        --query)
            QUERY="$2"
            shift 2
            ;;
        --ingest_only)
            INGEST_ONLY=true
            shift
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
echo " LLM-Forge: RAG Pipeline Demo"
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

# ---- Run RAG pipeline ----
echo "[INFO] Starting RAG pipeline..."
echo "[INFO] Documents path: $DOCS_PATH"

python -c "
import sys
sys.path.insert(0, '.')
from forge.rag.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(config_path='$CONFIG')

# Ingest documents
print('[INFO] Ingesting documents...')
pipeline.ingest('$DOCS_PATH')
pipeline.save()
print('[INFO] Document ingestion completed!')

# Query mode
ingest_only = $( [ "$INGEST_ONLY" = true ] && echo "True" || echo "False" )
if not ingest_only:
    query = '$QUERY'
    if not query:
        # Interactive mode
        print()
        print('=' * 50)
        print(' RAG Interactive Query Mode')
        print(' Type your question, or \"quit\" to exit')
        print('=' * 50)
        while True:
            try:
                query = input('\nQuestion: ').strip()
                if query.lower() in ('quit', 'exit', 'q'):
                    break
                if not query:
                    continue
                result = pipeline.query(query)
                print(f'\nAnswer: {result[\"answer\"]}')
                print(f'\nSources ({len(result[\"sources\"])} documents retrieved)')
            except (EOFError, KeyboardInterrupt):
                break
        print('\n[INFO] RAG demo finished.')
    else:
        result = pipeline.query(query)
        print(f'\nQuestion: {result[\"question\"]}')
        print(f'Answer: {result[\"answer\"]}')
        print(f'\nSources ({len(result[\"sources\"])} documents retrieved)')
"

echo "========================================="
echo "[INFO] RAG pipeline demo completed!"
echo "========================================="
