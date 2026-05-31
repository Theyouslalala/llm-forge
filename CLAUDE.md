# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-Forge is a full-stack LLM system built from scratch in PyTorch. It implements the complete LLM lifecycle: BPE tokenizer → GPT pretraining → SFT (LoRA) → DPO alignment → RAG → Agent → Multimodal → Evaluation. The model uses LLaMA-style architecture (RoPE, SwiGLU, RMSNorm, GQA). All code is in `forge/` package, configs in `configs/` YAML files.

## Commands

```bash
# Install (editable, with all extras)
pip install -e ".[rag,eval,multimodal,dev]"

# Run all tests (CPU-only, no GPU needed)
pytest tests/

# Run single test file
pytest tests/test_model.py

# Run single test class/method
pytest tests/test_model.py::TestGPTModel::test_forward_pass

# Generate sample data
python -c "from forge.data.sample_data import generate_sample_data; generate_sample_data()"

# Training pipeline (each step accepts config path as argument)
python -m forge.tokenizer.train_tokenizer --data_path data/pretrain/train.txt --vocab_size 32000
python -m forge.training.pretrain configs/pretrain.yaml
python -m forge.training.sft configs/sft.yaml
python -m forge.training.dpo configs/dpo.yaml

# Code formatting
black forge/ tests/
isort forge/ tests/
```

## Architecture

### Module dependency flow
```
model/transformer  ←── training/{pretrain,sft,dpo}
       ↑                      ↑
model/lora          data/dataset + data/preprocessor
model/embeddings    tokenizer/bpe_tokenizer
model/attention
model/feedforward
```

- `forge.model.transformer.GPTModel` is the central class. All training, evaluation, and agent modules depend on it.
- `forge.model.lora.apply_lora_to_model()` mutates a GPTModel in-place, replacing target Linear layers with LoRA wrappers.
- Training modules (`pretrain.py`, `sft.py`, `dpo.py`) are standalone entry points run via `python -m forge.training.<name>`. Each reads a YAML config, constructs model/data/trainer, and runs.
- `forge.training.trainer.Trainer` is the universal trainer. `DPOTrainer` extends it with DPO-specific loss. The trainer handles fp16, gradient accumulation, checkpointing, and periodic eval.
- RAG (`forge.rag`) and Agent (`forge.agent`) are independent subsystems that can use any trained GPTModel.

### Key model design decisions
- **GPTConfig** (`transformer.py`): dataclass with `from_dict()` for YAML loading. `n_kv_heads` enables GQA when set < `n_heads`.
- **RoPE**: cos/sin cache shape is `(max_seq_len, head_dim)` — duplicated from `(max_seq_len, head_dim/2)` via `torch.cat` in `embeddings.py`.
- **Tied weights**: `lm_head.weight = token_embedding.embedding.weight` when `tie_weights=True`.
- **generate()**: default `eos_token_id=3` (matches BPE tokenizer's `<eos>`).

### Configuration system
YAML configs in `configs/` with sections: `model:`, `tokenizer:`, `training:`, `data:`. Loaded via `yaml.safe_load()`. `GPTConfig.from_dict()` filters unknown keys automatically.

### Test structure
All tests use `sys.path.insert` for imports (no conftest.py). Tests create small models (d_model=32, n_layers=2) for speed. RAG tests use numpy fallback (no FAISS). Agent tests use mock/stub objects.

## Key files

| File | Role |
|------|------|
| `forge/model/transformer.py` | GPTModel, GPTConfig, TransformerBlock — the core model |
| `forge/model/lora.py` | LoRALinear, apply_lora_to_model, get_lora_parameters |
| `forge/training/trainer.py` | Universal Trainer with fp16/grad accum/checkpointing |
| `forge/training/dpo.py` | DPOTrainer extending Trainer with preference loss |
| `forge/tokenizer/bpe_tokenizer.py` | BPETokenizer: train, encode, decode, save/load |
| `forge/rag/rag_pipeline.py` | RAGPipeline orchestrating full RAG flow |
| `forge/agent/react_agent.py` | ReActAgent with Thought-Action-Observation loop |
| `forge/evaluation/metrics.py` | BLEU, ROUGE-L, exact match, F1, perplexity |
| `forge/data/sample_data.py` | generate_sample_data() for quick demo setup |
| `configs/*.yaml` | Training/model configuration |
