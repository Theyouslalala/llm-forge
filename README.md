# LLM-Forge: 全栈大语言模型系统

[English](#english) | [中文](#中文)

---

## 中文

### 项目简介

**LLM-Forge** 是一个从零构建的全栈大语言模型系统，覆盖当前LLM领域的主流技术栈。项目以PyTorch为基础，不依赖高级封装框架，完整实现了从Tokenizer训练、模型预训练、监督微调(SFT)、DPO对齐，到RAG检索增强、Agent智能体、多模态理解和评估体系的全流程。

### 核心特性

| 模块 | 技术要点 |
|------|---------|
| **Tokenizer** | BPE算法从零实现，支持中文分词，自定义词表训练 |
| **模型架构** | GPT Transformer (RoPE + SwiGLU + RMSNorm)，支持GQA |
| **LoRA微调** | 低秩适配实现参数高效微调，支持merge/unmerge |
| **预训练** | 因果语言建模，混合精度(fp16/bf16)，梯度累积，cosine LR |
| **SFT微调** | 指令微调，Alpaca格式数据，LoRA/QLoRA支持 |
| **DPO对齐** | Direct Preference Optimization，无需奖励模型 |
| **RAG系统** | 文档加载→文本切分→向量嵌入→FAISS检索→增强生成 |
| **Agent框架** | ReAct推理 + Function Calling + 任务规划 + 记忆系统 |
| **多模态** | ViT视觉编码器 + Cross-Attention图文融合 |
| **评估体系** | 统一Harness框架，PPL/BLEU/ROUGE/EM/F1多维指标 |

### 项目结构

```
llm-forge/
├── configs/                    # 训练配置文件 (YAML)
├── forge/                      # 核心代码包
│   ├── tokenizer/              # BPE分词器
│   ├── model/                  # 模型架构 (GPT + ViT + LoRA + Multimodal)
│   ├── training/               # 训练流水线 (Pretrain + SFT + DPO)
│   ├── data/                   # 数据处理
│   ├── rag/                    # RAG检索增强系统
│   ├── agent/                  # Agent智能体框架
│   │   └── tools/              # 工具集 (计算器/搜索/代码执行/文件读取)
│   ├── evaluation/             # 评估体系
│   │   └── benchmarks/         # 基准测试
│   └── utils/                  # 工具函数
├── scripts/                    # 运行脚本
├── notebooks/                  # Jupyter演示
└── tests/                      # 单元测试
```

### 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    LLM-Forge 架构                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ ViT视觉   │  │ BPE      │  │ 文档加载器        │  │
│  │ 编码器    │  │ Tokenizer│  │ 文本切分器        │  │
│  └────┬─────┘  └────┬─────┘  │ 向量嵌入器        │  │
│       │              │        │ FAISS向量库        │  │
│       ▼              ▼        └────────┬─────────┘  │
│  ┌─────────────────────────┐           │            │
│  │   Multimodal GPT Model  │           ▼            │
│  │  ┌───────────────────┐  │  ┌──────────────────┐  │
│  │  │ RoPE + Multi-Head  │  │  │   RAG Pipeline   │  │
│  │  │ Attention          │  │  │   检索增强生成     │  │
│  │  ├───────────────────┤  │  └──────────────────┘  │
│  │  │ SwiGLU FFN         │  │                        │
│  │  ├───────────────────┤  │  ┌──────────────────┐  │
│  │  │ RMSNorm            │  │  │  Agent Framework  │  │
│  │  ├───────────────────┤  │  │  ReAct推理        │  │
│  │  │ LoRA适配层         │  │  │  工具调用         │  │
│  │  └───────────────────┘  │  │  任务规划         │  │
│  └─────────────────────────┘  │  记忆系统         │  │
│                                └──────────────────┘  │
│  ┌──────────────────────────────────────────────┐    │
│  │           Training Pipeline                   │    │
│  │  Pretrain → SFT → DPO → Evaluation           │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 快速开始

#### 环境要求

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (GPU训练)

#### 安装

```bash
# 创建conda环境
conda create -n llm-forge python=3.10 -y
conda activate llm-forge

# 安装依赖
pip install -e .

# 安装可选依赖
pip install -e ".[rag]"    # RAG功能
pip install -e ".[eval]"   # 评估功能
pip install -e ".[multimodal]"  # 多模态功能
```

#### 训练流程

```bash
# 1. 生成示例数据
python -c "from forge.data.sample_data import generate_sample_data; generate_sample_data()"

# 2. 训练Tokenizer
bash scripts/train_tokenizer.sh

# 3. 预训练
bash scripts/pretrain.sh

# 4. SFT微调
bash scripts/sft.sh

# 5. DPO对齐
bash scripts/dpo.sh

# 6. 评估
bash scripts/evaluate.sh
```

#### RAG使用

```bash
# 运行RAG演示
bash scripts/run_rag.sh
```

#### Agent使用

```bash
# 运行Agent演示
bash scripts/run_agent.sh
```

### 模型规模

| 配置 | 参数量 | 显存需求 | 适用场景 |
|------|--------|---------|---------|
| Small (默认) | ~85M | ~2GB | 快速实验、CPU调试 |
| Medium | ~250M | ~6GB | 单卡训练、效果验证 |
| Large | ~500M | ~12GB | 完整训练、效果最佳 |

### 关键技术细节

#### 1. RoPE旋转位置编码
相比传统绝对位置编码，RoPE通过旋转矩阵编码相对位置信息，支持任意长度泛化。

#### 2. SwiGLU激活函数
LLaMA风格的门控激活单元，`SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)`，比GELU效果更好。

#### 3. RMSNorm归一化
相比LayerNorm，RMSNorm省略了均值中心化，计算更快且效果相当。

#### 4. LoRA低秩适配
冻结原始权重，只训练低秩矩阵 `ΔW = BA`，参数量减少99%+。

#### 5. DPO对齐
直接利用偏好数据优化策略，无需训练奖励模型：`L = -log(σ(β(log π(y_w) - log π(y_l))))`

---

## English

### Overview

**LLM-Forge** is a full-stack Large Language Model system built from scratch, covering the complete LLM technology stack. The project implements everything from Tokenizer training, model pretraining, supervised fine-tuning (SFT), DPO alignment, to RAG, Agent, Multimodal, and evaluation framework.

### Key Features

- **From-scratch Transformer**: GPT architecture with RoPE, SwiGLU, RMSNorm (LLaMA-style)
- **Complete Training Pipeline**: BPE Tokenizer → Pretraining → SFT → DPO
- **RAG System**: Document loading → Text splitting → Vector embedding → FAISS retrieval → Augmented generation
- **Agent Framework**: ReAct reasoning + Function Calling + Task planning + Memory system
- **Multimodal**: ViT vision encoder + Cross-Attention fusion
- **Evaluation**: Unified harness with PPL/BLEU/ROUGE/EM/F1 metrics

### Quick Start

```bash
conda create -n llm-forge python=3.10 -y
conda activate llm-forge
pip install -e .
python -c "from forge.data.sample_data import generate_sample_data; generate_sample_data()"
bash scripts/pretrain.sh
```

### License

MIT License - see [LICENSE](LICENSE)

### Acknowledgements

- Transformer architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- RoPE: [RoFormer](https://arxiv.org/abs/2104.09864)
- LoRA: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- DPO: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- ReAct: [Reasoning and Acting](https://arxiv.org/abs/2210.03629)
