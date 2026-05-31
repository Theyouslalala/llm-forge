# LLM-Forge 深度优化指南与技术学习手册

> 本文档基于 LLM-Forge 项目的实际代码，系统性分析每个模块的优化空间，深入讲解底层原理，并提供面试级别的技术问答。适用于项目复盘、技术面试准备和后续迭代优化。

---

## 目录

- [第1章 模型架构优化](#第1章-模型架构优化)
- [第2章 训练效率优化](#第2章-训练效率优化)
- [第3章 Tokenizer 优化](#第3章-tokenizer-优化)
- [第4章 RAG 系统优化](#第4章-rag-系统优化)
- [第5章 Agent 系统优化](#第5章-agent-系统优化)
- [第6章 评估体系优化](#第6章-评估体系优化)
- [第7章 数据流水线优化](#第7章-数据流水线优化)
- [第8章 工程实践与面试高频问题](#第8章-工程实践与面试高频问题)
- [附录 优化优先级矩阵](#附录-优化优先级矩阵)

---

## 第1章 模型架构优化

### 1.1 Flash Attention / scaled_dot_product_attention

**当前实现分析**

`forge/model/attention.py:82-96` 手动实现 attention 计算：

```python
# 当前代码
scale = math.sqrt(self.head_dim)
attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
# ... causal mask + softmax ...
attn_weights = self.attn_dropout(attn_weights)
out = torch.matmul(attn_weights, v)
```

这段代码需要显式构造 `(batch, heads, seq_len, seq_len)` 的 attention matrix，显存复杂度为 O(n²)。对于 seq_len=2048、n_heads=12、batch_size=8 的场景，仅 attention matrix 就占用 `8 * 12 * 2048 * 2048 * 4 bytes ≈ 1.5 GB`。

**优化方案：使用 PyTorch 原生 SDPA**

PyTorch 2.0+ 提供 `torch.nn.functional.scaled_dot_product_attention`，自动选择最优后端：

```python
import torch.nn.functional as F

def forward(self, x, attention_mask=None, use_cache=False, past_kv=None):
    # ... q, k, v projection and RoPE (unchanged) ...

    # GQA: repeat k, v before SDPA
    k = self._repeat_kv(k)
    v = self._repeat_kv(v)

    # 构建 causal mask
    causal_mask = torch.triu(
        torch.full((seq_len, kv_len), float("-inf"), device=x.device), diagonal=1
    )

    # SDPA: 自动选择 Flash Attention / Memory-Efficient / Math backend
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=causal_mask,
        dropout_p=self.attn_dropout.p if self.training else 0.0,
        is_causal=False,  # 已手动构建 mask
    )

    out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    out = self.o_proj(out)
    return out, present_kv
```

**原理深度讲解**

Flash Attention 的核心思想是 **IO-aware 算法**。标准 attention 的瓶颈不是计算量，而是 HBM（高带宽显存）访问：

| 操作 | 标准 Attention | Flash Attention |
|------|---------------|-----------------|
| 显存复杂度 | O(n²) | O(n) |
| HBM 读写次数 | O(n²d + n²) | O(n²d² / M) |
| 需要物化 S 矩阵 | 是 | 否 |

其中 `M` 是 SRAM 大小。Flash Attention 将 Q、K、V 分块（tiling），在 SRAM 中完成 softmax 和 matmul，避免将完整的 n×n attention matrix 写入 HBM。

**关键技术点：**
1. **Tiling**: 将 Q 分为 Br 块，K/V 分为 Bc 块，每块在 SRAM 中计算
2. **Online Softmax**: 使用 `m_new = max(m_old, rowmax(S_block))` 增量更新 softmax 的最大值，避免两遍扫描
3. **重计算（Recomputation）**: 反向传播时不保存 attention matrix，而是从 Q、K、V 重新计算，用时间换空间

Flash Attention v2 进一步优化：减少非 matmul FLOPs、更好的 warp 调度、支持 head_dim 维度的并行。

**面试高频问题**

> **Q: Flash Attention 为什么能加速训练？**
> A: 瓶颈在 HBM 访问而非计算。标准 attention 需要将 O(n²) 的 S 矩阵写入 HBM 再读回做 softmax，Flash Attention 通过 tiling 在 SRAM 中完成整个计算，减少 HBM 读写次数。实际加速比取决于 seq_len 和硬件 SRAM 大小。

> **Q: Flash Attention 和标准 attention 的数值结果是否一致？**
> A: 由于 online softmax 的浮点精度差异，结果会有微小偏差（通常 1e-5 级别），但不影响训练收敛。

---

### 1.2 Weight Tying Bug 修复

**当前实现分析**

`forge/model/transformer.py:80-83`：

```python
if config.tie_weights:
    self.lm_head.weight = self.token_embedding.embedding.weight  # Line 81

self._init_weights()  # Line 83 — 这会重新初始化 lm_head.weight!
```

`_init_weights()` 遍历所有 `nn.Linear` 模块并执行 `nn.init.normal_(module.weight, mean=0.0, std=0.02)`。由于 `lm_head.weight` 已经指向 `token_embedding.embedding.weight`，这个初始化会覆盖已绑定的权重，导致 weight tying 形同虚设——虽然两个参数共享同一个 tensor，但该 tensor 的值被随机初始化覆盖了。

**修复方案**

```python
# 方案一：先初始化，再 tying（推荐）
self._init_weights()
if config.tie_weights:
    self.lm_head.weight = self.token_embedding.embedding.weight

# 方案二：_init_weights 中跳过 lm_head
def _init_weights(self):
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear) and name != "lm_head":
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**原理深度讲解**

Weight Tying（权重共享）由 Press & Wolf (2017) 提出，核心思想是 embedding matrix `E ∈ R^{V×d}` 和 output projection `W ∈ R^{d×V}` 在语义上都是"token 表示"，可以共享。

收益：
- **参数量减少**: 一个 32000 词表、d_model=768 的模型，embedding 参数量为 32000 × 768 = 24.6M，tying 后直接减半
- **正则化效果**: 共享权重相当于对 embedding 和 output projection 施加了强约束
- **训练稳定性**: 输入和输出空间的表示一致性

**面试高频问题**

> **Q: Weight Tying 适用于哪些场景？**
> A: 适用于词表量大、模型参数相对较小的场景。当模型参数远大于词表参数时（如 7B 模型 + 32K 词表），收益很小。GPT-2、T5 使用了 weight tying，LLaMA 没有使用。

> **Q: 除了 input-output tying，还有哪些 weight sharing 方式？**
> A: 跨层参数共享（Universal Transformer）、attention QKV 共享（部分 MQA 变体）、FFN 共享。

---

### 1.3 Gradient Checkpointing

**原理讲解**

训练一个 Transformer 模型时，前向传播需要保存每一层的中间激活值，用于反向传播计算梯度。以一个 TransformerBlock 为例：

| 激活值 | 显存占用 |
|--------|---------|
| Q, K, V 投影结果 | 3 × B × H × S × d_h × dtype_size |
| Attention output | B × H × S × d_h × dtype_size |
| FFN 中间结果 | B × S × d_ff × dtype_size |
| Norm 输入（×2） | 2 × B × S × d_model × dtype_size |

对于 d_model=768, n_layers=12, seq_len=1024, batch_size=8, fp16 的模型，激活值显存约为 `12 × 8 × 1024 × 768 × 4 × 6 ≈ 1.8 GB`。

Gradient Checkpointing 的策略是：**前向传播时不保存中间激活，反向传播时重新计算**。

**实现方案**

```python
import torch
from torch.utils.checkpoint import checkpoint

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...
        self.use_checkpoint = getattr(config, 'gradient_checkpointing', False)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, past_key_values=None):
        x = self.token_embedding(input_ids)
        # ... attention_mask processing ...

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            if self.use_checkpoint and self.training and not use_cache:
                x, present_kv = checkpoint(
                    layer, x, attention_mask, use_cache, past_kv,
                    use_reentrant=False,
                )
            else:
                x, present_kv = layer(x, attention_mask=attention_mask, use_cache=use_cache, past_kv=past_kv)
            # ...

        x = self.norm(x)
        logits = self.lm_head(x)
        # ... loss computation ...
```

**显存-计算 Trade-off**

| 策略 | 显存 | 计算量 |
|------|------|--------|
| 无 checkpoint | O(N × L) | 1x FLOPs |
| 逐层 checkpoint | O(N + L) | ~1.33x FLOPs |
| 逐层 + 选择性 | 可调 | 可调 |

其中 N 是 batch 维度的激活，L 是层数。逐层 checkpoint 需要额外的前向计算（约 33% 开销），但显存从 O(N×L) 降到 O(N+L)。

**面试高频问题**

> **Q: Gradient Checkpointing 的时间开销是多少？为什么是 33%？**
> A: 每个 checkpointed segment 需要重新做一次前向传播来获取激活值。假设模型有 L 层，反向传播本身需要 L 次前向计算（链式法则），checkpoint 额外增加了 L 次前向，总计算量从 2L（前向+反向）变为 3L，增加了 50%。但由于反向传播的 FLOPs 约为前向的 2 倍，实际增加约 33%。

> **Q: 如何选择 checkpoint boundary？**
> A: 常见策略是逐层 checkpoint（每个 TransformerBlock 一个 checkpoint）。更细粒度（如 attention 和 FFN 分别 checkpoint）可以进一步省显存，但重计算开销更大。

---

### 1.4 GQA (Grouped Query Attention) 完整实现

**当前实现分析**

`forge/model/attention.py` 已支持 `n_kv_heads` 参数，`_repeat_kv()` 方法在推理时复制 KV heads：

```python
def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
    if self.n_rep == 1:
        return x
    bs, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bs, n_kv_heads, self.n_rep, seq_len, head_dim)
    return x.reshape(bs, self.n_heads, seq_len, head_dim)
```

**MHA → MQA → GQA 演进**

```
MHA (Multi-Head Attention):    Q: h heads, K: h heads, V: h heads
MQA (Multi-Query Attention):   Q: h heads, K: 1 head,  V: 1 head
GQA (Grouped Query Attention): Q: h heads, K: g heads, V: g heads  (g < h, g | h)
```

GQA 是 MHA 和 MQA 的折中：将 h 个 query heads 分为 g 组，每组共享一对 KV heads。

**显存分析（KV Cache）**

推理时 KV Cache 占用：`2 × n_layers × n_kv_heads × seq_len × head_dim × dtype_size`

| 方案 | n_kv_heads (h=32) | KV Cache (2048 tokens, 32 layers, fp16) |
|------|-------------------|----------------------------------------|
| MHA | 32 | 2 × 32 × 32 × 2048 × 128 × 2B = 1.07 GB |
| GQA (g=8) | 8 | 2 × 32 × 8 × 2048 × 128 × 2B = 268 MB |
| MQA | 1 | 2 × 32 × 1 × 2048 × 128 × 2B = 33.5 MB |

GQA 相比 MHA 节省 75% KV Cache 显存，同时训练质量损失很小（LLaMA 2 70B 使用 GQA g=8）。

**面试高频问题**

> **Q: GQA 的 `_repeat_kv` 操作在训练时和推理时有什么不同？**
> A: 训练时 `_repeat_kv` 会增加显存占用（展开后的 K/V 矩阵），但它让 attention 计算可以复用标准的多头 attention 代码。推理时可以将 repeat 操作融合到 attention kernel 中（如 Flash Decoding），避免显式展开。

> **Q: 为什么 MQA 在实践中效果下降明显，而 GQA 几乎无损？**
> A: MQA 将所有 query heads 压缩到一个 KV head，丢失了多样性。GQA 保留了 g 个 KV heads（通常 g=8），足以捕获不同的语义子空间。

---

### 1.5 SwiGLU 激活函数深度分析

**当前实现分析**

`forge/model/feedforward.py:6-19`：

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
```

**SwiGLU vs ReLU FFN**

标准 FFN：`FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂`

SwiGLU FFN：`FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)`

SwiGLU 引入了 **门控机制**：`W_gate` 分支决定"激活哪些特征"，`W_up` 分支提供"特征值"，两者逐元素相乘。

SiLU（Swish）激活函数：`SiLU(x) = x · σ(x)`，其中 `σ` 是 sigmoid。与 ReLU 相比，SiLU 在 x < 0 时不是硬截断，而是平滑趋近于 0，保持了一定的梯度流。

**参数量差异**

标准 FFN 有 2 个线性层（`d_model → d_ff → d_model`），参数量为 `2 × d_model × d_ff`。

SwiGLU 有 3 个线性层（`d_model → d_ff` 两次，`d_ff → d_model` 一次），参数量为 `3 × d_model × d_ff`。

为保持总参数量不变，SwiGLU 的 `d_ff` 通常设为 `(2/3) × 4 × d_model`（而非 `4 × d_model`），再向上取整到最近的 256 的倍数。

**面试高频问题**

> **Q: 为什么 LLaMA 选择 SwiGLU 而非 ReLU/GELU？**
> A: Shazeer (2020) 的消融实验表明 GLU 变体（SwiGLU、GeGLU）在相同参数量下，语言模型 perplexity 优于 ReLU 和 GELU。门控机制让网络可以学习更复杂的特征交互。

> **Q: SwiGLU 的 d_ff 如何选择？**
> A: LLaMA 使用 `d_ff = 2/3 × 4 × d_model`，然后取整到 256 的倍数。例如 d_model=768 时，`d_ff = 2048`（而非标准的 3072）。

---

## 第2章 训练效率优化

### 2.1 学习率调度：Warmup + Cosine Decay

**当前实现分析**

`forge/training/pretrain.py:73-75`：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=train_cfg["learning_rate"] * 0.1
)
```

当前只有 Cosine Decay，没有 Warmup 阶段。训练初期，模型参数随机初始化，梯度方差大，直接使用大学习率容易导致训练不稳定。

**优化方案：Linear Warmup + Cosine Decay**

```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        # Warmup 阶段：线性增长
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine Decay 阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

# 使用方式
total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
warmup_steps = int(total_steps * 0.05)  # 5% warmup
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
```

**学习率曲线对比**

```
LR
│    /\
│   /  \
│  /    \_____
│ /          \___
│/               \___
└────────────────────→ steps
  warmup  cosine decay

vs 当前（无 warmup）:

LR
│\
│ \_____
│       \___
│           \___
└────────────────────→ steps
  直接开始 decay
```

**原理深度讲解**

**为什么 Warmup 能稳定训练？**

训练初期，Adam/AdamW 的二阶矩估计 `v_t` 尚未收敛。`v_t` 是梯度平方的指数移动平均，初始值为 0，需要若干步才能积累到合理水平。此时有效学习率为 `lr / sqrt(v_t + ε)`，由于 `v_t ≈ 0`，有效学习率被放大到 `lr / sqrt(ε)`（ε 通常为 1e-8），导致参数更新幅度过大。

Warmup 通过在初期使用小学习率，让 `v_t` 有足够的时间积累，避免初期的大步更新破坏训练。

**Cosine Decay vs Linear Decay**

Cosine Decay 在训练中期保持较高学习率，后期快速下降，适合 LLM 训练：
- 中期保持高 LR 有助于跳出局部最优
- 后期低 LR 有助于精细收敛
- 比 Linear Decay 在 perplexity 上通常好 0.1-0.3

**面试高频问题**

> **Q: Warmup 步数一般怎么设置？**
> A: 通常为总步数的 1%-10%。LLaMA 使用 2000 步 warmup。对于小模型/小数据集，5% 是合理默认值。

> **Q: 为什么不用 CosineAnnealingWarmRestarts（带重启的 cosine）？**
> A: LLM 训练通常不使用 warm restart，因为：(1) 学习率突然跳回大会破坏已学到的表示；(2) LLM 训练一般只跑 1-2 个 epoch，不需要重启。

---

### 2.2 Weight Decay 分组

**当前实现分析**

`forge/training/pretrain.py:65-70`：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),  # 所有参数统一 weight_decay
    lr=train_cfg["learning_rate"],
    weight_decay=train_cfg["weight_decay"],
    betas=(0.9, 0.95),
)
```

**优化方案：参数分组**

```python
def get_parameter_groups(model, weight_decay):
    """将参数分为两组：需要 weight decay 和不需要的。"""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # bias、RMSNorm/LayerNorm 的 weight 不施加 weight decay
        if param.ndim <= 1 or "norm" in name.lower() or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

# 使用方式
param_groups = get_parameter_groups(model, weight_decay=0.1)
optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))
```

**为什么 bias 和 norm 参数不施加 Weight Decay？**

Weight Decay 的数学形式是 `L_reg = λ/2 × ||w||²`，梯度为 `∂L_reg/∂w = λw`。

- **Bias 参数**: 通常只有 1 个值 per neuron，正则化效果微乎其微，反而会引入偏移
- **Norm 参数 (LayerNorm/RMSNorm)**: 控制激活值的尺度（scale），施加 weight decay 会压缩激活值范围，导致梯度消失
- **Embedding**: 通常施加 weight decay，但争议较大

**AdamW vs Adam + L2**

| | Adam + L2 | AdamW |
|---|----------|-------|
| 正则化方式 | L2 梯度加入 Adam | weight decay 直接从参数中减去 |
| 公式 | `g_t = ∇L + λw` | `g_t = ∇L`, `w = w - lr*λ*w` |
| 效果 | 被 Adam 的自适应学习率缩放 | 正则化效果与学习率解耦 |

AdamW 的 weight decay 不经过 Adam 的动量和二阶矩，因此正则化效果更纯粹。

**面试高频问题**

> **Q: AdamW 的 weight_decay 一般设多少？**
> A: LLaMA 使用 0.1，GPT-2 使用 0.01。一般在 0.01-0.1 之间调参。

> **Q: 为什么 AdamW 比 Adam + L2 更好？**
> A: Loshchilov & Hutter (2019) 证明 Adam + L2 的 weight decay 被 Adam 的自适应学习率缩放，导致正则化效果不均匀。AdamW 将 weight decay 从梯度计算中解耦，正则化效果更一致。

---

### 2.3 混合精度训练深入

**当前实现分析**

`forge/training/trainer.py:43`：

```python
self.scaler = torch.amp.GradScaler("cuda", enabled=fp16)
```

当前使用 fp16，但未支持 bf16。

**fp16 vs bf16 对比**

| 特性 | fp16 (half) | bf16 (bfloat16) |
|------|------------|-----------------|
| 位数 | 1 sign + 5 exp + 10 mantissa | 1 sign + 8 exp + 7 mantissa |
| 数值范围 | ±65504 | ±3.4×10³⁸ |
| 精度 | 高（10 位尾数） | 低（7 位尾数） |
| 需要 Loss Scaling | 是 | 通常不需要 |
| 硬件支持 | V100+, A100, RTX 3090 | A100+, RTX 4090 |

**关键区别**：bf16 的数值范围与 fp32 相同（8 位指数），因此不容易出现上溢/下溢，通常不需要 loss scaling。但 bf16 的精度较低（7 位尾数 vs fp16 的 10 位），在需要高精度的场景（如 softmax、layer norm）建议使用 fp32。

**优化方案：自动选择精度**

```python
def __init__(self, ..., fp16=True, bf16=False):
    if bf16 and torch.cuda.is_bf16_supported():
        self.dtype = torch.bfloat16
        self.use_scaler = False
    elif fp16:
        self.dtype = torch.float16
        self.use_scaler = True
    else:
        self.dtype = torch.float32
        self.use_scaler = False

    self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

def _train_step(self, batch):
    with torch.amp.autocast("cuda", dtype=self.dtype):
        outputs = self.model(**batch)
        loss = outputs["loss"] / self.gradient_accumulation_steps

    if self.use_scaler:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()

    if self._step_counter % self.gradient_accumulation_steps == 0:
        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
```

**Loss Scaling 原理**

fp16 的最小正数约为 `2^{-24} ≈ 5.96e-8`。训练中梯度可能小于此值，导致下溢为 0。

Loss Scaling 的做法是：将 loss 乘以一个大数（如 1024），反向传播后梯度也相应放大，更新参数前再除以这个数（unscale）。这样小梯度不会下溢。

Dynamic Loss Scaling 的策略：
1. 初始 scale = 2^16
2. 如果连续 N 步没有 inf/nan，scale 翻倍
3. 一旦出现 inf/nan，scale 减半，跳过该步更新

**面试高频问题**

> **Q: 什么时候用 fp16，什么时候用 bf16？**
> A: V100 只支持 fp16。A100/RTX 4090 推荐 bf16（不需要 loss scaling，更稳定）。如果精度敏感（如推理/评估），用 fp32。

> **Q: 混合精度训练中，哪些操作必须用 fp32？**
> A: Softmax、LayerNorm/RMSNorm、Loss 计算。PyTorch 的 autocast 会自动处理这些。

---

## 第3章 Tokenizer 优化

### 3.1 Byte-level Fallback

**当前实现分析**

`forge/tokenizer/bpe_tokenizer.py:111`：

```python
ids.append(self.token_to_id.get(token, self.special_tokens["<unk>"))
```

当前实现中，如果一个字符不在词表中（例如罕见的 Unicode 字符、emoji），会直接映射为 `<unk>`（ID=1），丢失了原始信息。

**优化方案：Byte-level BPE**

```python
def _get_word_tokens(self, text: str) -> list[list[str]]:
    words = self._pattern.findall(text)
    result = []
    for w in words:
        tokens = []
        for c in w:
            if c in self.token_to_id or len(c.encode('utf-8')) == 1:
                tokens.append(c)
            else:
                # Byte-level fallback: 将未知字符转为 byte 表示
                for byte in c.encode('utf-8'):
                    byte_token = f"<0x{byte:02X}>"
                    tokens.append(byte_token)
        result.append(tokens)
    return result

def train(self, texts, verbose=False):
    # 在初始词表中加入所有 256 个 byte token
    for i in range(256):
        byte_token = f"<0x{i:02X}>"
        if byte_token not in self.token_to_id:
            self.token_to_id[byte_token] = len(self.token_to_id)
    # ... rest of training ...
```

**原理讲解**

GPT-2/3/4 的 tokenizer 都使用 byte-level BPE：
1. 将输入文本先编码为 UTF-8 bytes
2. 对 bytes（而非 Unicode 字符）做 BPE 合并
3. 初始词表大小为 256（所有可能的 byte 值）

好处：
- **100% 覆盖率**: 任何 Unicode 字符都能被 byte 序列表示
- **无 `<unk>`**: 不会丢失信息
- **语言无关**: 中文、日文、emoji 都能处理

**面试高频问题**

> **Q: Byte-level BPE 的缺点是什么？**
> A: 对于中文等多字节语言，同一个字符可能被拆成 3 个 byte token（UTF-8 编码），导致序列变长、训练效率下降。解决方案是增大词表或使用针对目标语言优化的 pre-tokenization。

> **Q: BPE 和 WordPiece、Unigram 有什么区别？**
> A: BPE 是自底向上合并频率最高的 pair；WordPiece 是自底向上合并使语言模型似然最大的 pair；Unigram 是自顶向下从大词表中删减使整体似然损失最小的 token。SentencePiece 库同时支持 BPE 和 Unigram。

---

### 3.2 Pre-tokenization 正则

**当前实现分析**

`forge/tokenizer/bpe_tokenizer.py:21`：

```python
self._pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[一-鿿]|[a-zA-Z]+|[0-9]+|[^\s\w一-鿿]""")
```

这个正则将文本分为：英文缩写、中文字符、英文单词、数字、标点符号。

**可优化方向**

GPT-4 的 `cl100k_base` tokenizer 使用更精细的 pre-tokenization：

```python
# GPT-4 cl100k_base 的 pre-tokenization regex（简化版）
PAT = r"""
(?i:'s|'t|'re|'ve|'m|'ll|'d)  # 英文缩写
|[^\r\n\p{L}\p{N}]?\p{L}+     # 以可选标点开头的字母序列
|\p{N}{1,3}                    # 1-3位数字
| ?[^\s\p{L}\p{N}]+[\r\n]*    # 标点+可选换行
|\s*[\r\n]+                    # 换行
|\s+(?!\S)                     # 尾部空格
|\s+                           # 其他空格
"""
```

**面试高频问题**

> **Q: 为什么需要 pre-tokenization？直接对原始文本做 BPE 有什么问题？**
> A: 不做 pre-tokenization 会跨越单词边界做合并，例如 "the cat" 可能合并出 "the_" 或 "_cat" 这样的 token，导致词表膨胀且语义不清晰。Pre-tokenization 确保合并只在"词"内部发生。

---

### 3.3 词表大小选择

**Trade-off 分析**

| 词表大小 | 优点 | 缺点 |
|---------|------|------|
| 8000 | 每个 token 训练充分 | 序列长、推理慢、中文效率低 |
| 32000 | 平衡选择 | 中等 |
| 64000 | 中文效率高、序列短 | embedding 层参数多、稀有 token 训练不足 |
| 128000 | 极高效率 | 参数浪费严重、需要更多训练数据 |

**参数量影响**

Embedding 层参数量 = `vocab_size × d_model`。

| vocab_size | d_model=768 | d_model=1024 |
|-----------|-------------|--------------|
| 32000 | 24.6M | 32.8M |
| 64000 | 49.2M | 65.5M |
| 128000 | 98.3M | 131.1M |

对于 100-500M 参数的模型，词表大小对参数量的影响显著。32000 是合理默认值。

**面试高频问题**

> **Q: 如何选择合适的词表大小？**
> A: 经验法则：(1) 训练数据量越大，词表可以越大；(2) 目标语言的字符多样性越高，词表应越大；(3) 模型参数量越大，词表可以越大。LLaMA 用 32K，Qwen 用 152K（大幅增加中文效率）。

---

## 第4章 RAG 系统优化

### 4.1 Embedding 模型质量

**当前实现分析**

`forge/rag/embedder.py:40-53`：

```python
def _fallback_embed(self, texts):
    import hashlib
    dim = 384
    embeddings = []
    for text in texts:
        h = hashlib.md5(text.encode()).hexdigest()
        vec = np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), dim*2), 2)])
        # ...
```

Fallback 使用 MD5 hash 生成伪 embedding。MD5 是确定性的哈希函数，**完全没有语义信息**——"猫"和"狗"的 hash embedding 毫无相关性，而"猫"和"今天天气很好"的 hash embedding 可能随机相似。

**优化方案：集成真实 Embedding 模型**

```python
# 推荐模型（中文场景）
CHINESE_EMBEDDING_MODELS = {
    "small": "shibing624/text2vec-base-chinese",      # 384 dim, ~100MB
    "medium": "BAAI/bge-small-zh-v1.5",               # 512 dim, ~100MB
    "large": "BAAI/bge-large-zh-v1.5",                # 1024 dim, ~1.3GB
    "best": "BAAI/bge-m3",                             # 1024 dim, multilingual
}

# 使用 sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
embeddings = model.encode(
    ["你好世界", "今天天气很好"],
    normalize_embeddings=True,  # L2 归一化，方便余弦相似度计算
    batch_size=32,
)
```

**Embedding 模型选型要点**

| 维度 | 考虑因素 |
|------|---------|
| 维度 | 384-1024，越高表达能力越强但检索越慢 |
| 速度 | 小模型（100MB）推理速度是大模型的 5-10 倍 |
| 质量 | 参考 MTEB/C-MTEB 排行榜 |
| 语言 | 中文场景优先选择专门的中文模型 |

**面试高频问题**

> **Q: 如何评估 Embedding 模型的质量？**
> A: 使用 MTEB (Massive Text Embedding Benchmark) 排行榜，评估维度包括：检索（NDCG@10）、分类（Accuracy）、聚类（V-measure）、重排序（MAP）等。中文场景关注 C-MTEB。

> **Q: Embedding 模型需要 fine-tune 吗？**
> A: 通用场景不需要。但如果领域术语多（医疗、法律），fine-tune 可以显著提升检索质量。常用方法：对比学习（contrastive learning），使用 (query, positive_doc, negative_doc) 三元组训练。

---

### 4.2 Chunking 策略

**当前实现分析**

`forge/rag/text_splitter.py` 使用递归字符切分，按 `["\n\n", "\n", " ", ""]` 的优先级逐步切分。

**优化方向**

**1. 语义切分（Semantic Chunking）**

```python
# 语义切分的核心思想：在语义边界处切分
# 步骤：
# 1. 按句子分割文本
# 2. 计算相邻句子的 embedding 相似度
# 3. 在相似度骤降处切分（语义转折点）

def semantic_split(text, embedder, threshold=0.5):
    sentences = split_into_sentences(text)
    embeddings = embedder.embed(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

**2. Parent-Child Chunking**

```
Parent Chunk (1024 tokens): "机器学习是人工智能的一个分支..."
├── Child Chunk 1 (256 tokens): "机器学习是人工智能的一个分支，它使用数据来训练算法。"
├── Child Chunk 2 (256 tokens): "常见的机器学习方法包括监督学习、无监督学习和强化学习。"
└── Child Chunk 3 (256 tokens): "深度学习是机器学习的一个子集，使用多层神经网络。"
```

检索时用 Child Chunk（更精确匹配），生成时用 Parent Chunk（更完整上下文）。

**面试高频问题**

> **Q: Chunk size 对 RAG 质量有什么影响？**
> A: 小 chunk（128-256 tokens）检索精度高但上下文不完整；大 chunk（1024+ tokens）上下文完整但检索噪声多。经验法则是 256-512 tokens，overlap 10-20%。

> **Q: Overlap 的作用是什么？**
> A: 防止关键信息被切断在两个 chunk 的边界。例如一个完整的定义可能跨越 chunk 边界，overlap 确保至少一个 chunk 包含完整定义。

---

### 4.3 检索优化：Hybrid Search

**当前实现分析**

`forge/rag/retriever.py` 仅支持向量检索（dense retrieval）和 MMR 多样性检索。

**优化方案：Hybrid Search + RRF**

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """结合 BM25 稀疏检索和向量稠密检索。"""

    def __init__(self, embedder, vector_store, bm25_corpus=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = BM25Okapi(bm25_corpus) if bm25_corpus else None

    def retrieve(self, query, top_k=5, alpha=0.5):
        # Dense retrieval
        dense_results = self.vector_store.search(
            self.embedder.embed_query(query), top_k=top_k * 3
        )

        # Sparse retrieval (BM25)
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k * 3]

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60  # RRF 常数

        for rank, doc in enumerate(dense_results):
            doc_id = doc.get("id", doc["content"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (k + rank + 1)

        for rank, idx in enumerate(bm25_top):
            doc_id = self.vector_store.documents[idx].get("id", self.vector_store.documents[idx]["content"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)

        # 按 RRF 分数排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
```

**Dense vs Sparse Retrieval**

| | Dense (向量) | Sparse (BM25) |
|---|------------|---------------|
| 原理 | 语义相似度 | 词频匹配 |
| 优点 | 理解同义词、语义 | 精确匹配关键词、速度快 |
| 缺点 | 可能忽略精确匹配 | 无法理解同义词 |
| 适用 | 语义检索 | 关键词检索 |

**RRF (Reciprocal Rank Fusion)**

RRF 的公式：`score(d) = Σ 1 / (k + rank_i(d))`，其中 k=60 是常数。

RRF 的好处是不需要对不同检索方法的分数做归一化（不同方法的分数范围不同），直接用排名融合。

**面试高频问题**

> **Q: 为什么 Hybrid Search 通常比纯 Dense 或纯 Sparse 好？**
> A: 两种方法互补。Dense 擅长语义匹配（"汽车"匹配"轿车"），Sparse 擅长精确匹配（"GPT-4"精确匹配"GPT-4"）。融合后覆盖面更广。

> **Q: RRF 中的 k 参数怎么选？**
> A: k=60 是原始论文（Cormack et al., 2009）的推荐值。k 越大，排名靠后的文档权重越高。实际中 k 在 30-100 之间差异不大。

---

### 4.4 向量库索引类型选择

**FAISS 索引类型对比**

| 索引类型 | 构建速度 | 查询速度 | 内存占用 | 精度 |
|---------|---------|---------|---------|------|
| Flat (暴力搜索) | 最快 | 最慢 | 最高 | 100% |
| IVF (倒排索引) | 中等 | 快 | 中等 | 95-99% |
| HNSW (图索引) | 慢 | 最快 | 高 | 97-99% |
| IVF+PQ (乘积量化) | 中等 | 快 | 最低 | 90-95% |

**选择建议**

- **< 10K 文档**: Flat（暴力搜索足够快）
- **10K - 1M 文档**: IVF 或 HNSW
- **> 1M 文档**: IVF+PQ（节省内存）

**面试高频问题**

> **Q: HNSW 的原理是什么？**
> A: HNSW (Hierarchical Navigable Small World) 是基于图的索引。构建时，每个新向量与已有的近邻建立连接，形成多层图结构。查询时从顶层开始，逐层向下贪心搜索，每层找到当前最近的节点，最终在底层得到精确的 top-k 结果。查询复杂度 O(log N)。

> **Q: IVF 的 nlist 参数怎么设置？**
> A: 经验法则是 `nlist = sqrt(N)` 到 `4 * sqrt(N)`，其中 N 是文档数量。nprobe（查询时搜索的倒排列表数）通常设为 nlist 的 1-10%。

---

## 第5章 Agent 系统优化

### 5.1 ReAct 循环的 Context 管理

**当前实现分析**

`forge/agent/react_agent.py:56-86`：

```python
def run(self, query):
    conversation = f"{system_prompt}\n\nUser: {query}\n"
    for turn in range(self.max_turns):
        response = self._generate(conversation, max_new_tokens=512)
        conversation += response
        # ... 无上下文截断 ...
```

每轮循环都会将完整的 conversation 传入模型，随着工具调用的 Observation 内容累积，context 可能迅速膨胀到超出模型的 `max_seq_len`。

**优化方案：滑动窗口 + 摘要压缩**

```python
def _truncate_conversation(self, conversation: str, max_tokens: int = 2048) -> str:
    """截断对话，保留系统提示和最近的交互。"""
    tokens = self.tokenizer.encode(conversation)
    if len(tokens) <= max_tokens:
        return conversation

    # 保留 system prompt（前 N 个 token）
    system_end = conversation.find("User:")
    system_part = conversation[:system_end] if system_end > 0 else ""

    # 保留最近的交互（从末尾截取）
    recent_tokens = tokens[-(max_tokens - len(self.tokenizer.encode(system_part))):]
    recent_part = self.tokenizer.decode(recent_tokens, skip_special=True)

    return system_part + "\n... (历史已压缩) ...\n" + recent_part

def _summarize_history(self, conversation: str) -> str:
    """用模型自身压缩历史。"""
    summary_prompt = "请用3句话总结以下对话的关键信息：\n" + conversation[-2000:]
    summary = self._generate(summary_prompt, max_new_tokens=200)
    return f"历史摘要: {summary}"
```

**面试高频问题**

> **Q: Agent 的 context window 管理有哪些策略？**
> A: (1) 滑动窗口：保留最近 N 轮；(2) 摘要压缩：用 LLM 总结历史；(3) 摘要 + 滑动窗口：对旧历史做摘要，保留最近几轮原始对话；(4) 记忆系统：将重要信息存入外部记忆，按需检索。

> **Q: 滑动窗口和摘要压缩各有什么优缺点？**
> A: 滑动窗口简单高效但会丢失早期信息。摘要压缩保留了关键信息但引入额外的 LLM 调用开销，且摘要可能丢失细节。

---

### 5.2 Function Calling 协议

**当前实现分析**

`forge/agent/react_agent.py:40-48` 使用正则表达式解析文本格式的 Action：

```python
def _parse_action(self, text):
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)
    # ...
```

这种方式依赖模型严格按照格式输出，容易出错。

**优化方案：结构化 JSON Schema**

```python
class FunctionCallAgent:
    """使用 JSON Schema 的结构化 Function Calling。"""

    TOOL_SCHEMA = {
        "calculator": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        },
        # ...
    }

    def _build_system_prompt(self):
        tools_json = json.dumps(list(self.TOOL_SCHEMA.values()), ensure_ascii=False, indent=2)
        return f"""你是一个智能助手。你可以使用以下工具：

{tools_json}

当你需要使用工具时，请输出以下 JSON 格式：
{{"tool": "工具名", "parameters": {{"参数名": "参数值"}}}}

当你不需要工具时，直接输出回答。
"""

    def _parse_tool_call(self, text: str) -> Optional[dict]:
        """解析 JSON 格式的工具调用。"""
        # 尝试提取 JSON 块
        json_match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"[^{}]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None
```

**面试高频问题**

> **Q: OpenAI 的 Function Calling 和 ReAct 有什么区别？**
> A: ReAct 是通用的推理框架，模型通过文本格式输出 Action。Function Calling 是结构化的协议，模型输出 JSON 格式的工具调用，解析更可靠。OpenAI 的 Function Calling 实际上是 ReAct 的结构化版本。

> **Q: 如何让模型正确输出 JSON 格式的工具调用？**
> A: (1) 在 system prompt 中给出明确的格式说明和示例；(2) 使用 constrained decoding（如 Outlines、Guidance）强制输出合法 JSON；(3) Fine-tune 模型以适应工具调用格式。

---

### 5.3 记忆系统增强

**当前实现分析**

`forge/agent/memory.py:72-80`：

```python
def _simple_embed(self, text: str) -> np.ndarray:
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    vec = np.array([int(h[i:i+2], 16) / 255.0 ...])
```

与 RAG 的 Embedder 相同，使用 MD5 hash 生成伪 embedding，**完全没有语义检索能力**。

**优化方案**

```python
class EnhancedAgentMemory:
    def __init__(self, embedder, max_short_term=20):
        self.embedder = embedder  # 复用 RAG 的 Embedder
        self.short_term = []
        self.long_term = []
        self.long_term_embeddings = []
        self.importance_scores = []  # 重要性评分
        self.access_counts = []      # 访问次数（用于衰减）

    def add_long_term(self, content, importance=1.0):
        self.long_term.append({"content": content, "timestamp": time.time()})
        self.long_term_embeddings.append(self.embedder.embed_query(content))
        self.importance_scores.append(importance)
        self.access_counts.append(0)

    def search_long_term(self, query, top_k=3, recency_weight=0.1):
        if not self.long_term_embeddings:
            return []

        query_emb = self.embedder.embed_query(query)
        scores = []
        current_time = time.time()

        for i, emb in enumerate(self.long_term_embeddings):
            # 语义相似度
            semantic_score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            # 重要性加权
            importance = self.importance_scores[i]
            # 时效性衰减
            age_hours = (current_time - self.long_term[i]["timestamp"]) / 3600
            recency = 1.0 / (1.0 + recency_weight * age_hours)

            final_score = semantic_score * importance * recency
            scores.append(final_score)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            self.access_counts[idx] += 1
            entry = self.long_term[idx].copy()
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results
```

**面试高频问题**

> **Q: Agent 记忆系统有哪些设计模式？**
> A: (1) 短期记忆（对话窗口）；(2) 长期记忆（向量数据库）；(3) 工作记忆（当前任务的中间状态）；(4) 情景记忆（具体事件）；(5) 语义记忆（抽象知识）。MemGPT、Generative Agents 等论文提出了不同的记忆架构。

> **Q: 如何评估记忆系统的质量？**
> A: (1) 检索准确率：给定 query，记忆系统能否返回相关记忆；(2) 时效性：旧记忆是否被适当衰减；(3) 容量管理：记忆过多时的遗忘策略。

---

### 5.4 错误恢复与重试

**当前实现分析**

`forge/agent/react_agent.py:77-82`：

```python
if action_name in self.tools:
    observation = self.tools[action_name].run(action_input)
    conversation += f"\nObservation: {observation}\n"
else:
    conversation += f"\nObservation: 错误 - 未知工具 '{action_name}'\n"
```

工具调用失败时只是在对话中记录错误，没有重试机制。

**优化方案**

```python
class RobustReActAgent(ReActAgent):
    def __init__(self, *args, max_retries=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries

    def _execute_tool_with_retry(self, tool_name, tool_input):
        """带重试的工具执行。"""
        if tool_name not in self.tools:
            return self._find_alternative_tool(tool_name, tool_input)

        for attempt in range(self.max_retries + 1):
            try:
                result = self.tools[tool_name].run(tool_input)
                if result and not result.startswith("Error"):
                    return result
            except Exception as e:
                logger.warning(f"Tool {tool_name} attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避

        return f"工具 {tool_name} 执行失败，已重试 {self.max_retries} 次。"

    def _find_alternative_tool(self, tool_name, tool_input):
        """当工具不存在时，尝试找到替代工具。"""
        # 简单的名称相似度匹配
        alternatives = {
            "search": "web_search",
            "calculate": "calculator",
            "exec": "code_executor",
        }
        alt = alternatives.get(tool_name)
        if alt and alt in self.tools:
            logger.info(f"Using alternative tool {alt} instead of {tool_name}")
            return self.tools[alt].run(tool_input)
        return f"未知工具 '{tool_name}'，可用工具: {list(self.tools.keys())}"
```

**面试高频问题**

> **Q: Agent 的错误处理有哪些策略？**
> A: (1) 重试（retry）：适合临时性错误；(2) 回退（fallback）：使用替代工具或方法；(3) 跳过（skip）：跳过当前步骤继续执行；(4) 终止（abort）：安全相关错误立即终止；(5) 反馈（feedback）：将错误信息反馈给模型重新推理。

---

## 第6章 评估体系优化

### 6.1 确定性评估

**当前实现分析**

`forge/evaluation/harness.py:51-57`：

```python
output_ids = model.generate(
    input_tensor,
    max_new_tokens=max_new_tokens,
    temperature=0.7,   # 评估时使用 temperature=0.7，结果不确定
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)
```

评估时使用 `temperature=0.7` 和 `top_p=0.9`，导致每次评估结果不同，不可复现。

**优化方案**

```python
def evaluate_generation(self, model, tokenizer, test_data, ...):
    # ...

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.0,       # 确定性：greedy decoding
            eos_token_id=tokenizer.eos_token_id,
        )
    # temperature=0 时，generate() 内部使用 argmax，结果完全确定
```

**面试高频问题**

> **Q: 评估时为什么必须用 temperature=0？**
> A: 评估的目的是衡量模型的能力，而非引入随机性。temperature > 0 会引入采样噪声，导致同一模型在不同运行中得到不同的指标值，无法公平比较不同模型或不同 checkpoint。

> **Q: 训练时用 temperature > 0 有什么好处？**
> A: 训练时通常不直接使用 temperature（那是推理时的概念）。但在 RLHF/DPO 中，生成训练数据时使用 temperature > 0 可以增加数据多样性。

---

### 6.2 批量推理

**当前实现分析**

`forge/evaluation/harness.py:42-66`：

```python
for item in test_data:
    # 逐条生成，效率低
    input_tensor = torch.tensor([input_ids], device=device)
    output_ids = model.generate(input_tensor, ...)
```

逐条生成无法利用 GPU 的并行计算能力。

**优化方案**

```python
def evaluate_generation_batched(self, model, tokenizer, test_data,
                                 batch_size=8, max_new_tokens=128, device="cuda"):
    """批量推理评估。"""
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_references = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        prompts = [item.get("prompt", "") for item in batch]
        references = [item.get("reference", "") for item in batch]

        # 编码并 padding
        batch_ids = [tokenizer.encode(p, add_special=True) for p in prompts]
        max_len = max(len(ids) for ids in batch_ids)

        # Left padding（生成任务通常用 left padding）
        padded_ids = []
        attention_masks = []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_tensor = torch.tensor(padded_ids, device=device)
        mask_tensor = torch.tensor(attention_masks, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, out_ids in enumerate(output_ids):
            prediction = tokenizer.decode(out_ids.tolist(), skip_special=True)
            # 去掉 prompt 部分
            if prediction.startswith(prompts[j]):
                prediction = prediction[len(prompts[j]):]
            all_predictions.append(prediction.strip())
            all_references.append(references[j].strip())

    return compute_metrics(all_predictions, all_references)
```

**面试高频问题**

> **Q: 批量推理时为什么用 left padding？**
> A: 生成任务中，所有序列从左到右生成。Left padding 保证有效 token 在右侧对齐，padding 在左侧。这样 KV Cache 中有效 token 连续存储，计算效率更高。

> **Q: Dynamic Batching 和 Static Batching 有什么区别？**
> A: Static Batching 要求同一批次的所有序列长度相同（padding 到最长）。Dynamic Batching 允许不同请求在不同时刻完成，空出的 slot 可以插入新请求，提高 GPU 利用率。vLLM 的 Continuous Batching 更进一步，实现了 token 级别的调度。

---

## 第7章 数据流水线优化

### 7.1 Streaming Dataset

**当前实现分析**

`forge/data/dataset.py` 中的 `PretrainDataset` 在 `__init__` 中加载全部数据到内存。

**优化方案：IterableDataset**

```python
from torch.utils.data import IterableDataset

class StreamingPretrainDataset(IterableDataset):
    """流式读取大规模训练数据。"""

    def __init__(self, file_path, tokenizer, max_length=1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(worker_id=0, num_workers=1):
        # 支持多 worker 并行读取
        with open(self.file_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                # 每个 worker 处理不同的行
                if line_idx % num_workers != worker_id:
                    continue
                text = line.strip()
                if not text:
                    continue
                ids = self.tokenizer.encode(text, add_special=True)
                if len(ids) > self.max_length:
                    ids = ids[:self.max_length]
                yield {"input_ids": ids, "labels": ids}
```

**面试高频问题**

> **Q: IterableDataset 和 Map-style Dataset 的区别？**
> A: Map-style Dataset 支持随机访问（`dataset[i]`），需要实现 `__getitem__` 和 `__len__`。IterableDataset 按顺序迭代，适合大规模数据。Map-style Dataset 支持 shuffle，IterableDataset 需要手动实现 shuffle（如 buffer shuffle）。

> **Q: 多 worker 读取 IterableDataset 时如何避免数据重复？**
> A: 通过 `worker_id` 和 `num_workers` 做数据分片。PyTorch 的 `worker_init_fn` 可以设置每个 worker 的随机种子。

---

### 7.2 Data Packing

**当前实现分析**

当前每条数据独立 padding 到 `max_length`，当数据长度差异大时，padding 浪费大量计算。

**优化方案：Packing**

```python
def pack_sequences(tokenized_data, max_length=1024, eos_token_id=3):
    """将多条短文本拼接为一个固定长度的样本。"""
    packed_samples = []
    current_ids = []
    current_labels = []

    for item in tokenized_data:
        ids = item["input_ids"]

        if len(current_ids) + len(ids) <= max_length:
            current_ids.extend(ids)
            current_labels.extend(ids)
        else:
            # 填充到 max_length
            if len(current_ids) < max_length:
                pad_len = max_length - len(current_ids)
                current_ids.extend([0] * pad_len)  # pad_token_id = 0
                current_labels.extend([-100] * pad_len)
            packed_samples.append({
                "input_ids": current_ids[:max_length],
                "labels": current_labels[:max_length],
            })
            current_ids = list(ids)
            current_labels = list(ids)

    if current_ids:
        # 处理最后一个样本
        # ...

    return packed_samples
```

**Packing 的 Attention Mask 问题**

Packing 后，多条文本在同一序列中，但它们之间不应该相互 attend。需要使用 block-diagonal attention mask：

```python
def create_block_diagonal_mask(seq_ids, max_seq_len):
    """
    seq_ids: 每个 token 属于哪个序列 [0,0,0,1,1,1,2,2,...]
    返回: (max_seq_len, max_seq_len) 的 attention mask
    """
    mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
    for i in range(max_seq_len):
        for j in range(max_seq_len):
            if seq_ids[i] == seq_ids[j] and j <= i:  # 同一序列内的因果注意力
                mask[i][j] = 0.0
    return mask
```

**面试高频问题**

> **Q: Data Packing 能提升多少训练效率？**
> A: 取决于数据长度分布。如果平均长度为 max_length 的 25%，packing 可以提升约 3-4 倍吞量。但需要额外处理 attention mask，实际加速比通常为 2-3 倍。

---

### 7.3 数据质量过滤

**常见过滤策略**

```python
def filter_training_data(documents, min_length=50, max_length=100000):
    """数据质量过滤。"""
    filtered = []
    seen_hashes = set()  # 用于去重

    for doc in documents:
        text = doc["content"]

        # 1. 长度过滤
        if len(text) < min_length or len(text) > max_length:
            continue

        # 2. 去重（基于内容 hash）
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        # 3. 语言检测（可选）
        # 4. 质量评分（可选，可用 perplexity 过滤低质量文本）

        filtered.append(doc)

    return filtered
```

**面试高频问题**

> **Q: 为什么数据去重很重要？**
> A: Lee et al. (2022) 发现训练数据中的重复会导致：(1) 模型记忆训练数据而非学习泛化能力；(2) 训练不稳定（loss spike）；(3) 生成时输出训练数据片段（隐私风险）。

> **Q: 如何高效地对大规模数据去重？**
> A: (1) MinHash + LSH（Locality-Sensitive Hashing）：近似去重，速度快；(2) Suffix Array：精确子串去重；(3) Bloom Filter：精确字符串去重，空间效率高。

---

## 第8章 工程实践与面试高频问题

### 8.1 项目架构设计决策

**为什么选择 LLaMA-style 而非 GPT-2 style？**

| 组件 | GPT-2 | LLaMA (本项目) |
|------|-------|----------------|
| 归一化 | Post-Norm (LayerNorm after residual) | Pre-Norm (RMSNorm before attention/FFN) |
| 激活函数 | GELU | SwiGLU |
| 位置编码 | Learned Positional Embedding | RoPE |
| Attention | MHA | MHA + GQA |
| Bias | 有 bias | 无 bias (Linear, RMSNorm) |

**Pre-Norm vs Post-Norm**

```
Post-Norm (GPT-2):  x = LayerNorm(x + Attention(x))
Pre-Norm (LLaMA):   x = x + Attention(LayerNorm(x))
```

Pre-Norm 的优势：
- 梯度流更稳定：残差连接直接传递梯度，不经过 Norm 层
- 训练更稳定：不需要 warmup 也能训练（虽然有 warmup 更好）
- 收敛更快：相同学习率下 loss 下降更快

**面试高频问题**

> **Q: Pre-Norm 和 Post-Norm 哪个更好？**
> A: Pre-Norm 训练更稳定，但 Post-Norm 的最终性能可能更好（Xiong et al., 2020）。工业界普遍选择 Pre-Norm，因为训练稳定性更重要。LLaMA、GPT-3、PaLM 都使用 Pre-Norm。

---

### 8.2 性能分析与 Profiling

**GPU 显存分析**

一个 LLM 训练时的显存占用可以分为 4 部分：

```
总显存 = 模型参数 + 梯度 + 优化器状态 + 激活值
```

以 d_model=768, n_layers=12, vocab_size=32000 的模型为例（fp16 训练）：

| 组成部分 | 计算公式 | 显存占用 |
|---------|---------|---------|
| 模型参数 (fp16) | P × 2 bytes | ~150 MB |
| 梯度 (fp16) | P × 2 bytes | ~150 MB |
| Adam 状态 (m, v, fp32) | P × 8 bytes | ~600 MB |
| 参数副本 (fp32) | P × 4 bytes | ~300 MB |
| 激活值 (fp16, batch=8, seq=1024) | 约 6 × n_layers × B × S × d × 2 | ~1.8 GB |
| **总计** | | **~3 GB** |

**FLOPs 估算**

前向传播的 FLOPs：`F_fwd ≈ 2 × P × S × B`（每个参数做一次乘加）

总训练 FLOPs：`F_total ≈ 6 × P × S × B × N_steps`（前向 + 反向 ≈ 3 × 前向）

**torch.profiler 使用**

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        with record_function("forward"):
            outputs = model(**batch)
        with record_function("backward"):
            outputs["loss"].backward()
        prof.step()

# 打印 top 10 耗时操作
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**面试高频问题**

> **Q: 如何估算一个 LLM 的训练成本？**
> A: 使用 Chinchilla scaling law 的近似公式：`C ≈ 6 × P × D`，其中 P 是参数量，D 是训练 token 数。例如 1B 参数模型训练 100B tokens，需要 6 × 10^9 × 10^11 = 6 × 10^20 FLOPs。A100 的 BF16 算力约为 312 TFLOPS，理论最短时间为 6×10^20 / (312×10^12) ≈ 1923 秒 ≈ 0.53 小时（实际利用率约 30-50%，所以约 1-2 小时）。

> **Q: 什么是 MFU (Model FLOPs Utilization)？**
> A: MFU = 实际 FLOPs / 硬件峰值 FLOPs。工业级 LLM 训练的 MFU 通常在 40-60%。影响 MFU 的因素：通信开销、内存带宽瓶颈、kernel 融合效率。

---

### 8.3 分布式训练基础

**DDP (Distributed Data Parallel)**

```
GPU 0: Model Replica + Data Shard 0 → Grad 0 ─┐
GPU 1: Model Replica + Data Shard 1 → Grad 1 ─┼→ AllReduce → 更新参数
GPU 2: Model Replica + Data Shard 2 → Grad 2 ─┤
GPU 3: Model Replica + Data Shard 3 → Grad 3 ─┘
```

DDP 的核心：每个 GPU 持有完整模型副本，处理不同的数据批次，梯度通过 AllReduce 同步。

**FSDP (Fully Sharded Data Parallel)**

```
GPU 0: Layer 0-3 参数分片 + Layer 4-7 激活值
GPU 1: Layer 4-7 参数分片 + Layer 0-3 激活值

前向传播时：
  Layer 0: GPU 0 从所有 GPU 收集完整参数 → 计算 → 释放
  Layer 4: GPU 0 从所有 GPU 收集完整参数 → 计算 → 释放
```

FSDP 将模型参数、梯度、优化器状态分片到所有 GPU，需要时通过 AllGather 收集完整参数。

**对比**

| | DDP | FSDP |
|---|-----|------|
| 每 GPU 显存 | 完整模型 | 1/N 模型 |
| 通信量 | 2× 模型大小 | 3× 模型大小（额外 AllGather） |
| 适用场景 | 模型能放入单卡 | 模型太大无法放入单卡 |

**面试高频问题**

> **Q: 什么时候用 DDP，什么时候用 FSDP？**
> A: 模型能放入单卡时用 DDP（通信量小、实现简单）。模型太大时用 FSDP（如 7B 模型在 4×A100 上）。FSDP 的通信开销比 DDP 大约 1.5 倍。

> **Q: 模型并行和流水线并行是什么？**
> A: 模型并行（Tensor Parallel）将单个层的参数切分到多个 GPU（如 attention 的不同 head 在不同 GPU）。流水线并行（Pipeline Parallel）将不同层分配到不同 GPU，通过 micro-batch 实现流水线执行。

---

### 8.4 部署与推理优化

**KV Cache**

推理时，每生成一个新 token 需要计算 attention。如果不缓存 K/V，每步都需要重新计算所有 token 的 K/V，复杂度 O(n²)。

KV Cache 缓存已计算的 K/V，每步只需计算新 token 的 K/V，复杂度降到 O(n)。

```
无 KV Cache:
  Step 1: 计算 token 0 的 K, V          → O(1)
  Step 2: 计算 token 0-1 的 K, V        → O(2)
  Step 3: 计算 token 0-2 的 K, V        → O(3)
  总计: O(n²)

有 KV Cache:
  Step 1: 计算 token 0 的 K, V, 缓存    → O(1)
  Step 2: 计算 token 1 的 K, V, 追加    → O(1)
  Step 3: 计算 token 2 的 K, V, 追加    → O(1)
  总计: O(n)
```

**模型量化**

量化将模型参数从 fp16/fp32 降低到更低位数，减少显存占用和推理延迟。

| 量化类型 | 位数 | 显存节省 | 精度损失 |
|---------|------|---------|---------|
| INT8 | 8-bit | 2× (vs fp16) | 极小 |
| INT4 | 4-bit | 4× (vs fp16) | 小 |
| GPTQ | 4-bit | 4× | 小 |
| AWQ | 4-bit | 4× | 极小 |
| GGUF | 2-8 bit | 可变 | 取决于位数 |

**面试高频问题**

> **Q: INT8 量化为什么精度损失很小？**
> A: LLM 的权重分布通常近似正态分布，大部分值集中在 0 附近。INT8 有 256 个量化级别，足以捕获主要信息。LLM.int8()（Dettmers et al., 2022）发现 outlier features 需要用 fp16 保留，其余用 INT8 量化。

> **Q: vLLM 为什么比 HuggingFace 推理快？**
> A: vLLM 的核心创新是 PagedAttention：将 KV Cache 分页管理（类似操作系统虚拟内存），避免内存碎片化，支持更高的并发请求数。此外，Continuous Batching（动态批处理）和 Kernel 融合也贡献了显著加速。

> **Q: Speculative Decoding 是什么？**
> A: 用一个小模型（draft model）快速生成若干候选 token，然后用大模型一次性验证。如果小模型猜对了（概率足够高），直接接受，省去大模型的多次前向传播。验证是并行的（一个 batch），所以不会增加延迟。

---

## 附录 优化优先级矩阵

按 **实施难度** 和 **收益** 排序：

| 优先级 | 优化项 | 难度 | 收益 | 涉及文件 |
|--------|--------|------|------|---------|
| P0 | Flash Attention (SDPA) | 低 | 高 | `attention.py` |
| P0 | Weight Tying Bug 修复 | 低 | 中 | `transformer.py` |
| P0 | 评估时 temperature=0 | 低 | 中 | `harness.py` |
| P1 | Warmup + Cosine LR | 低 | 高 | `pretrain.py` |
| P1 | Weight Decay 分组 | 低 | 中 | `pretrain.py` |
| P1 | Byte-level BPE Fallback | 中 | 中 | `bpe_tokenizer.py` |
| P1 | 真实 Embedding 模型 | 低 | 高 | `embedder.py` |
| P2 | Gradient Checkpointing | 中 | 高 | `transformer.py` |
| P2 | 批量推理评估 | 中 | 中 | `harness.py` |
| P2 | Context 截断 (Agent) | 中 | 中 | `react_agent.py` |
| P2 | 记忆系统 Embedding 修复 | 低 | 中 | `memory.py` |
| P3 | Hybrid Search (BM25+向量) | 高 | 中 | `retriever.py` |
| P3 | Data Packing | 高 | 高 | `dataset.py` |
| P3 | Streaming Dataset | 中 | 中 | `dataset.py` |
| P3 | bf16 支持 | 低 | 低 | `trainer.py` |

---

## 面试知识速查卡

### 核心概念一句话总结

| 概念 | 一句话 |
|------|--------|
| RoPE | 通过旋转 Q/K 向量编码相对位置，旋转角度与位置成正比 |
| SwiGLU | 门控 FFN：SiLU(W_gate·x) ⊙ W_up·x，比 ReLU FFN 表达能力更强 |
| RMSNorm | 去掉均值中心化，只做方差归一化，比 LayerNorm 更快 |
| LoRA | 冻结原始权重，添加低秩分解 ΔW=BA，只训练 A 和 B |
| DPO | 直接从偏好数据学习，无需训练奖励模型 |
| RAG | 检索相关文档注入 prompt，减少幻觉 |
| ReAct | 推理(Thought)和行动(Action)交替进行的 Agent 范式 |
| Flash Attention | IO-aware 的 attention 实现，O(n) 显存，减少 HBM 访问 |
| GQA | Query 有 h 个 head，KV 只有 g 个 head (g<h)，节省 KV Cache |
| KV Cache | 缓存已计算的 K/V，避免重复计算，推理加速核心 |

### 常见面试追问链

```
面试官: 介绍一下你的项目架构
  → 为什么用 LLaMA-style？
    → Pre-Norm 和 Post-Norm 有什么区别？
    → SwiGLU 为什么比 ReLU 好？
    → RoPE 的原理是什么？

面试官: 你是怎么训练这个模型的？
  → 学习率怎么设置的？
    → 为什么需要 warmup？
    → Cosine decay 的好处是什么？
  → 混合精度训练怎么做的？
    → fp16 和 bf16 的区别？
    → Loss scaling 的原理？

面试官: RAG 系统怎么做的？
  → 检索用的什么方法？
    → Dense 和 Sparse 的区别？
    → 如何评估检索质量？
  → 文本怎么切分的？
    → Chunk size 怎么选？
    → Overlap 的作用？

面试官: Agent 系统怎么做的？
  → ReAct 的原理？
    → 和 Function Calling 的区别？
  → 记忆系统怎么设计的？
    → 短期和长期记忆的区别？
  → 错误怎么处理？
```

---

*本文档最后更新：2026-06-01*
*基于 LLM-Forge v1.0 代码库*
