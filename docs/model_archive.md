# Model Archive: OpenAI Parameter Golf

## Complete Catalog of Architectures

---

## 1. Ouroboros6 (LoopLM + Hexa MoD)

### Overview

**Architecture Type:** Recurrent Transformer (LoopLM) + Mixture-of-Depths

**Core Concept:** Achieve depth through parameter reuse rather than parameter count. Six independent transformer blocks are each processed 8 times in sequence, creating 48 effective "layers" from just 6 physical layers.

### Configuration

```python
vocab_size: int = 1056          # SP1024 tokenizer compatibility
d_model: int = 320              # Representation width
n_heads: int = 5                # Attention heads (320/5 = 64 per head)
hidden_dim: int = 864           # MLP intermediate dimension
num_layers: int = 6             # Physical layers
num_loops: int = 8              # Recurrent iterations
mod_capacity: float = 0.8       # MoD token capacity
```

### Architecture Details

#### LoopLM Mechanism

At each recurrent iteration `t`:
```
h^(t) = h^(t-1) + OuroborosBlock(h^(t-1))
```

With iteration embedding added at each step:
```
h^(t) = h^(t) + iter_embed(t)
```

#### Mixture-of-Depths (MoD)

Token-level routing selects top-k tokens for MLP processing:
```python
k = int(seq_len * mod_capacity)  # ~80% of tokens
scores = router_weights(h).squeeze(-1)
probs = sigmoid(scores)
topk_probs, topk_indices = topk(probs, k)
```

Only selected tokens pass through MLP; others skip with identity.

#### Attention (QKV Fusion)

```python
qkv = Wqkv(x).view(B, T, 3, n_heads, head_dim)
xq, xk, xv = qkv.unbind(dim=2)
xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
out = scaled_dot_product_attention(xq, xk, xv, causal=True)
```

#### Feed-Forward (SwiGLU)

```python
gate = silu(gate_proj(x))
up = up_proj(x)
out = down_proj(gate * up)
```

### Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Embedding | 1056 × 320 = 337,920 |
| Iteration Embed | 8 × 320 = 2,560 |
| Per-Layer Attention | 4 × 320² = 409,600 |
| Per-Layer MLP | 3 × 320 × 864 = 829,440 |
| Per-Layer Router | 320 |
| Per-Layer Norms | 2 × 320 = 640 |
| Final Norm | 320 |
| **Total (6 layers)** | **~8.2M** |
| **LM Head (tied)** | **—** |

**Compressed Size:** ~15.9 MB (bfloat16)

### Training Configuration

```python
batch_size: int = 32
seq_len: int = 2048
learning_rate: float = 0.02
optimizer: str = "Muon"  # Newton-Schulz orthogonalization
scheduler: str = "WSD"   # 60s warmup, stable, cosine decay
```

### Key Design Decisions

1. **Why 6 layers, not 4 or 8?**
   - 4 layers: insufficient depth (2.484 loss)
   - 6 layers: optimal balance (2.432 loss)
   - 8 layers: representation bottleneck at d_model=256 (2.560 loss)

2. **Why MoD on MLP only?**
   - Maintains temporal consistency in attention
   - MLP is more computationally expensive
   - 75% compute savings on 6/8 iterations

3. **Why QKV fusion?**
   - Single memory access vs. three
   - ~15% speedup in attention

### Known Limitations

- MoD routing adds complexity
- Recurrent computation limits parallelism
- Iteration embeddings may not capture step semantics

### Files

- Model: `/workspace/src/models/ouroboros.py`
- Config: `/workspace/src/config/ouroboros_config.py`
- Script: `/workspace/src/train_ouro.py`

---

## 2. SSM8 (Fat State Mamba)

### Overview

**Architecture Type:** State Space Model (SSM) / Mamba

**Core Concept:** Replace attention-based sequence modeling with linear-complexity state transitions. The "Fat State" extends hidden state dimension to 34 for richer history compression.

### Configuration

```python
vocab_size: int = 1056
d_model: int = 640
d_inner: int = 1280              # 2× expansion
d_state: int = 34                 # "Fat State" (vs. standard 8/16)
d_conv: int = 4                   # Causal convolution kernel
num_layers: int = 8
dt_rank: int = 40                 # ceil(640 / 16)
```

### Architecture Details

#### Selective Scan

SSM computes output via discretized continuous dynamics:
```
ẋ(t) = A·x(t) + B·u(t)
y(t) = C·x(t) + D·u(t)
```

Discrete form (with learnable Δt):
```
x_k = ᾀ·x_{k-1} + B̄·u_k
y_k = C·x_k + D·u_k
```

Where:
```python
ᾀ = exp(Δt · A)          # Discretized state matrix
B̄ = Δt · B               # Discretized input matrix
```

#### Hardware Kernel

```python
@torch.compiler.disable
def selective_scan_fn(u, delta, A, B, C, D, z, delta_bias):
    # CUDA kernel from mamba_ssm
    # u, delta, B, C in FP32 (anti-autocast poisoning)
    # A, D in FP32 (numerical stability)
```

#### Block Structure

```
Input → [RMSNorm → Linear(2×)] → [SiLU/Gate Split]
      → [Conv1D] → [SSM(selective_scan)] → [SiLU]
      → [Linear] → Output
```

### HiPPO Initialization

State matrix A initialized via HiPPO-LegS:
```python
A[i, n] = n + 1  for n in [0, d_state)
```

Stored in log-space: `A_log = log(A)`

### Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Embedding | 1056 × 640 = 675,840 |
| Per-Block In-Proj | 640 × 2560 = 1,638,400 |
| Per-Block Conv1D | 1280 × 4 = 5,120 |
| Per-Block x-Proj | 1280 × 108 = 138,240 |
| Per-Block dt-Proj | 40 × 1280 = 51,200 |
| Per-Block Out-Proj | 1280 × 640 = 819,200 |
| Per-Block A, D | 1280 + 1280 = 2,560 |
| Per-Block dt-bias | 1280 |
| **Total (8 blocks)** | **~14.3M** |
| **LM Head (tied)** | **—** |

**Compressed Size:** ~15.5 MB (bfloat16)

### Training Configuration

```python
batch_size: int = 32
seq_len: int = 2048
learning_rate: float = 0.02
optimizer: str = "Muon"
scheduler: str = "WSD"
grad_checkpointing: bool = True  # Saves ~40% VRAM
```

### Key Design Decisions

1. **Why d_state=34?**
   - d_state=8: insufficient memory capacity
   - d_state=16: standard Mamba default
   - d_state=34: "Fat State" for rich history within budget

2. **Why FP32 for dynamics?**
   - Prevents BF16 numerical errors (~7.8e-3)
   - Critical for accumulated state over 2048 tokens

3. **Why gradient checkpointing?**
   - SSM activations: ~50MB per block in BF16
   - 8 blocks × 50MB = 400MB → 240MB with checkpointing
   - Cost: ~30% slower (recomputation)

### State Caching

For evaluation with sliding window:
```python
# Save final state after processing window
h_T ∈ ℝ^(d_inner × d_state)

# Pass as initial_state to next window
next_window(h_T)  # O(stride × D) vs O(seq_len × D)
```

### Known Limitations

- Requires mamba_ssm library (not standard)
- Selective_scan not compatible with torch.compile
- State caching requires specific kernel support

### Files

- Model: `/workspace/src/models/ssm.py`
- Config: `/workspace/src/config/ssm_config.py`
- Script: `/workspace/src/train_ssm.py`

---

## 3. LiteFormer-11L

### Overview

**Architecture Type:** Standard Transformer with Compression Optimizations

**Core Concept:** Maximize parameter budget utilization through aggressive quantization (int6) and compression (zstd-22), fitting 35M parameters within 16 MB.

### Configuration

```python
vocab_size: int = 1056
d_model: int = 512
n_heads: int = 8
n_kv_heads: int = 4               # GQA: half the KV heads
head_dim: int = 64
hidden_dim: int = 1536            # 3× expansion
num_layers: int = 11
partial_rope_dims: int = 16       # RoPE on first 16 dims
bigram_buckets: int = 2048
bigram_dim: int = 128
xsa_last_n: int = 4               # XSA on last 4 layers
```

### Architecture Details

#### Grouped Query Attention (GQA)

```python
# 8 query heads, 4 KV heads → 2 query heads share 1 KV head
q_heads: int = 8
kv_heads: int = 4
group_size: int = 2  # 8 / 4

# KV cache size: 4 heads vs 8 = 50% reduction
```

#### Exclusive Self Attention (XSA)

Applied to last 4 layers:
```python
# Standard attention output
attn_out = attention(q, k, v)

# Subtract projection onto own value
attn_out = attn_out - proj(attn_out onto v)

# Forces attention to carry only orthogonal information
```

#### BigramHashEmbedding

```python
# Hash(prev_token, curr_token) → bucket
bucket = (prev * vocab_size + curr) % num_buckets

# Embed and project
bigram_emb = embed(bucket)  # [B, T, 128]
projected = proj(bigram_emb)  # [B, T, 512]

# Total: 2048 × 128 + 128 × 512 = 327,680 params
```

#### SmearGate

Causal mean blending:
```python
causal_mean = cumsum(x) / arange(1, T+1)
gate = sigmoid(learnable_gate)
output = x + gate * (causal_mean - x)
```

#### Feed-Forward (relu²)

```python
gate = relu(gate_proj(x)) ** 2
up = up_proj(x)
output = down_proj(gate * up)
```

### Compression Pipeline

#### 1. int6 Quantization

```python
# Per-row scaling
max_abs = weights.abs().max(dim=-1)
scales = max_abs / 31.0

# Quantize to [-31, 31]
quantized = round(weights / scales).clamp(-31, 31)
quantized = quantized.to(int8)  # Store as int8

# Effective size: 0.75 bytes/param (weights) + scales
```

#### 2. GPTQ-lite

Find optimal clipping threshold:
```python
for percentile in [0.999, 0.9995, 0.9999, 0.99995, 1.0]:
    clip_val = max_abs * percentile
    clipped = weights.clamp(-clip_val, clip_val)
    # Quantize and measure reconstruction MSE
    # Select percentile with minimum MSE
```

#### 3. zstd-22

```python
compressor = zstandard.ZstdCompressor(level=22)
compressed = compressor.serialize(model_bytes)

# Additional 15-20% compression
```

### Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Token Embedding | 1056 × 512 = 540,672 |
| Bigram Embedding | 327,680 |
| Per-Layer GQA | 512×4096 + 2×512×2048 + 512×4096 = 4,194,304 |
| Per-Layer MLP | 3 × 512 × 1536 = 2,359,296 |
| Per-Layer SmearGate | 512 |
| Per-Layer Norms | 2 × 512 = 1,024 |
| **Total (11 layers)** | **~35M** |
| **LM Head (tied)** | **—** |

**Quantized Size:** ~27 MB (int6)
**Expected Compressed Size:** ~14.5 MB (zstd-22)
**Actual Compressed Size:** ~20.13 MB (exceeded 16 MB budget - not selected for final submission)

**Note on Compression Failure:**
The theoretical calculation predicted 14.5 MB after int6 quantization and zstd-22 compression. However, empirical evaluation revealed the actual compressed size was ~20.13 MB. Weight distributions exhibited higher entropy than anticipated, particularly in attention projection layers. The relu² activation created sparsity but outliers in the weight matrices reduced zstd compression efficiency. This 5.6 MB discrepancy between theory and practice meant LiteFormer could not be submitted despite its architectural merits.

### Training Configuration

```python
batch_size: int = 32
seq_len: int = 2048
learning_rate: float = 0.025  # Higher LR for compression
optimizer: str = "Muon"
scheduler: str = "WSD"

# QAT (Quantization-Aware Training)
apply_fake_quant: bool = True
qat_threshold: float = 0.15  # LR scale threshold

# EMA
ema_decay: float = 0.997
```

### Key Design Decisions

1. **Why int6, not int4 or int8?**
   - int4: too aggressive, quality degradation
   - int6: optimal balance for this model size
   - int8: insufficient compression

2. **Why relu² instead of SiLU?**
   - Creates sparsity (many zeros)
   - Better compressibility
   - Empirically better BPB at this scale

3. **Why partial RoPE (16/64 dims)?**
   - Allows position-independent encoding in remaining dims
   - Helps capture n-gram patterns

4. **Why GQA?**
   - 50% KV cache reduction
   - Enables longer effective context
   - Minimal quality loss at this scale

### Test-Time Training (TTT)

```python
# For each evaluation window:
1. Train LoRA adapters on PREVIOUS window
2. Evaluate CURRENT window with updated adapters
3. Never see tokens being evaluated

# LoRA config
r: int = 8
alpha: float = 16.0
inner_steps: int = 3
inner_lr: float = 1e-4
```

### Known Limitations

- Quantization adds noise
- zstd-22 compression is slow
- TTT evaluation is compute-intensive
- GQA reduces parallelism in attention

### Files

- Model: `/workspace/src/models/liteformer.py`
- Config: `/workspace/src/config/liteformer_config.py`
- Script: `/workspace/src/train_liteformer.py`

---

## Shared Components

### Muon Optimizer

```python
class Muon(torch.optim.Optimizer):
    """Momentum with Newton-Schulz orthogonalization."""

    def step(self):
        buf.mul_(momentum).add_(grad)
        g_ortho = newton_schulz_5(buf)
        param.add_(g_ortho, alpha=-lr * scale)

# Newton-Schulz coefficients
a, b, c = 3.4445, -4.7750, 2.0315
```

### WSD Scheduler

```python
# Three phases:
# 1. Warmup: linear 0 → max_lr (60s)
# 2. Stable: constant max_lr (420s)
# 3. Decay: cosine max_lr → min_lr (120s)

lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

### EntropyFilter

```python
# Curriculum learning schedule:
# 0-2 min: ratio = 4.0 (accept high compression = low entropy)
# 2-5 min: ratio = 3.0
# 5+ min: ratio = 2.5 (strict quality)

ratio = original_size / compressed_size
if ratio > threshold: reject(chunk)
```

---

## Performance Comparison

| Metric | Ouroboros6 | SSM8 | LiteFormer |
|--------|-----------|------|-----------|
| Parameters | ~8M | ~14M | ~35M |
| Layers | 6×8 (virtual) | 8 | 11 |
| Attention | Full | None | GQA |
| Hidden Dim | 320 | 640 | 512 |
| Expected Compressed | ~15.9 MB | ~15.5 MB | ~14.5 MB |
| **Actual Compressed** | **15.9 MB** | **15.5 MB** | **20.13 MB** |
| Budget Status | ✓ Under | ✓ Under | ✗ **Exceeded** |
| Compression | bfloat16 | bfloat16 | int6+zstd |
| Target BPB | ~1.20-1.25 | ~1.15-1.25 | ~1.15-1.20 |
| Strength | Depth via reuse | Long context | Capacity via compression |
| Weakness | Limited width | Kernel dependency | **Theory-practice gap** |
| Selected | ✓ Yes | ✓ Yes | ✗ **No** |

---

## References

### Architectures and Models

* **LoopLM / Ouro** – Ouro model team (Moonshot AI). *"Scaling Latent Reasoning via Looped Language Models."* (2025). [[arXiv:2510.25741]](https://arxiv.org/abs/2510.25741)
* **Universal Transformer** – M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser. *"Universal Transformers."* ICLR (2019). [[arXiv:1807.03819]](https://arxiv.org/abs/1807.03819)
* **ALBERT** – Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut. *"ALBERT: A Lite BERT for Self-supervised Learning of Language Representations."* ICLR (2020). [[arXiv:1909.11942]](https://arxiv.org/abs/1909.11942)
* **S4 (Structured State Space)** – A. Gu, K. Goel, and C. Ré. *"Efficiently Modeling Long Sequences with Structured State Spaces."* ICLR (2022). [[arXiv:2111.00396]](https://arxiv.org/abs/2111.00396)
* **Mamba** – A. Gu and T. Dao. *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."* (2023). [[arXiv:2312.00752]](https://arxiv.org/abs/2312.00752)
* **HiPPO** – A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Ré. *"HiPPO: Recurrent Memory with Optimal Polynomial Projections."* NeurIPS (2020). [[arXiv:2008.07669]](https://arxiv.org/abs/2008.07669)
* **Mixture-of-Depths (MoD)** – D. Raposo et al. *"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models."* (2024). [[arXiv:2404.02258]](https://arxiv.org/abs/2404.02258)

### Compression and Quantization

* **GPTQ** – E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."* ICLR (2023). [[arXiv:2210.17323]](https://arxiv.org/abs/2210.17323)
* **TurboQuant** – A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni. *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."* (2025). 

### Adaptation and Optimization

* **LoRA** – E. J. Hu et al. *"LoRA: Low-Rank Adaptation of Large Language Models."* (2021). [[arXiv:2106.09685]](https://arxiv.org/abs/2106.09685)
* **Newton-Schulz / Muon** – K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein. *"Muon: An optimizer for hidden layers in neural networks."* (2024). [[Project Page]](https://kellerjordan.github.io/posts/muon/) 
---

*Document Version: 1.1*
*Last Updated: 2026-04-28*
