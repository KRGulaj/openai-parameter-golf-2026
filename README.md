# OpenAI Parameter Golf Archive

**My First AI Engineering Project: A Learning Journey**

---

## Overview

This repository contains my first-ever attempt at building AI systems. I created it for the OpenAI Parameter Golf competition—a challenge to train the best language model possible within a strict 16 MB parameter budget and 10-minute training time limit. What you're seeing is not the work of an experienced practitioner, but rather an honest chronicle of my learning process.

I experimented with three different architectural approaches, each teaching me something new about efficiency, constraints, and the nature of language modeling. This codebase represents my journey from naive initial attempts to increasingly sophisticated solutions.

---

## The Competition

The Parameter Golf competition imposed severe constraints:
- **Memory Budget:** 16 MB total for code + model weights combined
- **Training Time:** Exactly 10 minutes on 8× NVIDIA H100 GPUs
- **Evaluation Time:** Another 10 minutes for test-time adaptation
- **Metric:** Bits Per Byte (BPB) on held-out text

These constraints forced me to think deeply about what actually matters in model design.

---

## My Three Approaches

### Approach 1: Ouroboros (The LoopLM Era)

**My thinking:** If I can't add more parameters, why not reuse the same ones multiple times?

I built a recurrent transformer called Ouroboros where 6 independent layers are processed 8 times each, creating 48 effective "layers" from just 6 physical ones. I combined this with Mixture-of-Depths (MoD) routing—selectively processing only 80% of tokens through the MLP layers.

**What I learned:**
- A single layer processed 8 times (Ouroboros1) was insufficient
- Six layers processed 8 times (Ouroboros6) achieved optimal balance
- Eight layers (Ouroboros8) suffered from representation bottlenecks—d_model=256 was too narrow

**Configuration:**
- 6 layers × 8 iterations = 48 effective transformer layers
- ~8M parameters in bfloat16
- ~15.9 MB compressed

**My mistakes:** Initially I thought more loops would automatically mean better performance. I learned that representation width matters as much as depth.

---

### Approach 2: SSM (The State Space Pivot)

**My thinking:** Transformers have O(n²) attention complexity. State Space Models achieve O(n) with fixed-size hidden states.

I adopted the Mamba framework with what I call "Fat State"—extending the hidden state dimension to 34 (vs. standard 8 or 16). This provides superior long-range memory without additional parameters.

**Key decisions I made:**
- d_state=34 for richer history compression
- HiPPO initialization ensures optimal memory structure from the start
- FP32 precision for SSM dynamics parameters (A, D, dt) to prevent numerical instability

**Configuration:**
- 8 SSM layers with d_state=34
- ~14M parameters
- ~15.5 MB compressed

**What surprised me:** The "Fat State" design was more effective than I expected. The model could retain long-range dependencies that my transformer struggled with.

---

### Approach 3: LiteFormer (The Compression Maximizer)

**My thinking:** The 16 MB limit applies to *compressed* size, not raw parameters. With int6 quantization and zstd-22, I could fit 35M parameters.

I abandoned recurrence in favor of a standard Transformer with aggressive compression. The key insight from leaderboard analysis: models with 35-40M parameters compressed to ~15.5 MB consistently outperformed raw bfloat16 models.

**My innovations:**
- **Grouped Query Attention (GQA):** 8 query heads share 4 KV heads (50% cache reduction)
- **relu² Activation:** Creates sparsity for better compressibility
- **BigramHashEmbedding:** Captures local pair statistics (~327K parameters)
- **Exclusive Self Attention (XSA):** Forces attention to carry only orthogonal information

**Compression Pipeline I developed:**
1. **int6 Quantization:** Per-row scaling, values in [-31, 31]
2. **GPTQ-lite:** Optimal clipping threshold search (5 percentiles)
3. **zstd-22:** Additional 15-20% compression
4. **Expected:** 70 MB → 27 MB → 14.5 MB

**The Reality:**
- **Actual compressed size:** ~20.13 MB (not 14.5 MB)
- **Why it failed:** The compression ratio did not match theoretical expectations. Weight distributions had more entropy than anticipated, preventing zstd from achieving the projected compression.
- **The lesson:** Theoretical calculations don't always match empirical results. This is why LiteFormer was **not chosen** for my final submission.

**Configuration:**
- 11 standard Transformer layers
- ~35M parameters → ~20.13 MB after compression (exceeds 16 MB budget)

---

## The Evaluation: Score-First TTT

A crucial part of the competition I initially overlooked: the 10-minute evaluation phase. I developed a methodology called **Score-First Test-Time Training (TTT)**:

**The Principle:**
- Train LoRA adapters on the PREVIOUS context window
- Evaluate on the CURRENT window with updated weights
- Never see the tokens being evaluated

**Why this matters:**
- Traditional evaluation sees test tokens before predicting them
- Score-First TTT is the only valid form: learn from already-evaluated tokens only
- Sliding window with stride=64 provides 1984 tokens of effective context

**My LoRA Configuration:**
- Rank r=8, alpha=16.0
- Applied to attention projections (Q, K, V, O) for transformers
- Applied to in_proj, out_proj for SSM
- Learning rate: 3e-3 with AdamW
- 3 gradient steps per window when loss > threshold

**Temperature Adaptation:**
- Learnable temperature τ initialized to 1.0
- Optimized during TTT, clamped to [0.8, 2.0]
- Compensates for distribution shift

---

## Optimization Tricks I Learned

### Muon Optimizer with Newton-Schulz

I implemented a custom optimizer using 5-step Newton-Schulz iteration for gradient orthogonalization:

```python
# Coefficients: a=3.4445, b=-4.7750, c=2.0315
X = G / ||G||
for _ in range(5):
    A = X @ X.T
    B = b*A + c*A@A
    X = a*X + B@X
```

**Why it matters:** Prevents gradient norm explosion in deep recurrent models. Scale correction by aspect ratio ensures stable updates.

### Time-Based WSD Scheduler

I abandoned step-based scheduling for time-based control:
- **Warmup:** Linear 0 → max_lr (10% of time)
- **Stable:** Constant max_lr (70% of time)
- **Decay:** Cosine to min_lr (20% of time)

**Why this matters:** Eliminates sensitivity to JIT compilation variability and "Noisy Neighbors" in cloud infrastructure.

### Memory Optimizations

**RMSNorm:**
- Cheaper than LayerNorm (no mean centering)
- Prevents layer magnitude divergence

**Fused QKV Projection:**
- Single linear layer: d_model → 3*d_model
- Reduces memory accesses by 3×

**Weight Tying:**
- Embedding shared with LM head
- Saves ~540K parameters for vocab=1056, d_model=512

**Gradient Checkpointing (SSM8):**
- Saves ~40% VRAM
- Critical for 8× H100 with large d_state

### Data Pipeline Optimizations

**Zero-Copy Memmap:**
```python
np.memmap(file, dtype=np.uint16, mode='r')
```
- Disk-to-memory mapping without Python buffer copies

**GPU Casting:**
- uint8 from disk → int64 on GPU
- Maximizes PCIe bandwidth

**Pinned Memory:**
```python
tensor.pin_memory(device=torch.device("cuda:0"))
```
- Enables async GPU transfer

### Compilation Optimizations

**torch.compile:**
```python
torch.compile(model, mode="max-autotune")
```
- 20-30% throughput improvement after warmup
- Excludes selective_scan_fn via @torch.compiler.disable

**TF32 on H100:**
```python
torch.backends.cuda.matmul.allow_tf32 = True
```
- Faster matmul with minimal precision loss

---

## Repository Structure

```
/repo/
├── README.md
├── src/                          # Clean modular source code
│   ├── config/                   # Configuration dataclasses
│   ├── models/                   # Model architectures
│   ├── data/                     # Data loading
│   ├── training/                 # Training utilities
│   ├── utils/                    # Helpers
│   ├── train_ouro.py             # Monolithic Ouroboros6
│   ├── train_ssm.py              # Monolithic SSM8
│   └── train_liteformer.py       # Monolithic LiteFormer
├── tests/                        # Unit tests
└── docs/                         # Documentation
    ├── Parameter_GOLF_report.pdf # This file
    └── model_archive.md          # Detailed catalog

```

---

## Quick Start

### Installation

```bash
pip install torch>=2.0.0 numpy sentencepiece zstandard safetensors

# For SSM8 (optional)
pip install mamba-ssm causal-conv1d
```

### Training

```bash
# My Ouroboros6
python src/train_ouro.py

# My SSM8
python src/train_ssm.py

# My LiteFormer
python src/train_liteformer.py
```

### Environment Variables

```bash
export DATA_PATH="data.bin"
export TOTAL_SECONDS=600          # Training budget
export MAX_EVAL_TIME=600        # Evaluation budget
export BATCH_SIZE=32
export SEQ_LEN=2048
export LEARNING_RATE=0.02
```

---

## Performance Summary

| My Model | Parameters | Expected | Actual | Budget | Key Strength |
|----------|-----------|----------|--------|--------|--------------|
| Ouroboros6 | ~8M | ~15.9 MB | 15.9 MB | ✓ Pass | Depth via recurrence |
| SSM8 | ~14M | ~15.5 MB | 15.5 MB | ✓ Pass | Long context via structure |
| LiteFormer | ~35M | ~14.5 MB | **20.13 MB** | ✗ **Fail** | Theory-practice gap |

---

## Lessons from My First Project

### What I Did Right

1. **Iterated quickly:** Each model built on the previous
2. **Analyzed failures:** Ouroboros8's bottleneck taught me about width vs depth
3. **Read the leaderboard:** Compression-aware design came from studying top entries
4. **Time-based control:** Eliminated step sensitivity

### What I Did Wrong

1. **Initially ignored evaluation:** Almost missed the TTT requirement
2. **Started with byte-level tokens:** Incompatible with competition data
3. **Thought more loops = better:** Learned about representation bottlenecks
4. **Used step-based scheduling initially:** Failed due to JIT variability

### What I Would Do Differently

1. Start with leaderboard analysis, not architecture-first
2. Implement time-based control from day one
3. Test quantization compatibility earlier
4. Use structured logging instead of print statements

---

## Academic Context

My work builds on several key papers:
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

See my technical report in `docs/` for full citations.

---

## Acknowledgments

I would like to thank the authors of the foundational papers (LoopLM, Mamba, Muon) for open-sourcing their insights, and the organizers of the OpenAI Parameter Golf for providing such a unique, forcing-function constraint that accelerated my learning.

*Last Updated: 2026-04-28*
