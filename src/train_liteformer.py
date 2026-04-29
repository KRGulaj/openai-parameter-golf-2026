"""
LiteFormer-11L: Compressed Transformer for Parameter Golf
Parameter Golf Submission Script

Architecture:
- 11-layer standard Transformer with GQA (8 query / 4 KV heads)
- 3x MLP expansion with relu² activation
- Exclusive Self Attention (XSA) on last 4 layers
- SmearGate + BigramHashEmbedding
- int6 quantization + zstd-22 compression

Time Budget: 600 seconds
Memory Budget: ~14.5 MB compressed (35M params -> int6 + zstd)
"""

import os
import sys
import io
import time
import math
import threading
import queue
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
import zstandard as zstd


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LiteFormerConfig:
    """Configuration for LiteFormer-11L."""
    vocab_size: int = 1056
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 64
    hidden_dim: int = 1536
    num_layers: int = 11
    rope_theta: float = 10000.0
    partial_rope_dims: int = 16
    xsa_last_n: int = 4
    bigram_buckets: int = 2048
    bigram_dim: int = 128
    logit_softcap: float = 30.0
    eps: float = 1e-6


# Training config
TOTAL_SECONDS: float = 600.0
WARMUP_SECONDS: float = 60.0
BATCH_SIZE: int = 32
SEQ_LEN: int = 2048
LEARNING_RATE: float = 0.025


# =============================================================================
# DATA LOADING
# =============================================================================

class EntropyFilter:
    """Dynamic data filter using zlib compression ratios."""

    def __init__(self, initial_ratio: float = 4.0):
        self.current_ratio = initial_ratio
        self.stats_accepted = 0
        self.stats_rejected = 0

    def update_threshold_by_time(self, elapsed_minutes: float) -> None:
        if elapsed_minutes >= 5.0:
            self.current_ratio = 2.5
        elif elapsed_minutes >= 2.0:
            self.current_ratio = 3.0
        else:
            self.current_ratio = 4.0

    def is_valid(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        raw_bytes = chunk.tobytes()
        compressed = __import__('zlib').compress(raw_bytes, level=1)
        ratio = len(raw_bytes) / max(len(compressed), 1)
        if ratio > self.current_ratio:
            self.stats_rejected += 1
            return False
        self.stats_accepted += 1
        return True


class FastGolfDataLoader:
    """High-performance data loader with memmap and async prefetching."""

    def __init__(self, bin_path, batch_size, seq_len, entropy_filter, rank=0, world_size=1):
        self.bin_path = bin_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.entropy_filter = entropy_filter
        self.rank = rank
        self.world_size = world_size
        self.device_id = int(os.environ.get("LOCAL_RANK", "0"))
        self.chunk_size = batch_size * (seq_len + 1)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        self.last_good_chunk = None
        self.batch_queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        ptr = self.rank * self.chunk_size
        while not self.stop_event.is_set():
            if ptr + self.chunk_size > self.total_tokens:
                ptr = self.rank * self.chunk_size
            chunk = np.array(self.data[ptr:ptr + self.chunk_size], dtype=np.uint16)
            ptr += self.world_size * self.chunk_size
            if self.entropy_filter is not None:
                if self.entropy_filter.is_valid(chunk):
                    self.last_good_chunk = chunk
                else:
                    if self.last_good_chunk is not None:
                        chunk = self.last_good_chunk
                    else:
                        continue
            tensor = torch.from_numpy(chunk.astype(np.uint8)).view(self.batch_size, self.seq_len + 1)
            pinned = tensor.pin_memory(device=torch.device(f"cuda:{self.device_id}"))
            self.batch_queue.put(pinned)

    def get_batch(self, device):
        pinned = self.batch_queue.get()
        X = pinned[:, :-1].to(device, dtype=torch.long, non_blocking=True)
        Y = pinned[:, 1:].to(device, dtype=torch.long, non_blocking=True)
        return X, Y

    def stop(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=1.0)


# =============================================================================
# COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def precompute_rope(rope_dims, max_len, theta=10000.0):
    """Precompute RoPE frequencies."""
    half = rope_dims // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / rope_dims))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(xq, xk, cos, sin, rope_dims):
    """Apply RoPE to first rope_dims of each head."""
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    xq_r, xq_p = xq[..., :rope_dims], xq[..., rope_dims:]
    xk_r, xk_p = xk[..., :rope_dims], xk[..., rope_dims:]
    xq_r = xq_r * cos + rotate_half(xq_r) * sin
    xk_r = xk_r * cos + rotate_half(xk_r) * sin
    return (torch.cat([xq_r, xq_p], dim=-1), torch.cat([xk_r, xk_p], dim=-1))


# =============================================================================
# OPTIMIZER
# =============================================================================

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration for gradient orthogonalization."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalization."""

    def __init__(self, params, lr=0.025, momentum=0.95, weight_decay=0.04):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                g = p.grad
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mu).add_(g)
                g_ortho = zeropower_via_newtonschulz5(buf)
                scale = max(1.0, g_ortho.shape[0] / g_ortho.shape[1]) ** 0.5
                p.data.add_(g_ortho, alpha=-lr * scale)


# =============================================================================
# SPECIAL MODULES
# =============================================================================

class SmearGate(nn.Module):
    """Causal mean gate."""

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        T = x.shape[1]
        denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        causal_mean = x.cumsum(dim=1) / denom
        g = torch.sigmoid(self.gate)
        return x + g * (causal_mean - x)


class BigramHashEmbedding(nn.Module):
    """Hash-based bigram embedding."""

    def __init__(self, num_buckets, bigram_dim, d_model):
        super().__init__()
        self.num_buckets = num_buckets
        self.bigram_embed = nn.Embedding(num_buckets, bigram_dim)
        self.proj = nn.Linear(bigram_dim, d_model, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        prev_ids = torch.cat([
            torch.zeros(batch_size, 1, dtype=input_ids.dtype, device=input_ids.device),
            input_ids[:, :-1]
        ], dim=1)
        hash_val = (prev_ids * 1056 + input_ids) % self.num_buckets
        bigram_emb = self.bigram_embed(hash_val)
        return self.proj(bigram_emb)


# =============================================================================
# ATTENTION AND MLP
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """GQA with 8 query / 4 KV heads."""

    def __init__(self, config: LiteFormerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.n_heads // config.n_kv_heads

        self.norm = RMSNorm(config.d_model, config.eps)
        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, x, cos, sin, rope_dims, use_xsa=False):
        batch_size, seq_len, _ = x.shape
        h = self.norm(x)

        q = self.wq(h).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(h).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(h).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = apply_partial_rope(q, k, cos, sin, rope_dims)

        # Expand KV heads
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        if use_xsa:
            # Exclusive Self Attention
            dot = (out * v).sum(dim=-1, keepdim=True)
            v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-6
            proj = dot * v / v_norm_sq
            out = out - proj

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out)


class ReluSquaredMLP(nn.Module):
    """MLP with relu² activation."""

    def __init__(self, config: LiteFormerConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.eps)
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=False)

    def forward(self, x):
        h = self.norm(x)
        gate = F.relu(self.gate_proj(h)) ** 2
        up = self.up_proj(h)
        return self.down_proj(gate * up)


# =============================================================================
# MODEL
# =============================================================================

class LiteFormerModel(nn.Module):
    """LiteFormer-11L."""

    def __init__(self, config: LiteFormerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.bigram_embed = BigramHashEmbedding(config.bigram_buckets, config.bigram_dim, config.d_model)

        self.attn_layers = nn.ModuleList([GroupedQueryAttention(config) for _ in range(config.num_layers)])
        self.mlp_layers = nn.ModuleList([ReluSquaredMLP(config) for _ in range(config.num_layers)])
        self.smeargates = nn.ModuleList([SmearGate(config.d_model) for _ in range(config.num_layers)])
        self.norms = nn.ModuleList([RMSNorm(config.d_model, config.eps) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model, config.eps)

        freqs_cos, freqs_sin = precompute_rope(config.partial_rope_dims, 4096)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids) + self.bigram_embed(input_ids)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer_idx in range(self.config.num_layers):
            x_norm = self.norms[layer_idx](x)
            use_xsa = layer_idx >= (self.config.num_layers - self.config.xsa_last_n)
            attn_out = self.attn_layers[layer_idx](x_norm, freqs_cos, freqs_sin, self.config.partial_rope_dims, use_xsa)
            x = x + attn_out
            x = self.smeargates[layer_idx](x)
            mlp_out = self.mlp_layers[layer_idx](x)
            x = x + mlp_out

        return self.final_norm(x)


class LiteFormerForCausalLM(nn.Module):
    """Complete LiteFormer with LM head."""

    def __init__(self, config: LiteFormerConfig):
        super().__init__()
        self.config = config
        self.model = LiteFormerModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        hidden = self.model(input_ids)
        logits = self.lm_head(hidden)

        # Logit soft-capping
        if self.config.logit_softcap > 0:
            logits = torch.tanh(logits / self.config.logit_softcap) * self.config.logit_softcap

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return logits, loss


# =============================================================================
# COMPRESSION
# =============================================================================

def quantize_int6(weights):
    """Quantize weights to int6 with per-row scaling."""
    max_abs = weights.abs().max(dim=-1, keepdim=True)[0]
    scales = max_abs / 31.0
    scales = scales.clamp(min=1e-8)
    quantized = torch.round(weights / scales).clamp(-31, 31).to(torch.int8)
    return quantized, scales.squeeze().to(torch.bfloat16)


def compress_model(model):
    """Compress model with int6 + zstd-22."""
    state_dict = {}
    for name, tensor in model.state_dict().items():
        if tensor.dtype in [torch.float32, torch.bfloat16, torch.float16] and tensor.ndim >= 2:
            quantized, scales = quantize_int6(tensor)
            state_dict[name] = {
                "type": "int6",
                "data": quantized.numpy(),
                "scales": scales.numpy(),
            }
        else:
            state_dict[name] = {
                "type": "raw",
                "data": tensor.numpy(),
            }

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{
        k: v["data"] if isinstance(v, dict) else v
        for k, v in state_dict.items()
    })

    compressor = zstd.ZstdCompressor(level=22)
    compressed = compressor.compress(buffer.getvalue())

    return compressed, len(compressed)


# =============================================================================
# SCHEDULER
# =============================================================================

class WSDScheduler:
    """Warmup-Stable-Decay scheduler."""

    def __init__(self, optimizer, max_lr, total_seconds, warmup_seconds=60.0, min_lr=1e-6, decay_fraction=0.2):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_seconds = warmup_seconds
        self.stable_end = total_seconds * (1.0 - decay_fraction)
        self.decay_end = total_seconds
        self.current_lr = 0.0

    def step(self, elapsed_seconds):
        t = min(elapsed_seconds, self.decay_end)
        if t < self.warmup_seconds:
            lr = self.max_lr * (t / self.warmup_seconds)
        elif t < self.stable_end:
            lr = self.max_lr
        else:
            decay_progress = (t - self.stable_end) / (self.decay_end - self.stable_end)
            decay_progress = min(decay_progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        self.current_lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LiteFormerConfig()
    model = LiteFormerForCausalLM(config).to(device)
    model = torch.compile(model, mode="max-autotune")

    # Optimizer and scheduler
    optimizer = Muon(model.parameters(), lr=LEARNING_RATE)
    scheduler = WSDScheduler(optimizer, LEARNING_RATE, TOTAL_SECONDS, WARMUP_SECONDS)

    # Data loading
    bin_path = os.environ.get("DATA_PATH", "data/train.bin")
    if not Path(bin_path).exists():
        print(f"Data file not found: {bin_path}")
        return

    entropy_filter = EntropyFilter(initial_ratio=4.0)
    dataloader = FastGolfDataLoader(bin_path, BATCH_SIZE, SEQ_LEN, entropy_filter)

    # Cold start compilation
    print("Compiling model...")
    with torch.no_grad():
        dummy = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
        _ = model(dummy)
    torch.cuda.synchronize()

    # Training
    print(f"Training for {TOTAL_SECONDS}s...")
    start_time = time.perf_counter()
    model.train()
    step = 0
    total_loss = 0.0

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= TOTAL_SECONDS:
                break

            # Update learning rate
            scheduler.step(elapsed)

            # Update entropy filter
            entropy_filter.update_threshold_by_time(elapsed / 60.0)

            # Get batch
            X, Y = dataloader.get_batch(device)

            # Training step
            optimizer.zero_grad()
            logits, loss = model(X, labels=Y)

            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            step += 1

            if step % 10 == 0:
                avg_loss = total_loss / step
                print(f"Step {step} | Time: {elapsed:.1f}s | Loss: {avg_loss:.4f} | LR: {scheduler.current_lr:.6f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        dataloader.stop()

    # Save model
    print("Saving model...")
    model.eval()
    state_dict = model.state_dict()
    save_file(state_dict, "model.safetensors")

    # Compress and save
    print("Compressing model...")
    compressed, size = compress_model(model)
    with open("model.zstd", "wb") as f:
        f.write(compressed)
    print(f"Compressed size: {size / (1024*1024):.2f} MB")
    print("Training complete!")


if __name__ == "__main__":
    train()
