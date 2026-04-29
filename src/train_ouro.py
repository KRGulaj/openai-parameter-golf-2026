"""
Ouroboros6: Cascaded 6-Layer LoopLM with Hexa Mixture-of-Depths
Parameter Golf Submission Script

Architecture:
- 6 independent transformer blocks with unique parameters
- 8 recurrent iterations through the stack
- Mixture-of-Depths routing (80% token capacity per MLP)
- RoPE positional encoding with NTK-aware scaling
- Muon optimizer with Newton-Schulz 5-step orthogonalization

Time Budget: 600 seconds
Memory Budget: ~15.9 MB in bfloat16
"""

import os
import sys
import time
import math
import threading
import queue
from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OuroborosConfig:
    """Configuration for Ouroboros6 model."""
    vocab_size: int = 1056
    tie_weights: bool = True
    d_model: int = 320
    n_heads: int = 5
    hidden_dim: int = 864
    num_layers: int = 6
    num_loops: int = 8
    mod_capacity: float = 0.8
    dropout: float = 0.0
    eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 2048


# Training config
TOTAL_SECONDS: float = 600.0
WARMUP_SECONDS: float = 60.0
BATCH_SIZE: int = 32
SEQ_LEN: int = 2048
LEARNING_RATE: float = 0.02


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

    def __init__(
        self,
        bin_path: str,
        batch_size: int,
        seq_len: int,
        entropy_filter: Optional[EntropyFilter],
        rank: int = 0,
        world_size: int = 1
    ):
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
        self.last_good_chunk: Optional[np.ndarray] = None
        self.batch_queue: queue.Queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self) -> None:
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

    def get_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pinned = self.batch_queue.get()
        X = pinned[:, :-1].to(device, dtype=torch.long, non_blocking=True)
        Y = pinned[:, 1:].to(device, dtype=torch.long, non_blocking=True)
        return X, Y

    def stop(self) -> None:
        self.stop_event.set()
        self.worker_thread.join(timeout=1.0)


# =============================================================================
# COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE frequency tables."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings."""
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    xq_out = (xq * freqs_cos) + (rotate_half(xq) * freqs_sin)
    xk_out = (xk * freqs_cos) + (rotate_half(xk) * freqs_sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# OPTIMIZER
# =============================================================================

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
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

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.04):
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
# MODEL
# =============================================================================

class OuroborosAttention(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, config: OuroborosConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.norm = RMSNorm(config.d_model, config.eps)
        self.Wqkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self.norm(x)
        qkv = self.Wqkv(h).view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        xq, xk, xv = qkv.unbind(dim=2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        out = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out)


class OuroborosMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: OuroborosConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.eps)
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate = F.silu(self.gate_proj(h)) * self.up_proj(h)
        return self.down_proj(gate)


class MixtureOfDepthsRouter(nn.Module):
    """Token-level router for MoD."""

    def __init__(self, config: OuroborosConfig):
        super().__init__()
        self.router_weights = nn.Linear(config.d_model, 1, bias=False)
        self.capacity = config.mod_capacity

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch_size, seq_len, _ = x.shape
        k = max(1, int(seq_len * self.capacity))
        scores = self.router_weights(x).squeeze(-1)
        probs = torch.sigmoid(scores)
        topk_probs, topk_indices = torch.topk(probs, k, dim=1)
        return topk_probs, topk_indices, k


class Ouroboros6Model(nn.Module):
    """Ouroboros6: 6-Layer Cascaded LoopLM."""

    def __init__(self, config: OuroborosConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.iter_embedding = nn.Embedding(config.num_loops, config.d_model)

        self.attention_layers = nn.ModuleList([OuroborosAttention(config) for _ in range(config.num_layers)])
        self.mlp_layers = nn.ModuleList([OuroborosMLP(config) for _ in range(config.num_layers)])
        self.routers = nn.ModuleList([MixtureOfDepthsRouter(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model, config.eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(lambda m: self._init_weights(m))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        if iteration < self.config.num_loops:
            x = x + self.iter_embedding(torch.tensor(iteration, device=x.device)).unsqueeze(0).unsqueeze(1)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer_idx in range(self.config.num_layers):
            # Attention
            attn_out = self.attention_layers[layer_idx](x, freqs_cos, freqs_sin)
            x = x + attn_out

            # MoD routing for MLP
            topk_probs, topk_indices, k = self.routers[layer_idx](x)
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
            selected = x[batch_indices, topk_indices, :]
            mlp_out_selected = self.mlp_layers[layer_idx](selected)
            mlp_out_selected = mlp_out_selected * topk_probs.unsqueeze(-1)

            x_out = torch.zeros_like(x)
            x_out[batch_indices, topk_indices, :] = mlp_out_selected
            x = x + x_out

        return self.final_norm(x)


class Ouroboros6ForCausalLM(nn.Module):
    """Complete Ouroboros6 with LM head."""

    def __init__(self, config: OuroborosConfig):
        super().__init__()
        self.config = config
        self.model = Ouroboros6Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.model.embedding.weight

    def forward(self, input_ids: torch.Tensor, iteration: int = 0, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.model(input_ids, iteration)
        logits = self.lm_head(hidden)

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
# SCHEDULER
# =============================================================================

class WSDScheduler:
    """Warmup-Stable-Decay learning rate scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_seconds: float,
        warmup_seconds: float = 60.0,
        min_lr: float = 1e-6,
        decay_fraction: float = 0.2
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_seconds = warmup_seconds
        self.stable_end = total_seconds * (1.0 - decay_fraction)
        self.decay_end = total_seconds
        self.current_lr = 0.0

    def step(self, elapsed_seconds: float) -> float:
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
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OuroborosConfig()
    model = Ouroboros6ForCausalLM(config).to(device)
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
        for i in range(config.num_loops):
            _ = model(dummy, iteration=i)
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

            # Accumulate losses across loops
            optimizer.zero_grad()
            total_step_loss = 0.0

            for iteration in range(config.num_loops):
                logits, loss = model(X, iteration=iteration, labels=Y)
                if loss is not None:
                    loss.backward()
                    total_step_loss += loss.item()

            optimizer.step()

            total_loss += total_step_loss / config.num_loops
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
    print("Training complete!")


if __name__ == "__main__":
    train()
