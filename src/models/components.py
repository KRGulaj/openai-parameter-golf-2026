"""Shared neural network components across all model architectures.

This module provides reusable building blocks including normalization layers,
positional encodings, and the Muon optimizer with Newton-Schulz orthogonalization.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Normalization Layers
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is computationally cheaper than LayerNorm (no mean centering) and
    has been shown to be equally effective for transformer architectures.

    Reference:
        Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
        NeurIPS 2019.

    Attributes:
        eps: Small constant for numerical stability.
        weight: Learnable gain parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension to normalize over.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Normalized tensor with same shape as input.
        """
        # Compute RMS over the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    base_seq_len: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE frequency tables with optional NTK-aware scaling.

    Implements Rotary Position Embedding with support for long sequences
    via NTK-aware interpolation.

    Reference:
        Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary
        Position Embedding. arXiv:2104.09864.

    Args:
        dim: Dimension of the head (must be even).
        end: Maximum sequence length to precompute.
        theta: Base frequency (inverse wavelength).
        base_seq_len: Base sequence length for NTK scaling.

    Returns:
        Tuple of (cos_table, sin_table), each of shape [end, dim//2].
    """
    # NTK-aware scaling for long sequences
    if end > base_seq_len:
        scale = end / base_seq_len
        theta = theta * (scale ** (dim / (dim - 2)))

    # Compute frequencies: theta^{-2i/d} for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()

    # Duplicate frequencies for real-valued rotation
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions for RoPE application.

    Args:
        x: Input tensor of shape [..., dim].

    Returns:
        Rotated tensor with same shape.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys.

    Args:
        xq: Query tensor of shape [B, T, n_heads, head_dim].
        xk: Key tensor of shape [B, T, n_heads, head_dim].
        freqs_cos: Cosine frequencies of shape [T, head_dim].
        freqs_sin: Sine frequencies of shape [T, head_dim].

    Returns:
        Tuple of (rotated_queries, rotated_keys).
    """
    # Add batch and head dimensions to frequency tables
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    xq_out = (xq * freqs_cos) + (rotate_half(xq) * freqs_sin)
    xk_out = (xk * freqs_cos) + (rotate_half(xk) * freqs_sin)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_partial_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dims: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only the first rope_dims of each head.

    Used in LiteFormer to allow position-independent representation in
    the remaining dimensions.

    Args:
        xq: Query tensor [B, T, n_heads, head_dim].
        xk: Key tensor [B, T, n_heads, head_dim].
        cos: Cosine frequencies [T, rope_dims].
        sin: Sine frequencies [T, rope_dims].
        rope_dims: Number of dimensions to apply RoPE.

    Returns:
        Tuple of (modified_queries, modified_keys).
    """
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Split into RoPE and non-RoPE portions
    xq_r, xq_p = xq[..., :rope_dims], xq[..., rope_dims:]
    xk_r, xk_p = xk[..., :rope_dims], xk[..., rope_dims:]

    # Apply RoPE only to the first portion
    xq_r = xq_r * cos + rotate_half(xq_r) * sin
    xk_r = xk_r * cos + rotate_half(xk_r) * sin

    return (
        torch.cat([xq_r, xq_p], dim=-1).type_as(xq),
        torch.cat([xk_r, xk_p], dim=-1).type_as(xk)
    )


# =============================================================================
# Muon Optimizer
# =============================================================================

@torch.compile
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5
) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalizing gradient matrices.

    Implements 5-step Newton-Schulz iteration to compute an orthogonal
    approximation of the input matrix. Used in Muon optimizer for
    momentum orthogonalization.

    Reference:
        Bernstein, J. (2024). Learning via Curriculum Relaxation.
        arXiv:2409.2037 (Newton-Schulz coefficients).

    Args:
        G: Gradient matrix of shape [m, n].
        steps: Number of Newton-Schulz iterations (default 5).

    Returns:
        Orthogonalized matrix of same shape as G.
    """
    assert G.ndim == 2, "Input must be 2D matrix"

    # Coefficients tuned for 5-step convergence
    a, b, c = 3.4445, -4.7750, 2.0315

    # Transpose if m > n for efficiency
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T

    # Normalize to spectral radius
    X = G.bfloat16() / (G.norm() + 1e-7)

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon optimizer: momentum with Newton-Schulz orthogonalization.

    Muon applies momentum-based gradient accumulation followed by
    Newton-Schulz orthogonalization to maintain well-conditioned
    update directions. Particularly effective for training deep
    transformers and recurrent architectures.

    Reference:
        Bernstein, J. (2024). Learning via Curriculum Relaxation.

    Attributes:
        lr: Learning rate.
        momentum: Momentum coefficient.
        weight_decay: Weight decay coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.04
    ):
        """Initialize Muon optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate (default 0.02).
            momentum: Momentum coefficient (default 0.95).
            weight_decay: Weight decay coefficient (default 0.04).
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns loss.

        Returns:
            Loss value if closure provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Skip non-2D tensors (biases, embeddings handled separately)
                if g.ndim != 2:
                    continue

                # Weight decay
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # Momentum buffer
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]

                # Update momentum
                buf.mul_(mu).add_(g)

                # Orthogonalize
                g_ortho = zeropower_via_newtonschulz5(buf)

                # Scale correction by aspect ratio
                scale = max(1.0, g_ortho.shape[0] / g_ortho.shape[1]) ** 0.5

                # Parameter update
                p.data.add_(g_ortho, alpha=-lr * scale)

        return loss


# =============================================================================
# Weight Initialization
# =============================================================================

def init_transformer_weights(module: nn.Module, std: float = 0.02) -> None:
    """Initialize transformer module weights with small-init trick.

    Small initialization helps stabilize deep recurrent architectures by
    preventing variance explosion at startup.

    Args:
        module: PyTorch module to initialize.
        std: Standard deviation for normal initialization.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count total parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
