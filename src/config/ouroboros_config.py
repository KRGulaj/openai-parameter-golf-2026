"""Configuration dataclass for Ouroboros6 model.

Ouroboros implements a cascaded recurrent architecture (LoopLM) with
Mixture-of-Depths routing, optimized for the 16 MB parameter budget constraint.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OuroborosConfig:
    """Configuration for Ouroboros6 model (Cascaded 6-Layer LoopLM + Hexa MoD).

    Dimensions are calculated analytically to ensure 6 full blocks fit within
    ~15.9 MB in bfloat16 precision.

    Attributes:
        vocab_size: Size of token vocabulary (matches SP1024 tokenizer + specials).
        tie_weights: Whether to tie embedding and output projection weights.
        d_model: Model hidden dimension (representation width).
        n_heads: Number of attention heads (must divide d_model evenly).
        hidden_dim: MLP intermediate dimension (SwiGLU architecture).
        num_layers: Number of transformer blocks in the cascade.
        num_loops: Number of recurrent iterations through the block stack.
        mod_capacity: Mixture-of-Depths capacity fraction (fraction of tokens routed).
        dropout: Dropout probability (0.0 for competition to maximize capacity).
        eps: Epsilon for numerical stability in normalization.
        rope_theta: RoPE base frequency (inverse wavelength).
        max_seq_len: Maximum sequence length for training/evaluation.
    """

    # Competition compatibility
    vocab_size: int = 1056
    tie_weights: bool = True

    # Network dimensions - optimized for 6 layers within 16 MB
    d_model: int = 320
    n_heads: int = 5
    hidden_dim: int = 864
    num_layers: int = 6

    # Ouroboros-specific parameters
    num_loops: int = 8
    mod_capacity: float = 0.8

    # Regularization
    dropout: float = 0.0
    eps: float = 1e-6

    # Positional encoding
    rope_theta: float = 10000.0
    max_seq_len: int = 2048

    def __post_init__(self) -> None:
        """Validate configuration parameters and apply environment overrides."""
        # Environment variable overrides
        if os.environ.get("D_MODEL"):
            self.d_model = int(os.environ.get("D_MODEL", self.d_model))
        if os.environ.get("N_HEADS"):
            self.n_heads = int(os.environ.get("N_HEADS", self.n_heads))
        if os.environ.get("NUM_LOOPS"):
            self.num_loops = int(os.environ.get("NUM_LOOPS", self.num_loops))
        if os.environ.get("MOD_CAPACITY"):
            self.mod_capacity = float(os.environ.get("MOD_CAPACITY", self.mod_capacity))

        # Validation
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 0.0 < self.mod_capacity <= 1.0:
            raise ValueError(f"mod_capacity must be in (0, 1], got {self.mod_capacity}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head.

        Returns:
            Integer head dimension (d_model // n_heads).
        """
        return self.d_model // self.n_heads

    def estimate_parameters(self) -> int:
        """Estimate total parameter count.

        Returns:
            Estimated number of parameters (excluding RoPE buffers).
        """
        # Embedding
        embed_params = self.vocab_size * self.d_model

        # Per-layer parameters (Attention + MLP)
        attn_params = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        mlp_params = 3 * self.d_model * self.hidden_dim  # gate, up, down projections
        mod_params = self.d_model  # Router weight
        norm_params = 2 * self.d_model  # Pre-attn + pre-MLP norms
        layer_params = attn_params + mlp_params + mod_params + norm_params

        # Total
        total = embed_params + (self.num_layers * layer_params) + self.d_model  # + final norm
        if self.tie_weights:
            total -= self.d_model * self.vocab_size  # Subtract tied output

        return total
