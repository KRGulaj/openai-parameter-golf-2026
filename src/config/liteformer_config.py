"""Configuration dataclass for LiteFormer-11L model.

LiteFormer implements a standard Transformer with aggressive compression
(int6 quantization + zstd-22) to fit 35M parameters within 16 MB budget.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LiteFormerConfig:
    """Configuration for LiteFormer-11L model.

    Optimized for compression efficiency with int6 quantization and zstd-22.
    Uses GQA, relu² activation, and specialized architectural features for
    improved quantizability.

    Attributes:
        vocab_size: Token vocabulary size (SP1024 tokenizer).
        d_model: Model hidden dimension (512 for optimal Tensor Core usage).
        n_heads: Number of query attention heads (8).
        n_kv_heads: Number of key/value heads for GQA (4, half of n_heads).
        head_dim: Dimension per head (64 = d_model // n_heads).
        hidden_dim: MLP intermediate dimension (3x expansion for capacity).
        num_layers: Number of transformer layers (11).
        rope_theta: RoPE base frequency.
        partial_rope_dims: Number of head dimensions to apply RoPE (16 of 64).
        xsa_last_n: Apply Exclusive Self Attention on last N layers (4).
        bigram_buckets: Number of hash buckets for bigram embedding (2048).
        bigram_dim: Dimension of bigram embedding before projection (128).
        logit_softcap: Maximum logit magnitude via tanh squashing (30.0).
        eps: Epsilon for numerical stability.
        qat_threshold: Learning rate scale for quantization-aware training.
        ema_decay: Exponential moving average decay rate (0.997).
    """

    # Core dimensions
    vocab_size: int = 1056
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 64
    hidden_dim: int = 1536
    num_layers: int = 11

    # Positional encoding
    rope_theta: float = 10000.0
    partial_rope_dims: int = 16

    # Special modules
    xsa_last_n: int = 4
    bigram_buckets: int = 2048
    bigram_dim: int = 128
    logit_softcap: float = 30.0

    # Training
    eps: float = 1e-6
    qat_threshold: float = 0.15
    ema_decay: float = 0.997

    # Compression
    quant_bits: int = 6
    zstd_level: int = 22

    def __post_init__(self) -> None:
        """Validate configuration and apply environment overrides."""
        # Environment overrides
        if os.environ.get("D_MODEL"):
            self.d_model = int(os.environ.get("D_MODEL", self.d_model))
        if os.environ.get("N_HEADS"):
            self.n_heads = int(os.environ.get("N_HEADS", self.n_heads))
        if os.environ.get("NUM_LAYERS"):
            self.num_layers = int(os.environ.get("NUM_LAYERS", self.num_layers))

        # Derived values
        self.head_dim = self.d_model // self.n_heads

        # Validation
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.partial_rope_dims > self.head_dim:
            raise ValueError(
                f"partial_rope_dims ({self.partial_rope_dims}) must be <= head_dim ({self.head_dim})"
            )
        if self.bigram_buckets < 1:
            raise ValueError(f"bigram_buckets must be >= 1, got {self.bigram_buckets}")

    @property
    def kv_head_dim(self) -> int:
        """Dimension per KV head (may differ from query head in GQA).

        Returns:
            Integer dimension per KV head.
        """
        return self.d_model // self.n_kv_heads

    @property
    def gqa_group_size(self) -> int:
        """Number of query heads sharing each KV head in GQA.

        Returns:
            Integer group size (n_heads // n_kv_heads).
        """
        return self.n_heads // self.n_kv_heads

    def estimate_parameters(self) -> int:
        """Estimate total parameter count.

        Returns:
            Estimated number of parameters.
        """
        # Embedding
        embed_params = self.vocab_size * self.d_model

        # Bigram hash embedding
        bigram_params = self.bigram_buckets * self.bigram_dim + self.bigram_dim * self.d_model

        # Per-layer parameters
        # Attention: Q (d_model * d_model) + K/V (d_model * d_model * n_kv_heads / n_heads) + O (d_model * d_model)
        q_params = self.d_model * self.d_model
        kv_params = 2 * self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        o_params = self.d_model * self.d_model
        attn_params = q_params + kv_params + o_params

        # MLP: 3x expansion with gate, up, down
        mlp_params = 3 * self.d_model * self.hidden_dim

        # Norms: 2 per layer (pre-attn + pre-mlp)
        norm_params = 2 * self.d_model

        # SmearGate: d_model
        smeargate_params = self.d_model

        layer_params = attn_params + mlp_params + norm_params + smeargate_params

        # Total
        total = embed_params + bigram_params + (self.num_layers * layer_params) + self.d_model  # final norm

        return total

    def estimate_compressed_size(self) -> Tuple[float, float]:
        """Estimate compressed model size with int6 + zstd-22.

        Returns:
            Tuple of (uncompressed_mb, compressed_mb).
        """
        total_params = self.estimate_parameters()

        # int6: ~0.75 bytes per parameter (with row scales)
        uncompressed_mb = total_params * 0.75 / (1024 * 1024)

        # zstd-22: ~15-20% additional compression
        compressed_mb = uncompressed_mb * 0.82  # Conservative estimate

        return uncompressed_mb, compressed_mb
