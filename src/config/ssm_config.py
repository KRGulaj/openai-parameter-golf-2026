"""Configuration dataclass for SSM8 model.

SSM8 implements a "Fat State" Mamba architecture with extended hidden state
dimension (d_state=32) for improved long-context retention within parameter budget.
"""

import os
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SSMConfig:
    """Configuration for SSM8 "Fat State" Mamba model.

    The "Fat State" design prioritizes state retention capacity over layer depth,
    using d_state=32 compared to standard Mamba's d_state=8 or d_state=16.

    Attributes:
        vocab_size: Size of token vocabulary (matches SP1024 tokenizer).
        tie_weights: Whether to tie embedding and LM head weights.
        d_model: Main hidden dimension (multiple of 64 for Tensor Cores).
        d_inner: Expanded dimension (typically 2.0 * d_model).
        d_state: SSM hidden state dimension (32 for "Fat State" design).
        d_conv: Causal convolution kernel size (default 4).
        num_layers: Number of SSM blocks in the stack.
        head_adapter_rank: LoRA rank for output head adaptation (set to 0 to disable).
        dt_min: Minimum time step for discretization.
        dt_max: Maximum time step for discretization.
        bias: Whether to use bias in linear layers (False for efficiency).
        conv_bias: Whether to use bias in causal convolution (True).
        eps: Epsilon for numerical stability.
        qat_threshold: Learning rate scale threshold for fake quantization.
    """

    # Competition compatibility
    vocab_size: int = 1056
    tie_weights: bool = True

    # Network dimensions
    d_model: int = 640
    d_inner: int = 1280
    d_state: int = 34
    d_conv: int = 4
    num_layers: int = 8
    head_adapter_rank: int = 16

    # Dynamics initialization
    dt_min: float = 0.001
    dt_max: float = 0.1

    # Regularization
    bias: bool = False
    conv_bias: bool = True
    eps: float = 1e-6

    # Quantization
    qat_threshold: float = 0.15

    def __post_init__(self) -> None:
        """Validate configuration and apply environment overrides."""
        # Environment overrides
        if os.environ.get("D_STATE"):
            self.d_state = max(8, int(os.environ.get("D_STATE", self.d_state)))
        if os.environ.get("D_MODEL"):
            self.d_model = int(os.environ.get("D_MODEL", self.d_model))
        if os.environ.get("HEAD_ADAPTER_RANK"):
            self.head_adapter_rank = max(
                0, int(os.environ.get("HEAD_ADAPTER_RANK", self.head_adapter_rank))
            )

        # Validation
        if self.d_inner < self.d_model:
            raise ValueError(f"d_inner ({self.d_inner}) must be >= d_model ({self.d_model})")
        if self.d_state < 1:
            raise ValueError(f"d_state must be >= 1, got {self.d_state}")
        if self.d_conv < 1:
            raise ValueError(f"d_conv must be >= 1, got {self.d_conv}")

    @property
    def dt_rank(self) -> int:
        """Rank for time step parameter (typically ceil(d_model / 16)).

        Returns:
            Integer rank for dt projection.
        """
        return math.ceil(self.d_model / 16)

    def estimate_parameters(self) -> int:
        """Estimate total parameter count.

        Returns:
            Estimated number of parameters.
        """
        # Embedding
        embed_params = self.vocab_size * self.d_model

        # Per-block parameters
        # in_proj: d_model -> 2*d_inner (for x, z)
        in_proj_params = self.d_model * 2 * self.d_inner
        # x_proj: d_inner -> dt_rank + 2*d_state (for dt, B, C)
        x_proj_params = self.d_inner * (self.dt_rank + 2 * self.d_state)
        # dt_proj: dt_rank -> d_inner
        dt_proj_params = self.dt_rank * self.d_inner
        # out_proj: d_inner -> d_model
        out_proj_params = self.d_inner * self.d_model
        # conv1d: d_inner * d_conv
        conv_params = self.d_inner * self.d_conv
        # A_log and D: d_inner
        ssm_params = 2 * self.d_inner
        # dt_bias: d_inner
        dt_bias_params = self.d_inner

        block_params = (
            in_proj_params + x_proj_params + dt_proj_params +
            out_proj_params + conv_params + ssm_params + dt_bias_params
        )

        # Head adapter (LoRA)
        adapter_params = 0
        if self.head_adapter_rank > 0:
            adapter_params = 2 * self.d_model * self.head_adapter_rank

        total = embed_params + (self.num_layers * block_params) + adapter_params
        if self.tie_weights:
            total += self.vocab_size * self.d_model  # LM head (tied)

        return total
