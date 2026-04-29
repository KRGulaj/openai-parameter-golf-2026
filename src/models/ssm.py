"""SSM8: "Fat State" Mamba architecture for Parameter Golf.

This module implements an 8-layer Mamba State Space Model with extended
hidden state (d_state=32) for improved long-context retention. Uses the
mamba_ssm and causal_conv1d libraries for hardware-accelerated selective scan.
"""

import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import RMSNorm, count_parameters
from src.config.ssm_config import SSMConfig


# =============================================================================
# Hardware-Accelerated Selective Scan
# =============================================================================

try:
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    # Fallback implementations will raise errors if used


@torch.compiler.disable
def run_selective_scan(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    dt_bias: torch.Tensor,
    return_last_state: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Execute hardware selective scan kernel.

    This function is excluded from torch.compile graph as the underlying
    CUDA kernel cannot be traced by Inductor. All SSM dynamics parameters
    (A, D, dt) are kept in float32 for numerical stability.

    Args:
        u: Input activations [B, d_inner, L].
        dt: Time step parameters [B, d_inner, L].
        A: State matrix [d_inner, d_state].
        B: Input-dependent transition [B, d_state, L].
        C: Output projection [B, d_state, L].
        D: Skip connection [d_inner].
        z: Gating activations [B, d_inner, L].
        dt_bias: Time step bias [d_inner].
        return_last_state: Whether to return final state for caching.

    Returns:
        Tuple of (output, last_state).
    """
    if not MAMBA_AVAILABLE:
        raise RuntimeError("mamba_ssm and causal_conv1d must be installed")

    # Cast to float32 for numerical stability (anti-autocast poisoning)
    return selective_scan_fn(
        u.float(),
        dt.float(),
        A,
        B.float(),
        C.float(),
        D=D,
        z=z.float(),
        delta_bias=dt_bias,
        delta_softplus=True,
        return_last_state=return_last_state,
    )


# =============================================================================
# SSM8 Block (Fat State Mamba)
# =============================================================================

class SSM8Block(nn.Module):
    """Single Mamba SSM block with Fat State (d_state=32).

    The "Fat State" design uses d_state=32 compared to standard d_state=8/16,
    providing richer long-range memory without additional parameters.

    Supports recurrent state caching for efficient sliding-window evaluation.

    Attributes:
        config: SSM configuration.
        norm: Pre-block RMSNorm.
        in_proj: Input projection to (x, z, B, C, delta).
        conv1d: Causal convolution for local context.
        dt_rank: Rank for delta projection.
        x_proj: Projection from x to (delta, B, C).
        dt_proj: Projection from delta to d_inner.
        A_log: Log of state matrix A (HiPPO initialization).
        D: Skip connection parameter.
        out_proj: Output projection.
    """

    def __init__(self, config: SSMConfig):
        """Initialize SSM8 block.

        Args:
            config: SSM configuration.
        """
        super().__init__()
        self.config = config

        d_model = config.d_model
        d_inner = config.d_inner
        d_state = config.d_state
        d_conv = config.d_conv

        # Input normalization
        self.norm = RMSNorm(d_model, config.eps)

        # Input projection: d_model -> 2*d_inner (for x and z)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=config.conv_bias,
            kernel_size=d_conv,
            groups=d_inner,  # Depthwise separable
            padding=d_conv - 1,  # Causal padding
        )

        # Time step rank
        self.dt_rank = math.ceil(d_model / 16)

        # Projection from x to (delta, B, C)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Projection from delta to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # State matrix A (HiPPO initialization)
        A_init = self._init_A(d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A_init))

        # Skip connection D (learnable)
        self.D = nn.Parameter(torch.ones(d_inner))

        # dt bias (learnable)
        self.dt_bias = nn.Parameter(torch.randn(d_inner) * 0.01)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _init_A(self, d_inner: int, d_state: int) -> torch.Tensor:
        """Initialize state matrix A with HiPPO-LegS.

        Args:
            d_inner: Inner dimension.
            d_state: State dimension.

        Returns:
            Initialized A matrix [d_inner, d_state].
        """
        # HiPPO-LegS: A[i, j] = (2*i + 1)**0.5 * (2*j + 1)**0.5 / 2
        # Simplified: A[i, n] = n + 1 for n in [0, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        return A

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_last_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through SSM block.

        Args:
            x: Input tensor [B, L, d_model].
            initial_state: Optional initial state [B, d_inner, d_state].
            return_last_state: Whether to return final state.

        Returns:
            Tuple of (output, last_state).
        """
        batch_size, seq_len, _ = x.shape
        d_inner = self.config.d_inner
        d_state = self.config.d_state

        # Input normalization and projection
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)  # [B, L, d_inner] each

        # Causal convolution on x
        x_inner = x_inner.transpose(1, 2)  # [B, d_inner, L]
        x_inner = self.conv1d(x_inner)[..., :seq_len]
        x_inner = x_inner.transpose(1, 2)  # [B, L, d_inner]

        # Compute delta, B, C from x
        xbc = self.x_proj(x_inner)  # [B, L, dt_rank + 2*d_state]
        delta, B, C = torch.split(
            xbc,
            [self.dt_rank, d_state, d_state],
            dim=-1
        )

        # Project delta to d_inner
        delta = self.dt_proj(delta).transpose(1, 2)  # [B, d_inner, L]

        # Get A from log parameter (kept in FP32)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Transpose for selective_scan
        x_inner = x_inner.transpose(1, 2)  # [B, d_inner, L]
        B = B.transpose(1, 2)  # [B, d_state, L]
        C = C.transpose(1, 2)  # [B, d_state, L]
        z = z.transpose(1, 2)  # [B, d_inner, L]

        # Execute selective scan (FP32 for stability)
        y, last_state = run_selective_scan(
            x_inner,
            delta,
            A,
            B,
            C,
            self.D.float(),
            z,
            self.dt_bias.float(),
            return_last_state=return_last_state
        )

        # y: [B, d_inner, L]
        y = y.transpose(1, 2)  # [B, L, d_inner]

        # Output projection
        output = self.out_proj(y)

        return output, last_state


# =============================================================================
# Head Adapter (LoRA for output projection)
# =============================================================================

class HeadAdapter(nn.Module):
    """Low-rank adapter for LM head.

    Optional LoRA adaptation to provide additional capacity without
    full weight updates. Used during test-time training.

    Attributes:
        rank: LoRA rank.
        lora_A: Down projection.
        lora_B: Up projection.
    """

    def __init__(self, d_model: int, vocab_size: int, rank: int):
        """Initialize head adapter.

        Args:
            d_model: Model dimension.
            vocab_size: Vocabulary size.
            rank: LoRA rank.
        """
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Linear(d_model, rank, bias=False)
        self.lora_B = nn.Linear(rank, vocab_size, bias=False)

        # Initialize B to zero so adapter starts at identity
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        """Apply adapter to base logits.

        Args:
            x: Hidden states [B, L, d_model].
            base_logits: Base model logits [B, L, vocab_size].

        Returns:
            Modified logits [B, L, vocab_size].
        """
        if self.rank == 0:
            return base_logits
        adapter_out = self.lora_B(self.lora_A(x))
        return base_logits + adapter_out


# =============================================================================
# Complete SSM8 Model
# =============================================================================

class SSM8Model(nn.Module):
    """SSM8: 8-Layer "Fat State" Mamba.

    Architecture: 8 SSM blocks with d_state=32, optimized for long
    context retention within 16 MB parameter budget.

    Attributes:
        config: Model configuration.
        embedding: Token embedding table.
        blocks: SSM blocks.
        norm: Final RMSNorm.
        head_adapter: Optional LoRA head adapter.
    """

    def __init__(self, config: SSMConfig):
        """Initialize SSM8 model.

        Args:
            config: SSM configuration.
        """
        super().__init__()
        self.config = config

        if not MAMBA_AVAILABLE:
            raise RuntimeError(
                "mamba_ssm and causal_conv1d are required. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )

        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # SSM blocks
        self.blocks = nn.ModuleList([
            SSM8Block(config) for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model, config.eps)

        # Optional head adapter (LoRA)
        if config.head_adapter_rank > 0:
            self.head_adapter = HeadAdapter(
                config.d_model, config.vocab_size, config.head_adapter_rank
            )
        else:
            self.head_adapter = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Initialize dt_proj bias for stable discretization
        for block in self.blocks:
            dt_init_std = self.config.dt_rank ** -0.5
            nn.init.uniform_(block.dt_bias, self.config.dt_min, self.config.dt_max)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[list] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass through SSM8.

        Args:
            input_ids: Input token IDs [B, L].
            states: Optional list of initial states for each block.
            return_states: Whether to return final states.

        Returns:
            Tuple of (hidden_states, final_states).
        """
        x = self.embedding(input_ids)

        new_states = [] if return_states else None

        for i, block in enumerate(self.blocks):
            initial_state = states[i] if states else None
            residual = x
            x, last_state = block(x, initial_state, return_last_state=return_states)
            x = x + residual  # Residual connection

            if return_states:
                new_states.append(last_state)

        x = self.norm(x)
        return x, new_states

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return count_parameters(self)


# =============================================================================
# LM Head and Full Model Wrapper
# =============================================================================

class SSM8ForCausalLM(nn.Module):
    """Complete SSM8 model with LM head for causal language modeling.
    """

    def __init__(self, config: SSMConfig):
        """Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model = SSM8Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_weights:
            self.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """Forward pass with optional loss.

        Args:
            input_ids: Input token IDs [B, L].
            labels: Target token IDs [B, L].
            states: Optional initial states.

        Returns:
            Tuple of (logits, loss, final_states).
        """
        hidden_states, new_states = self.model(input_ids, states)
        logits = self.lm_head(hidden_states)

        # Apply head adapter if present
        if self.model.head_adapter is not None:
            logits = self.model.head_adapter(hidden_states, logits)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss, new_states

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return self.model.get_parameter_count()


# =============================================================================
# Utility Functions
# =============================================================================

def create_ssm8_model(
    vocab_size: int = 1056,
    d_model: int = 640,
    d_state: int = 34,
    **kwargs
) -> SSM8ForCausalLM:
    """Factory function to create SSM8 model.

    Args:
        vocab_size: Token vocabulary size.
        d_model: Model hidden dimension.
        d_state: SSM state dimension.
        **kwargs: Additional config arguments.

    Returns:
        Initialized SSM8ForCausalLM model.
    """
    config = SSMConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        **kwargs
    )
    return SSM8ForCausalLM(config)
