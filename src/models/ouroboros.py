"""Ouroboros6: Cascaded 6-Layer LoopLM with Hexa Mixture-of-Depths.

This module implements the Ouroboros architecture: a recurrent transformer
where 6 layers are looped over multiple iterations with token-level routing
to selectively process only a fraction of tokens at each MLP layer.
"""

from typing import Tuple, Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    init_transformer_weights,
    count_parameters,
)
from src.config.ouroboros_config import OuroborosConfig


# =============================================================================
# Attention Mechanism
# =============================================================================

class OuroborosAttention(nn.Module):
    """Multi-head attention with RoPE and optional KV caching.

    Uses FlashAttention-compatible scaled dot-product attention with
    RoPE positional encoding applied only to new tokens.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        dropout_p: Dropout probability.
        norm: Pre-attention RMSNorm.
        Wqkv: Fused QKV projection.
        wo: Output projection.
    """

    def __init__(self, config: OuroborosConfig):
        """Initialize attention module.

        Args:
            config: Ouroboros configuration.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.norm = RMSNorm(config.d_model, config.eps)

        # Fused QKV projection for efficiency
        self.Wqkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV caching.

        Args:
            x: Input tensor [B, T, d_model].
            freqs_cos: Cosine frequencies [T, head_dim].
            freqs_sin: Sine frequencies [T, head_dim].
            past_key_value: Cached KV tensors from previous step.
            use_cache: Whether to cache KV for generation.

        Returns:
            Tuple of (output, past_key_value).
        """
        batch_size, seq_len, _ = x.shape

        # Pre-normalization
        h = self.norm(x)

        # Fused QKV projection
        qkv = self.Wqkv(h)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        xq, xk, xv = qkv.unbind(dim=2)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV cache handling
        if past_key_value is not None:
            past_k, past_v = past_key_value
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        # Transpose for attention [B, n_heads, T, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # FlashAttention-compatible scaled dot-product
        dropout_rate = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            dropout_p=dropout_rate,
            is_causal=True
        )

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out), past_kv


# =============================================================================
# Feed-Forward Network (SwiGLU)
# =============================================================================

class OuroborosMLP(nn.Module):
    """SwiGLU feed-forward network.

    Uses the SwiGLU activation: Swish(xW_gate) * (xW_up).
    More efficient than standard GELU + provides gating.

    Reference:
        Shazeer, N. (2020). GLU Variants Improve Transformer.

    Attributes:
        norm: Pre-MLP RMSNorm.
        gate_proj: Gating projection.
        up_proj: Up projection.
        down_proj: Down projection.
    """

    def __init__(self, config: OuroborosConfig):
        """Initialize MLP module.

        Args:
            config: Ouroboros configuration.
        """
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.eps)
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor [B, T, d_model].

        Returns:
            Output tensor [B, T, d_model].
        """
        h = self.norm(x)
        # SwiGLU: silu(xW_gate) * (xW_up)
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        return self.down_proj(gate * up)


# =============================================================================
# Mixture-of-Depths Router
# =============================================================================

class MixtureOfDepthsRouter(nn.Module):
    """Token-level router for Mixture-of-Depths.

    Implements differentiable routing to selectively process only a
    fraction of tokens through the MLP layer, reducing compute by
    (1 - capacity) on average.

    Reference:
        Raposo, D., et al. (2024). Mixture-of-Depths: Dynamically allocating
        compute in transformers.

    Attributes:
        router_weights: Learnable routing projection.
        capacity: Fraction of tokens to process.
    """

    def __init__(self, config: OuroborosConfig):
        """Initialize router.

        Args:
            config: Ouroboros configuration.
        """
        super().__init__()
        self.router_weights = nn.Linear(config.d_model, 1, bias=False)
        self.capacity = config.mod_capacity

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute routing decisions.

        Args:
            x: Input tensor [B, T, d_model].

        Returns:
            Tuple of (topk_probs, topk_indices, k).
        """
        batch_size, seq_len, _ = x.shape
        k = max(1, int(seq_len * self.capacity))

        # Router scores -> probabilities
        scores = self.router_weights(x).squeeze(-1)
        probs = torch.sigmoid(scores)

        # Select top-k tokens
        topk_probs, topk_indices = torch.topk(probs, k, dim=1)
        return topk_probs, topk_indices, k


# =============================================================================
# Complete Ouroboros6 Model
# =============================================================================

class Ouroboros6Model(nn.Module):
    """Ouroboros6: 6-Layer Cascaded LoopLM with Hexa MoD.

    Architecture: 6 transformer blocks with unique parameters each,
    processed recursively for num_loops iterations. Each block has
    independent MoD routing for its MLP layer.

    The "Hexa" designation refers to 6 independent MoD routers (one per layer).

    Attributes:
        config: Model configuration.
        embedding: Token embedding table.
        iter_embedding: Iteration/step embedding.
        attention_i: Attention modules for each layer.
        mlp_i: MLP modules for each layer.
        router_i: Router modules for each layer.
        final_norm: Final RMSNorm before LM head.
    """

    def __init__(self, config: OuroborosConfig):
        """Initialize Ouroboros6 model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.iter_embedding = nn.Embedding(config.num_loops, config.d_model)

        # Six independent transformer blocks
        self.attention_layers = nn.ModuleList([
            OuroborosAttention(config) for _ in range(config.num_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            OuroborosMLP(config) for _ in range(config.num_layers)
        ])
        self.routers = nn.ModuleList([
            MixtureOfDepthsRouter(config) for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model, config.eps)

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize weights
        self.apply(lambda m: init_transformer_weights(m, std=0.02))

        # Weight tying
        if config.tie_weights:
            # Will be set externally via tie_weights() method
            pass

    def tie_weights(self, lm_head: nn.Linear) -> None:
        """Tie embedding weights to LM head.

        Args:
            lm_head: Output projection layer.
        """
        lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        iteration: int = 0,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass through Ouroboros6.

        Args:
            input_ids: Input token IDs [B, T].
            iteration: Current loop iteration (0 to num_loops-1).
            past_key_values: Cached KV tensors from previous step.
            use_cache: Whether to cache KV for generation.

        Returns:
            Tuple of (hidden_states, past_key_values).
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding + iteration embedding
        x = self.embedding(input_ids)
        if iteration < self.config.num_loops:
            x = x + self.iter_embedding(
                torch.tensor(iteration, device=x.device)
            ).unsqueeze(0).unsqueeze(1)

        # Get RoPE frequencies for current sequence
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        # Process through 6 layers
        new_past_key_values = [] if use_cache else None

        for layer_idx in range(self.config.num_layers):
            # Attention with residual
            attn_out, layer_past_kv = self.attention_layers[layer_idx](
                x, freqs_cos, freqs_sin,
                past_key_value=past_key_values[layer_idx] if past_key_values else None,
                use_cache=use_cache
            )
            if use_cache:
                new_past_key_values.append(layer_past_kv)
            x = x + attn_out

            # MoD routing for MLP
            topk_probs, topk_indices, k = self.routers[layer_idx](x)

            # Gather selected tokens for MLP processing
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
            selected = x[batch_indices, topk_indices, :]

            # Process selected tokens through MLP
            mlp_out_selected = self.mlp_layers[layer_idx](selected)
            mlp_out_selected = mlp_out_selected * topk_probs.unsqueeze(-1)

            # Scatter back to original positions
            x_out = torch.zeros_like(x)
            x_out[batch_indices, topk_indices, :] = mlp_out_selected
            x = x + x_out

        # Final normalization
        x = self.final_norm(x)

        return x, new_past_key_values

    def get_input_embeddings(self) -> nn.Embedding:
        """Get token embedding layer."""
        return self.embedding

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return count_parameters(self)


# =============================================================================
# LM Head and Full Model Wrapper
# =============================================================================

class Ouroboros6ForCausalLM(nn.Module):
    """Complete Ouroboros6 model with LM head for causal language modeling.

    Wraps the base model with a language modeling head and provides
    utility methods for training and generation.
    """

    def __init__(self, config: OuroborosConfig):
        """Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model = Ouroboros6Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_weights:
            self.model.tie_weights(self.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        iteration: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Input token IDs [B, T].
            labels: Target token IDs [B, T] for loss computation.
            iteration: Current loop iteration.

        Returns:
            Tuple of (logits, loss).
        """
        hidden_states, _ = self.model(input_ids, iteration=iteration)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return count_parameters(self)

    def estimate_memory(self) -> dict:
        """Estimate memory requirements.

        Returns:
            Dictionary with memory estimates in MB.
        """
        param_bytes = self.get_parameter_count() * 2  # bfloat16
        param_mb = param_bytes / (1024 * 1024)

        # Activation memory (approximate for training)
        batch_size = 32
        seq_len = 2048
        activation_mb = (
            batch_size * seq_len * self.config.d_model * 4 *  # 4 bytes per float32
            self.config.num_loops * 2  # Forward + backward
        ) / (1024 * 1024)

        return {
            "parameters_mb": param_mb,
            "activations_mb": activation_mb,
            "total_mb": param_mb + activation_mb,
        }
