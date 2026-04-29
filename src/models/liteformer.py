"""LiteFormer-11L: Compressed Transformer with int6+zstd-22.

This module implements a standard 11-layer transformer optimized for
aggressive compression. Key features:
- Grouped Query Attention (GQA) for KV cache reduction
- relu² activation for sparsity and compressibility
- Exclusive Self Attention (XSA) for focused attention
- SmearGate for causal blending
- BigramHashEmbedding for local pair statistics
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    RMSNorm,
    precompute_freqs_cis,
    apply_partial_rope,
    init_transformer_weights,
    count_parameters,
)
from src.config.liteformer_config import LiteFormerConfig


# =============================================================================
# SmearGate: Causal Mean Blending
# =============================================================================

class SmearGate(nn.Module):
    """Causal mean gate for local context blending.

    Blends token representations with their cumulative causal mean
    using a learnable gate parameter. Helps model leverage local
    smoothness in language without additional attention cost.

    Attributes:
        gate: Learnable gating parameter [d_model].
    """

    def __init__(self, d_model: int):
        """Initialize SmearGate.

        Args:
            d_model: Model dimension.
        """
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal mean blending.

        Args:
            x: Input tensor [B, T, d_model].

        Returns:
            Blended tensor [B, T, d_model].
        """
        batch_size, seq_len, _ = x.shape

        # Compute cumulative mean (causal)
        denom = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype)
        denom = denom.view(1, seq_len, 1)
        causal_mean = x.cumsum(dim=1) / denom

        # Gated blend
        g = torch.sigmoid(self.gate)
        return x + g * (causal_mean - x)


# =============================================================================
# BigramHashEmbedding
# =============================================================================

class BigramHashEmbedding(nn.Module):
    """Hash-based bigram embedding for local pair statistics.

    Embeds bigrams (prev_token, curr_token) using hash-based bucketing.
    Captures local token pair patterns beyond unigram statistics.
    Efficient: ~327K parameters covering top 2048 most frequent bigrams.

    Attributes:
        num_buckets: Number of hash buckets.
        bigram_dim: Embedding dimension.
        bigram_embed: Bigram embedding table [num_buckets, bigram_dim].
        proj: Projection to d_model [bigram_dim, d_model].
    """

    def __init__(self, num_buckets: int, bigram_dim: int, d_model: int):
        """Initialize bigram embedding.

        Args:
            num_buckets: Number of hash buckets (2048).
            bigram_dim: Bigram embedding dimension (128).
            d_model: Model dimension for projection.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.bigram_dim = bigram_dim

        self.bigram_embed = nn.Embedding(num_buckets, bigram_dim)
        self.proj = nn.Linear(bigram_dim, d_model, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute bigram embeddings.

        Args:
            input_ids: Token IDs [B, T].

        Returns:
            Bigram features [B, T, d_model].
        """
        batch_size, seq_len = input_ids.shape

        # Get previous tokens (padded with 0)
        prev_ids = torch.cat([
            torch.zeros(batch_size, 1, dtype=input_ids.dtype, device=input_ids.device),
            input_ids[:, :-1]
        ], dim=1)

        # Hash bigram to bucket index
        # Simple hash: (prev * vocab_size + curr) % num_buckets
        hash_val = (prev_ids * 1056 + input_ids) % self.num_buckets

        # Embed and project
        bigram_emb = self.bigram_embed(hash_val)
        return self.proj(bigram_emb)


# =============================================================================
# Grouped Query Attention (GQA)
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention.

    GQA reduces KV cache size by sharing KV heads across multiple
    query heads. Here: 8 query heads share 4 KV heads (group_size=2).

    Reference:
        Ainslie, J., et al. (2023). GQA: Training Generalized
        Multi-Query Transformer Models.

    Attributes:
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per head.
        group_size: Query heads per KV head.
        norm: Pre-attention RMSNorm.
        wq: Query projection.
        wk: Key projection.
        wv: Value projection.
        wo: Output projection.
    """

    def __init__(self, config: LiteFormerConfig):
        """Initialize GQA module.

        Args:
            config: LiteFormer configuration.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.gqa_group_size

        assert config.n_heads % config.n_kv_heads == 0

        self.norm = RMSNorm(config.d_model, config.eps)

        # Projections
        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_dims: int,
        use_xsa: bool = False
    ) -> torch.Tensor:
        """Forward pass with GQA.

        Args:
            x: Input tensor [B, T, d_model].
            cos: Cosine RoPE frequencies [T, rope_dims].
            sin: Sine RoPE frequencies [T, rope_dims].
            rope_dims: Dimensions to apply RoPE.
            use_xsa: Whether to apply Exclusive Self Attention.

        Returns:
            Output tensor [B, T, d_model].
        """
        batch_size, seq_len, _ = x.shape

        # Normalize
        h = self.norm(x)

        # Project to Q, K, V
        q = self.wq(h).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(h).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(h).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply partial RoPE
        q, k = apply_partial_rope(q, k, cos, sin, rope_dims)

        # Expand KV heads to match query heads
        k = k.repeat_interleave(self.group_size, dim=2)  # [B, T, n_heads, head_dim]
        v = v.repeat_interleave(self.group_size, dim=2)

        # Transpose for attention
        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlashAttention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=True
        )

        # Exclusive Self Attention (optional)
        if use_xsa:
            attn_out = self._apply_xsa(attn_out, v)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(attn_out)

    def _apply_xsa(
        self,
        attn_out: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Apply Exclusive Self Attention.

        Subtracts projection of output onto own value vector,
        forcing attention to carry only orthogonal information.

        Args:
            attn_out: Attention output [B, n_heads, T, head_dim].
            v: Value tensor [B, n_heads, T, head_dim].

        Returns:
            Modified attention output.
        """
        # Compute projection onto own value
        # attn_out: [B, n_heads, T, head_dim]
        # v: [B, n_heads, T, head_dim]
        # We want: attn_out - proj(attn_out onto v)

        # Dot product for each position
        dot = (attn_out * v).sum(dim=-1, keepdim=True)  # [B, n_heads, T, 1]

        # Norm squared of v
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-6

        # Projection
        proj = dot * v / v_norm_sq

        return attn_out - proj


# =============================================================================
# Feed-Forward Network (relu²)
# =============================================================================

class ReluSquaredMLP(nn.Module):
    """MLP with relu² activation.

    Uses ReLU squared instead of SiLU/SwiGLU. relu² provides:
    - Sparse activations (many zeros) improving compressibility
    - Better BPB at this scale per leaderboard analysis
    - 3x expansion ratio for capacity

    Attributes:
        norm: Pre-MLP RMSNorm.
        gate_proj: Gating projection.
        up_proj: Up projection.
        down_proj: Down projection.
    """

    def __init__(self, config: LiteFormerConfig):
        """Initialize MLP.

        Args:
            config: LiteFormer configuration.
        """
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.eps)
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with relu² activation.

        Args:
            x: Input tensor [B, T, d_model].

        Returns:
            Output tensor [B, T, d_model].
        """
        h = self.norm(x)

        # relu²: ReLU(x)²
        gate = F.relu(self.gate_proj(h)) ** 2
        up = self.up_proj(h)

        return self.down_proj(gate * up)


# =============================================================================
# LiteFormer-11L Model
# =============================================================================

class LiteFormerModel(nn.Module):
    """LiteFormer-11L: Standard Transformer with compression optimizations.

    Architecture: 11 layers with GQA, partial RoPE, XSA on last 4 layers,
    relu² MLP, SmearGate, and BigramHashEmbedding.

    Attributes:
        config: Model configuration.
        embedding: Token embedding.
        bigram_embed: Bigram hash embedding.
        layers: Transformer layers.
        final_norm: Final RMSNorm.
    """

    def __init__(self, config: LiteFormerConfig):
        """Initialize LiteFormer model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.bigram_embed = BigramHashEmbedding(
            config.bigram_buckets,
            config.bigram_dim,
            config.d_model
        )

        # Transformer layers
        self.attn_layers = nn.ModuleList([
            GroupedQueryAttention(config) for _ in range(config.num_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            ReluSquaredMLP(config) for _ in range(config.num_layers)
        ])
        self.smeargates = nn.ModuleList([
            SmearGate(config.d_model) for _ in range(config.num_layers)
        ])
        self.norms = nn.ModuleList([
            RMSNorm(config.d_model, config.eps) for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.d_model, config.eps)

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.partial_rope_dims,
            4096,  # Max sequence length
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize
        self.apply(lambda m: init_transformer_weights(m, std=0.02))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs [B, T].

        Returns:
            Hidden states [B, T, d_model].
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids) + self.bigram_embed(input_ids)

        # Get RoPE frequencies
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        # Layers
        for layer_idx in range(self.config.num_layers):
            # Pre-norm
            x_norm = self.norms[layer_idx](x)

            # Attention
            use_xsa = layer_idx >= (self.config.num_layers - self.config.xsa_last_n)
            attn_out = self.attn_layers[layer_idx](
                x_norm, freqs_cos, freqs_sin,
                self.config.partial_rope_dims,
                use_xsa=use_xsa
            )
            x = x + attn_out

            # SmearGate
            x = self.smeargates[layer_idx](x)

            # MLP
            mlp_out = self.mlp_layers[layer_idx](x)
            x = x + mlp_out

        # Final norm
        x = self.final_norm(x)
        return x

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return count_parameters(self)


# =============================================================================
# LM Head and Full Model Wrapper
# =============================================================================

class LiteFormerForCausalLM(nn.Module):
    """Complete LiteFormer model with LM head."""

    def __init__(self, config: LiteFormerConfig):
        """Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model = LiteFormerModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional loss.

        Args:
            input_ids: Input token IDs [B, T].
            labels: Target token IDs [B, T].

        Returns:
            Tuple of (logits, loss).
        """
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        # Logit soft-capping
        if self.config.logit_softcap > 0:
            logits = torch.tanh(logits / self.config.logit_softcap) * self.config.logit_softcap

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return self.model.get_parameter_count()


# =============================================================================
# Quantization Utilities
# =============================================================================

def quantize_int6(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to int6 with per-row scaling.

    int6: values in [-31, 31], stored as int8.
    Scale per row stored as bfloat16.

    Args:
        weights: Weight tensor [out_features, in_features].

    Returns:
        Tuple of (quantized_int8, scales_bfloat16).
    """
    # Per-row scaling
    scales = weights.abs().max(dim=-1, keepdim=True)[0] / 31.0
    scales = scales.clamp(min=1e-8)

    # Quantize
    quantized = torch.round(weights / scales).clamp(-31, 31).to(torch.int8)

    return quantized, scales.squeeze().to(torch.bfloat16)


def dequantize_int6(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize int6 weights.

    Args:
        quantized: Quantized int8 tensor.
        scales: Per-row scales [out_features].

    Returns:
        Dequantized float tensor.
    """
    return quantized.float() * scales.unsqueeze(-1)
