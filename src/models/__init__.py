"""Neural network models for Parameter Golf.

This module provides implementations of three model architectures:
- Ouroboros6: Cascaded LoopLM with Mixture-of-Depths
- SSM8: "Fat State" Mamba State Space Model
- LiteFormer-11L: Compressed Transformer with int6+zstd
"""

from src.models.components import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    apply_partial_rope,
    Muon,
    zeropower_via_newtonschulz5,
    init_transformer_weights,
    count_parameters,
)
from src.models.ouroboros import (
    Ouroboros6Model,
    Ouroboros6ForCausalLM,
    OuroborosAttention,
    OuroborosMLP,
    MixtureOfDepthsRouter,
)
from src.models.ssm import (
    SSM8Model,
    SSM8ForCausalLM,
    SSM8Block,
    HeadAdapter,
    run_selective_scan,
)
from src.models.liteformer import (
    LiteFormerModel,
    LiteFormerForCausalLM,
    GroupedQueryAttention,
    ReluSquaredMLP,
    SmearGate,
    BigramHashEmbedding,
    quantize_int6,
    dequantize_int6,
)

__all__ = [
    # Components
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "apply_partial_rope",
    "Muon",
    "zeropower_via_newtonschulz5",
    "init_transformer_weights",
    "count_parameters",
    # Ouroboros
    "Ouroboros6Model",
    "Ouroboros6ForCausalLM",
    "OuroborosAttention",
    "OuroborosMLP",
    "MixtureOfDepthsRouter",
    # SSM
    "SSM8Model",
    "SSM8ForCausalLM",
    "SSM8Block",
    "HeadAdapter",
    "run_selective_scan",
    # LiteFormer
    "LiteFormerModel",
    "LiteFormerForCausalLM",
    "GroupedQueryAttention",
    "ReluSquaredMLP",
    "SmearGate",
    "BigramHashEmbedding",
    "quantize_int6",
    "dequantize_int6",
]
