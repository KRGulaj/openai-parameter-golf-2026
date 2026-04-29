"""Configuration classes for Parameter Golf models.

This module provides dataclass-based configuration for the three model
architectures: Ouroboros (LoopLM), SSM (Mamba), and LiteFormer (Standard Transformer).
"""

from src.config.ouroboros_config import OuroborosConfig
from src.config.ssm_config import SSMConfig
from src.config.liteformer_config import LiteFormerConfig

__all__ = [
    "OuroborosConfig",
    "SSMConfig",
    "LiteFormerConfig",
]
