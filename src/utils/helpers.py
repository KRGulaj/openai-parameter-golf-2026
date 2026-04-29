"""General utility functions for Parameter Golf.
"""

import sys
import logging
from typing import Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count total parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: Count only trainable parameters.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_memory(
    model: nn.Module,
    batch_size: int = 32,
    seq_len: int = 2048,
    dtype_bytes: int = 2
) -> dict:
    """Estimate model memory requirements.

    Args:
        model: PyTorch model.
        batch_size: Training batch size.
        seq_len: Sequence length.
        dtype_bytes: Bytes per parameter (2 for bfloat16).

    Returns:
        Dictionary with memory estimates in MB.
    """
    param_count = count_parameters(model)
    param_mb = (param_count * dtype_bytes) / (1024 * 1024)

    # Estimate activation memory (rough)
    # Forward + backward + gradients
    activation_mb = (
        batch_size * seq_len * 512 * dtype_bytes * 4  # 4x for activations
    ) / (1024 * 1024)

    # Optimizer states (momentum buffer etc.)
    optimizer_mb = (param_count * dtype_bytes * 2) / (1024 * 1024)  # 2x for momentum

    return {
        "parameters_mb": param_mb,
        "activations_mb": activation_mb,
        "optimizer_mb": optimizer_mb,
        "total_training_mb": param_mb + activation_mb + optimizer_mb,
        "total_inference_mb": param_mb + activation_mb,
    }


def format_size(size_bytes: float) -> str:
    """Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "15.9 MB").
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for logging.
        format_str: Optional custom format string.

    Returns:
        Configured logger.
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    logger = logging.getLogger("parameter_golf")

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)

    return logger


def get_device() -> torch.device:
    """Get the appropriate device for computation.

    Returns:
        CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_flops(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    num_layers: int
) -> int:
    """Estimate FLOPs for one forward pass.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        num_layers: Number of layers.

    Returns:
        Estimated FLOPs.
    """
    # Attention: 2 * B * T * d_model^2
    attn_flops = 2 * batch_size * seq_len * d_model * d_model

    # FFN: 2 * B * T * d_model * hidden_dim
    # Assuming 4x expansion
    ffn_flops = 8 * batch_size * seq_len * d_model * d_model

    # Per layer
    layer_flops = attn_flops + ffn_flops

    # Total
    total_flops = num_layers * layer_flops

    # LM head
    head_flops = 2 * batch_size * seq_len * d_model * vocab_size

    return total_flops + head_flops
