"""Model compression utilities for Parameter Golf.

Implements int6 quantization, GPTQ-lite, and zstd compression
to fit 35M parameters within 16 MB budget.
"""

import io
import math
from typing import Dict, Tuple, Optional, BinaryIO

import numpy as np
import torch
import torch.nn as nn
import zstandard as zstd


# =============================================================================
# int6 Quantization
# =============================================================================

def quantize_tensor_int6(
    tensor: torch.Tensor,
    clip_percentile: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to int6 with per-row scaling.

    int6: values in [-31, 31], stored as int8.
    Scale per row stored as bfloat16.

    Args:
        tensor: Weight tensor [out_features, in_features].
        clip_percentile: Optional percentile for clipping outliers.

    Returns:
        Tuple of (quantized_int8, scales_bfloat16).
    """
    original_shape = tensor.shape

    # Reshape to 2D if needed
    if tensor.ndim > 2:
        tensor = tensor.view(-1, original_shape[-1])

    # Optional outlier clipping
    if clip_percentile is not None:
        max_val = tensor.abs().quantile(clip_percentile)
        tensor = tensor.clamp(-max_val, max_val)

    # Per-row scaling
    max_abs = tensor.abs().max(dim=-1, keepdim=True)[0]
    scales = max_abs / 31.0
    scales = scales.clamp(min=1e-8)

    # Quantize to [-31, 31]
    quantized = torch.round(tensor / scales).clamp(-31, 31)
    quantized = quantized.to(torch.int8)

    # Scales as bfloat16
    scales_bf16 = scales.squeeze().to(torch.bfloat16)

    return quantized, scales_bf16


def dequantize_tensor_int6(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    """Dequantize int6 tensor.

    Args:
        quantized: Quantized int8 tensor.
        scales: Per-row scales.
        original_shape: Optional shape to restore.

    Returns:
        Dequantized float tensor.
    """
    dequantized = quantized.float() * scales.unsqueeze(-1)

    if original_shape is not None:
        dequantized = dequantized.view(original_shape)

    return dequantized


# =============================================================================
# GPTQ-lite
# =============================================================================

def gptq_lite_find_clip_threshold(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    percentiles: Tuple[float, ...] = (0.999, 0.9995, 0.9999, 0.99995, 1.0)
) -> Tuple[float, float]:
    """Find optimal clipping threshold to minimize reconstruction error.

    GPTQ-lite tries multiple percentiles and selects the one
    that minimizes MSE between original and quantized weights.

    Args:
        weight: Weight tensor [out_features, in_features].
        hessian: Hessian approximation for importance weighting.
        percentiles: Candidate percentiles to try.

    Returns:
        Tuple of (best_percentile, best_mse).
    """
    best_mse = float("inf")
    best_percentile = 1.0

    max_abs = weight.abs().max()

    for p in percentiles:
        clip_val = max_abs * p
        clipped = weight.clamp(-clip_val, clip_val)

        # Simple quantization without GPTQ error correction
        scales = clipped.abs().max(dim=-1, keepdim=True)[0] / 31.0
        scales = scales.clamp(min=1e-8)
        quant = torch.round(clipped / scales).clamp(-31, 31)
        dequant = quant * scales

        # MSE
        mse = ((weight - dequant) ** 2).mean().item()

        if mse < best_mse:
            best_mse = mse
            best_percentile = p

    return best_percentile, best_mse


# =============================================================================
# Model Compression
# =============================================================================

def compress_model_int6_zstd(
    model: nn.Module,
    zstd_level: int = 22
) -> bytes:
    """Compress model weights with int6 + zstd.

    Args:
        model: PyTorch model.
        zstd_level: Zstd compression level (1-22).

    Returns:
        Compressed bytes.
    """
    state_dict = model.state_dict()
    compressed_dict = {}

    for name, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]:
            # Quantize 2D weights
            if tensor.ndim >= 2 and tensor.numel() > 1024:
                orig_shape = tensor.shape
                reshaped = tensor.view(-1, orig_shape[-1])
                quantized, scales = quantize_tensor_int6(reshaped)

                compressed_dict[name] = {
                    "type": "int6",
                    "quantized": quantized.numpy(),
                    "scales": scales.numpy(),
                    "shape": orig_shape,
                }
            else:
                # Keep small tensors as bfloat16
                compressed_dict[name] = {
                    "type": "bf16",
                    "tensor": tensor.to(torch.bfloat16).numpy(),
                }
        else:
            # Keep other dtypes as-is
            compressed_dict[name] = {
                "type": "raw",
                "tensor": tensor.numpy(),
            }

    # Serialize with numpy
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{
        k: v if isinstance(v, np.ndarray) else str(v)
        for k, v in compressed_dict.items()
    })

    # Apply zstd
    compressor = zstd.ZstdCompressor(level=zstd_level)
    compressed = compressor.compress(buffer.getvalue())

    return compressed


def decompress_model_int6_zstd(
    compressed: bytes,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Decompress model weights from int6 + zstd.

    Args:
        compressed: Compressed bytes.
        device: Target device.

    Returns:
        Decompressed state dict.
    """
    # Decompress zstd
    decompressor = zstd.ZstdDecompressor()
    decompressed = decompressor.decompress(compressed)

    # Load numpy
    buffer = io.BytesIO(decompressed)
    np_dict = np.load(buffer, allow_pickle=True)

    state_dict = {}
    for name, data in np_dict.items():
        if isinstance(data, np.ndarray):
            state_dict[name] = torch.from_numpy(data).to(device)
        else:
            # Metadata, skip
            pass

    return state_dict


# =============================================================================
# Artifact Size Estimation
# =============================================================================

def estimate_compressed_size(
    model: nn.Module,
    compression_ratio: float = 0.82
) -> Tuple[float, float]:
    """Estimate compressed model size.

    Args:
        model: PyTorch model.
        compression_ratio: Expected zstd compression ratio.

    Returns:
        Tuple of (uncompressed_mb, compressed_mb).
    """
    total_params = sum(p.numel() for p in model.parameters())

    # int6: ~0.75 bytes per parameter (weights + scales)
    uncompressed_mb = total_params * 0.75 / (1024 * 1024)

    # zstd compression
    compressed_mb = uncompressed_mb * compression_ratio

    return uncompressed_mb, compressed_mb


def check_artifact_budget(
    model: nn.Module,
    code_bytes: int,
    budget_bytes: int = 16_000_000
) -> Tuple[bool, float]:
    """Check if model fits within artifact budget.

    Args:
        model: PyTorch model.
        code_bytes: Size of submission code.
        budget_bytes: Budget limit.

    Returns:
        Tuple of (fits, total_mb).
    """
    _, model_mb = estimate_compressed_size(model)
    model_bytes = model_mb * 1024 * 1024

    total_bytes = model_bytes + code_bytes
    total_mb = total_bytes / (1024 * 1024)

    fits = total_bytes <= budget_bytes

    return fits, total_mb


# =============================================================================
# Fake Quantization for QAT
# =============================================================================

class FakeQuantize(torch.autograd.Function):
    """Fake quantization for quantization-aware training.

    Simulates int6 quantization during training while maintaining
    gradient flow.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, num_bits: int = 6) -> torch.Tensor:
        """Forward pass with fake quantization.

        Args:
            x: Input tensor.
            num_bits: Number of bits (default 6).

        Returns:
            Fake-quantized tensor.
        """
        max_val = 2 ** (num_bits - 1) - 1

        # Compute scale
        scale = x.abs().max() / max_val
        scale = scale.clamp(min=1e-8)

        # Quantize and dequantize
        quantized = torch.round(x / scale).clamp(-max_val - 1, max_val)
        dequantized = quantized * scale

        return dequantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Straight-through estimator."""
        return grad_output, None, None


def apply_fake_quantization(
    model: nn.Module,
    num_bits: int = 6
) -> None:
    """Apply fake quantization to model weights.

    Args:
        model: Model to quantize.
        num_bits: Number of bits.
    """
    fake_quant = FakeQuantize.apply

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data = fake_quant(module.weight.data, num_bits)
