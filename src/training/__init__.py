"""Training utilities for Parameter Golf.

Provides schedulers, compression, and test-time training utilities.
"""

from src.training.scheduler import (
    WSDScheduler,
    CosineDecayScheduler,
    ConstantScheduler,
)
from src.training.compression import (
    quantize_tensor_int6,
    dequantize_tensor_int6,
    gptq_lite_find_clip_threshold,
    compress_model_int6_zstd,
    decompress_model_int6_zstd,
    estimate_compressed_size,
    check_artifact_budget,
    FakeQuantize,
    apply_fake_quantization,
)
from src.training.ttt import (
    LoRALayer,
    LinearWithLoRA,
    TestTimeTrainer,
    sliding_window_evaluate,
    compute_bits_per_byte,
)

__all__ = [
    # Schedulers
    "WSDScheduler",
    "CosineDecayScheduler",
    "ConstantScheduler",
    # Compression
    "quantize_tensor_int6",
    "dequantize_tensor_int6",
    "gptq_lite_find_clip_threshold",
    "compress_model_int6_zstd",
    "decompress_model_int6_zstd",
    "estimate_compressed_size",
    "check_artifact_budget",
    "FakeQuantize",
    "apply_fake_quantization",
    # TTT
    "LoRALayer",
    "LinearWithLoRA",
    "TestTimeTrainer",
    "sliding_window_evaluate",
    "compute_bits_per_byte",
]
