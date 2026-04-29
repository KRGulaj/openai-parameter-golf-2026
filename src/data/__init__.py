"""Data loading and filtering for Parameter Golf.

Provides high-performance data loading with memory-mapped I/O and
entropy-based quality filtering.
"""

from src.data.dataloader import (
    FastGolfDataLoader,
    SimpleDataLoader,
    create_dataloader,
)
from src.data.filtering import (
    EntropyFilter,
    ByteEntropyFilter,
    benchmark_filter,
)

__all__ = [
    "FastGolfDataLoader",
    "SimpleDataLoader",
    "create_dataloader",
    "EntropyFilter",
    "ByteEntropyFilter",
    "benchmark_filter",
]
