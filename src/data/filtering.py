"""Data filtering with entropy-based quality heuristics.

This module implements zlib-based entropy filtering with curriculum learning,
gradually increasing quality requirements during training.
"""

import zlib
from typing import Optional

import numpy as np
import torch


class EntropyFilter:
    """Dynamic data filter using zlib compression ratios.

    Uses zlib level 1 for high-throughput entropy estimation.
    Implements curriculum learning: threshold decreases from 4.0 to 2.5
    over training to gradually increase data quality.

    Attributes:
        current_ratio: Current compression ratio threshold.
        initial_ratio: Starting threshold.
        stats_accepted: Number of accepted chunks.
        stats_rejected: Number of rejected chunks.
    """

    def __init__(self, initial_ratio: float = 4.0):
        """Initialize entropy filter.

        Args:
            initial_ratio: Starting compression ratio threshold (default 4.0).
                           Lower ratio = higher quality data.
                           Ratio 1.0 = incompressible (random data).
                           Ratio > 2.0 = compressible (repetitive/low entropy).
        """
        self.current_ratio = initial_ratio
        self.initial_ratio = initial_ratio
        self.stats_accepted = 0
        self.stats_rejected = 0

    def update_threshold_by_time(self, elapsed_minutes: float) -> None:
        """Update threshold based on training progress.

        Curriculum schedule:
        - 0-2 min: ratio = 4.0 (lenient, high throughput)
        - 2-5 min: ratio = 3.0 (moderate)
        - 5+ min: ratio = 2.5 (strict, high quality)

        Args:
            elapsed_minutes: Training time elapsed in minutes.
        """
        if elapsed_minutes >= 5.0:
            self.current_ratio = 2.5
        elif elapsed_minutes >= 2.0:
            self.current_ratio = 3.0
        else:
            self.current_ratio = 4.0

    def update_threshold_by_step(self, step: int, total_steps: int) -> None:
        """Update threshold based on training step.

        Alternative to time-based scheduling for deterministic behavior.

        Args:
            step: Current training step.
            total_steps: Total expected training steps.
        """
        progress = step / max(total_steps, 1)
        if progress >= 0.8:
            self.current_ratio = 2.5
        elif progress >= 0.4:
            self.current_ratio = 3.0
        else:
            self.current_ratio = 4.0

    def compute_ratio(self, chunk: np.ndarray) -> float:
        """Compute compression ratio for a data chunk.

        Args:
            chunk: Numpy array of token IDs (uint8).

        Returns:
            Compression ratio (original_size / compressed_size).
            Higher ratio = more compressible = lower entropy.
        """
        if chunk.size == 0:
            return float("inf")

        raw_bytes = chunk.tobytes()
        original_size = len(raw_bytes)

        if original_size == 0:
            return float("inf")

        # Level 1: Fastest, minimal compression
        compressed = zlib.compress(raw_bytes, level=1)
        compressed_size = len(compressed)

        if compressed_size == 0:
            return float("inf")

        return original_size / compressed_size

    def is_valid(self, chunk: np.ndarray) -> bool:
        """Evaluate if a data chunk passes the quality filter.

        Args:
            chunk: Numpy array of token IDs.

        Returns:
            True if chunk is accepted, False if rejected.
        """
        ratio = self.compute_ratio(chunk)

        if ratio > self.current_ratio:
            self.stats_rejected += 1
            return False
        else:
            self.stats_accepted += 1
            return True

    def get_yield_ratio(self) -> float:
        """Get the percentage of accepted chunks.

        Returns:
            Acceptance rate as percentage (0-100).
        """
        total = self.stats_accepted + self.stats_rejected
        if total == 0:
            return 0.0
        return (self.stats_accepted / total) * 100.0

    def get_stats(self) -> dict:
        """Get filtering statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "accepted": self.stats_accepted,
            "rejected": self.stats_rejected,
            "yield_ratio": self.get_yield_ratio(),
            "current_threshold": self.current_ratio,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats_accepted = 0
        self.stats_rejected = 0


class ByteEntropyFilter:
    """Alternative filter using byte-level unique count.

    Measures number of unique bytes in chunk as entropy proxy.
    Faster than zlib but less precise.

    Attributes:
        min_unique_threshold: Minimum unique bytes threshold.
    """

    def __init__(self, min_unique_threshold: int = 200):
        """Initialize byte filter.

        Args:
            min_unique_threshold: Minimum unique bytes (0-256).
        """
        self.min_unique_threshold = min_unique_threshold
        self.stats_accepted = 0
        self.stats_rejected = 0

    def is_valid(self, chunk: np.ndarray) -> bool:
        """Evaluate chunk based on byte diversity.

        Args:
            chunk: Numpy array of token IDs.

        Returns:
            True if chunk passes filter.
        """
        if chunk.size == 0:
            return False

        unique_bytes = len(np.unique(chunk))

        if unique_bytes < self.min_unique_threshold:
            self.stats_rejected += 1
            return False
        else:
            self.stats_accepted += 1
            return True

    def get_yield_ratio(self) -> float:
        """Get acceptance rate."""
        total = self.stats_accepted + self.stats_rejected
        if total == 0:
            return 0.0
        return (self.stats_accepted / total) * 100.0


def benchmark_filter(filter_fn, chunk_size: int = 4096, iterations: int = 10000) -> dict:
    """Benchmark filter performance.

    Args:
        filter_fn: Function that takes numpy array and returns bool.
        chunk_size: Size of test chunks.
        iterations: Number of iterations.

    Returns:
        Timing statistics.
    """
    import time

    # Generate test data
    good_data = np.random.randint(0, 256, size=chunk_size, dtype=np.uint8)
    bad_data = np.zeros(chunk_size, dtype=np.uint8)  # All zeros (highly compressible)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        filter_fn(good_data)
        filter_fn(bad_data)
    elapsed = time.perf_counter() - start

    avg_time_us = (elapsed / (iterations * 2)) * 1e6

    return {
        "total_time_ms": elapsed * 1000,
        "avg_time_us": avg_time_us,
        "iterations": iterations,
    }
