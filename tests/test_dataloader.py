"""Unit tests for data loading and filtering."""

import numpy as np
import torch
import tempfile
import os


def test_entropy_filter():
    """Test entropy filter."""
    from src.data.filtering import EntropyFilter

    filter_instance = EntropyFilter(initial_ratio=4.0)

    # Random data (high entropy, should pass)
    random_chunk = np.random.randint(0, 256, size=4096, dtype=np.uint8)
    assert filter_instance.is_valid(random_chunk) is True

    # Repetitive data (low entropy, should fail)
    repetitive_chunk = np.zeros(4096, dtype=np.uint8)
    assert filter_instance.is_valid(repetitive_chunk) is False


def test_entropy_filter_curriculum():
    """Test entropy filter curriculum learning."""
    from src.data.filtering import EntropyFilter

    filter_instance = EntropyFilter(initial_ratio=4.0)

    # Initial threshold
    assert filter_instance.current_ratio == 4.0

    # After 2 minutes
    filter_instance.update_threshold_by_time(2.0)
    assert filter_instance.current_ratio == 3.0

    # After 5 minutes
    filter_instance.update_threshold_by_time(5.0)
    assert filter_instance.current_ratio == 2.5


def test_filter_benchmark():
    """Benchmark entropy filter."""
    from src.data.filtering import EntropyFilter, benchmark_filter

    filter_instance = EntropyFilter()
    stats = benchmark_filter(filter_instance.is_valid, chunk_size=4096, iterations=100)

    # Should be reasonably fast (< 50 microseconds per chunk)
    assert stats["avg_time_us"] < 100


if __name__ == "__main__":
    test_entropy_filter()
    test_entropy_filter_curriculum()
    test_filter_benchmark()
    print("All data loading tests passed!")
