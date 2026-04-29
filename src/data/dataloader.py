"""Fast data loader with memory-mapped I/O and async prefetching.

Implements FastGolfDataLoader optimized for multi-GPU training with
entropy filtering, pinned memory, and non-blocking GPU transfers.
"""

import os
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.data.filtering import EntropyFilter


class FastGolfDataLoader:
    """High-performance data loader for Parameter Golf.

    Features:
    - Memory-mapped file I/O (zero-copy reads)
    - Multi-threaded background prefetching
    - Entropy filtering with caching
    - Pinned memory for fast GPU transfer
    - Distributed striding for multi-GPU

    Attributes:
        bin_path: Path to tokenized binary file.
        batch_size: Batch size.
        seq_len: Sequence length.
        entropy_filter: Entropy filter instance.
        rank: GPU rank for distributed training.
        world_size: Total GPUs.
        device_id: CUDA device ID.
    """

    def __init__(
        self,
        bin_path: str,
        batch_size: int,
        seq_len: int,
        entropy_filter: Optional[EntropyFilter],
        rank: int = 0,
        world_size: int = 1
    ):
        """Initialize data loader.

        Args:
            bin_path: Path to .bin file with uint16 tokens.
            batch_size: Number of sequences per batch.
            seq_len: Sequence length (input + target).
            entropy_filter: Optional entropy filter.
            rank: GPU rank (0 to world_size-1).
            world_size: Total number of GPUs.
        """
        self.bin_path = bin_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.entropy_filter = entropy_filter
        self.rank = rank
        self.world_size = world_size

        # CUDA device for pinned memory
        self.device_id = int(os.environ.get("LOCAL_RANK", rank))

        # Chunk size: batch_size * (seq_len + 1) for input + target
        self.chunk_size = batch_size * (seq_len + 1)

        # Memory-mapped file
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        # Cache for last valid chunk (prevents deadlocks on filtered data)
        self.last_good_chunk: Optional[np.ndarray] = None

        # Async queue and thread
        self.batch_queue: queue.Queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

    def start(self) -> "FastGolfDataLoader":
        """Start background prefetching thread.

        Returns:
            Self for chaining.
        """
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        return self

    def _worker_loop(self) -> None:
        """Background thread for async data loading."""
        # Each rank starts at its own offset
        ptr = self.rank * self.chunk_size

        while not self.stop_event.is_set():
            # Wraparound with sharding
            if ptr + self.chunk_size > self.total_tokens:
                ptr = self.rank * self.chunk_size

            # Read chunk
            chunk = np.array(self.data[ptr:ptr + self.chunk_size], dtype=np.uint16)

            # Distributed stride
            ptr += self.world_size * self.chunk_size

            # Entropy filtering
            if self.entropy_filter is not None:
                if self.entropy_filter.is_valid(chunk):
                    self.last_good_chunk = chunk
                else:
                    if self.last_good_chunk is not None:
                        chunk = self.last_good_chunk
                    else:
                        continue  # Skip until first valid chunk

            # Convert to tensor and pin memory
            tensor_uint8 = torch.from_numpy(chunk.astype(np.uint8)).view(
                self.batch_size, self.seq_len + 1
            )
            pinned_tensor = tensor_uint8.pin_memory(
                device=torch.device(f"cuda:{self.device_id}")
            )

            self.batch_queue.put(pinned_tensor)

    def get_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch with non-blocking GPU transfer.

        Args:
            device: Target CUDA device.

        Returns:
            Tuple of (input_ids, target_ids).
        """
        pinned_tensor = self.batch_queue.get()

        # Non-blocking transfer to GPU with casting
        X = pinned_tensor[:, :-1].to(device, dtype=torch.long, non_blocking=True)
        Y = pinned_tensor[:, 1:].to(device, dtype=torch.long, non_blocking=True)

        return X, Y

    def stop(self) -> None:
        """Stop the background thread."""
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)

    def __iter__(self):
        """Iterator interface."""
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch."""
        device = torch.device(f"cuda:{self.device_id}")
        return self.get_batch(device)

    def __del__(self) -> None:
        """Cleanup."""
        self.stop()


class SimpleDataLoader:
    """Simplified data loader without async prefetching.

    Suitable for evaluation or debugging.
    """

    def __init__(
        self,
        bin_path: str,
        batch_size: int,
        seq_len: int,
        max_samples: Optional[int] = None
    ):
        """Initialize simple loader.

        Args:
            bin_path: Path to tokenized binary file.
            batch_size: Batch size.
            seq_len: Sequence length.
            max_samples: Maximum number of samples to load.
        """
        self.bin_path = bin_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_samples = max_samples

        # Load data
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        # Calculate number of batches
        self.chunk_size = batch_size * (seq_len + 1)
        self.num_batches = self.total_tokens // self.chunk_size
        if max_samples is not None:
            self.num_batches = min(self.num_batches, max_samples // batch_size)

        self.current_batch = 0

    def __len__(self) -> int:
        """Number of batches."""
        return self.num_batches

    def __iter__(self):
        """Iterator interface."""
        self.current_batch = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch."""
        if self.current_batch >= self.num_batches:
            raise StopIteration

        ptr = self.current_batch * self.chunk_size
        chunk = np.array(self.data[ptr:ptr + self.chunk_size], dtype=np.uint16)
        tensor = torch.from_numpy(chunk.astype(np.uint8)).view(
            self.batch_size, self.seq_len + 1
        )

        X = tensor[:, :-1].long()
        Y = tensor[:, 1:].long()

        self.current_batch += 1
        return X, Y


def create_dataloader(
    bin_path: str,
    batch_size: int,
    seq_len: int,
    use_filter: bool = True,
    rank: int = 0,
    world_size: int = 1
) -> FastGolfDataLoader:
    """Factory function for creating data loader.

    Args:
        bin_path: Path to .bin file.
        batch_size: Batch size.
        seq_len: Sequence length.
        use_filter: Whether to use entropy filtering.
        rank: GPU rank.
        world_size: Total GPUs.

    Returns:
        Configured FastGolfDataLoader.
    """
    filter_instance = EntropyFilter(initial_ratio=4.0) if use_filter else None
    return FastGolfDataLoader(
        bin_path=bin_path,
        batch_size=batch_size,
        seq_len=seq_len,
        entropy_filter=filter_instance,
        rank=rank,
        world_size=world_size
    ).start()
