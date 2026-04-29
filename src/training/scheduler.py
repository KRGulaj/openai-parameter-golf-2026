"""Learning rate schedulers for Parameter Golf.

Implements WSD (Warmup-Stable-Decay) schedule optimized for short
10-minute training runs with time-based control.
"""

import math
from typing import Optional

import torch


class WSDScheduler:
    """Warmup-Stable-Decay learning rate scheduler.

    Three-phase schedule:
    1. Warmup: Linear increase from 0 to max_lr
    2. Stable: Constant max_lr
    3. Decay: Cosine decay to min_lr

    Time-based control ensures consistent behavior across different
    hardware speeds and compilation overheads.

    Reference:
        Hägele, A., et al. (2024). Scaling Laws for Learning Rate
        Schedules. arXiv:2404.13022.

    Attributes:
        optimizer: PyTorch optimizer.
        max_lr: Maximum learning rate.
        min_lr: Minimum learning rate.
        warmup_seconds: Warmup duration.
        total_seconds: Total training time.
        current_lr: Current learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_seconds: float,
        warmup_seconds: float = 60.0,
        min_lr: float = 1e-6,
        decay_fraction: float = 0.2
    ):
        """Initialize WSD scheduler.

        Args:
            optimizer: PyTorch optimizer.
            max_lr: Peak learning rate.
            total_seconds: Total training time in seconds.
            warmup_seconds: Warmup duration.
            min_lr: Final learning rate.
            decay_fraction: Fraction of time for decay phase.
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_seconds = warmup_seconds
        self.total_seconds = total_seconds
        self.decay_fraction = decay_fraction

        # Calculate phase boundaries
        self.stable_end = total_seconds * (1.0 - decay_fraction)
        self.decay_end = total_seconds

        self.current_lr = 0.0
        self.elapsed_seconds = 0.0

    def step(self, elapsed_seconds: float) -> float:
        """Update learning rate based on elapsed time.

        Args:
            elapsed_seconds: Time since training start.

        Returns:
            Current learning rate.
        """
        self.elapsed_seconds = elapsed_seconds

        # Clamp to total time
        t = min(elapsed_seconds, self.total_seconds)

        if t < self.warmup_seconds:
            # Warmup phase: linear
            progress = t / self.warmup_seconds
            lr = self.max_lr * progress
        elif t < self.stable_end:
            # Stable phase: constant
            lr = self.max_lr
        else:
            # Decay phase: cosine
            decay_progress = (t - self.stable_end) / (self.decay_end - self.stable_end)
            decay_progress = min(decay_progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        self.current_lr = lr

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_lr_scale(self) -> float:
        """Get current learning rate as fraction of max.

        Returns:
            Scale in [0, 1].
        """
        if self.max_lr > 0:
            return self.current_lr / self.max_lr
        return 0.0

    def is_warmup(self) -> bool:
        """Check if currently in warmup phase."""
        return self.elapsed_seconds < self.warmup_seconds

    def is_stable(self) -> bool:
        """Check if currently in stable phase."""
        return self.warmup_seconds <= self.elapsed_seconds < self.stable_end

    def is_decay(self) -> bool:
        """Check if currently in decay phase."""
        return self.elapsed_seconds >= self.stable_end


class CosineDecayScheduler:
    """Simple cosine decay scheduler.

    Continuous cosine annealing from max_lr to min_lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_seconds: float,
        min_lr: float = 1e-6
    ):
        """Initialize cosine scheduler.

        Args:
            optimizer: PyTorch optimizer.
            max_lr: Initial learning rate.
            total_seconds: Total training time.
            min_lr: Final learning rate.
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_seconds = total_seconds

    def step(self, elapsed_seconds: float) -> float:
        """Update learning rate.

        Args:
            elapsed_seconds: Elapsed time.

        Returns:
            Current learning rate.
        """
        t = min(elapsed_seconds, self.total_seconds)
        progress = t / self.total_seconds

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class ConstantScheduler:
    """Constant learning rate scheduler."""

    def __init__(self, optimizer: torch.optim.Optimizer, lr: float):
        """Initialize constant scheduler.

        Args:
            optimizer: PyTorch optimizer.
            lr: Constant learning rate.
        """
        self.optimizer = optimizer
        self.lr = lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, elapsed_seconds: float) -> float:
        """Return constant learning rate.

        Args:
            elapsed_seconds: Unused (for interface compatibility).

        Returns:
            Constant learning rate.
        """
        return self.lr
