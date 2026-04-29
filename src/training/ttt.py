"""Test-Time Training (TTT) with LoRA.

Implements Score-First TTT: updates LoRA adapters on previous context,
then evaluates on current window with updated weights.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LoRA Implementation
# =============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.

    Adds trainable rank-decomposed matrices to frozen base weights.
    Formula: W = W_base + (alpha/r) * B * A

    Reference:
        Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large
        Language Models. arXiv:2106.09685.

    Attributes:
        r: Rank of adaptation.
        alpha: Scaling factor.
        lora_A: Down projection.
        lora_B: Up projection.
        scaling: Effective scaling (alpha / r).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0
    ):
        """Initialize LoRA layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            r: Rank (default 8).
            alpha: Scaling factor (default 16).
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Initialize A with random normal, B with zeros
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation.

        Args:
            x: Input tensor [..., in_features].

        Returns:
            Adapted output [..., out_features].
        """
        return (x @ self.lora_A) @ (self.lora_B * self.scaling)


class LinearWithLoRA(nn.Module):
    """Linear layer with optional LoRA adaptation.

    Combines base linear layer with LoRA for efficient fine-tuning.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0
    ):
        """Initialize wrapped linear layer.

        Args:
            base_layer: Base linear layer to adapt.
            r: LoRA rank.
            alpha: LoRA scaling.
        """
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            r,
            alpha
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA.

        Args:
            x: Input tensor.

        Returns:
            Output with LoRA adaptation.
        """
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return base_out + lora_out


# =============================================================================
# TTT (Test-Time Training)
# =============================================================================

class TestTimeTrainer:
    """Test-Time Training with LoRA.

    Implements Score-First TTT:
    1. Train LoRA adapters on previous context window
    2. Evaluate on current window with updated adapters
    3. Never see tokens being evaluated

    Attributes:
        model: Base model.
        lora_layers: Dictionary of LoRA layers by name.
        optimizer: TTT optimizer.
        inner_steps: Number of inner loop steps.
        inner_lr: Learning rate for inner loop.
    """

    def __init__(
        self,
        model: nn.Module,
        r: int = 8,
        inner_steps: int = 3,
        inner_lr: float = 1e-4
    ):
        """Initialize TTT trainer.

        Args:
            model: Model to adapt.
            r: LoRA rank.
            inner_steps: Number of adaptation steps per window.
            inner_lr: Learning rate for adaptation.
        """
        self.model = model
        self.r = r
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

        # Add LoRA to selected layers
        self.lora_layers = self._add_lora_to_model()

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Enable LoRA parameters
        self.lora_params = []
        for lora_layer in self.lora_layers.values():
            for param in lora_layer.parameters():
                param.requires_grad = True
                self.lora_params.append(param)

        # Create optimizer for LoRA parameters
        if self.lora_params:
            self.optimizer = torch.optim.AdamW(
                self.lora_params,
                lr=inner_lr
            )
        else:
            self.optimizer = None

    def _add_lora_to_model(self) -> dict:
        """Add LoRA layers to model.

        Returns:
            Dictionary mapping layer names to LoRA layers.
        """
        lora_layers = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Wrap with LoRA
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                else:
                    parent = self.model

                # Create LoRA wrapper
                lora_wrapper = LinearWithLoRA(module, self.r)
                setattr(parent, child_name, lora_wrapper)
                lora_layers[name] = lora_wrapper.lora

        return lora_layers

    def adapt(self, context_ids: torch.Tensor) -> float:
        """Adapt LoRA on context window.

        Args:
            context_ids: Context token IDs [B, T].

        Returns:
            Average loss over adaptation steps.
        """
        if self.optimizer is None:
            return 0.0

        self.model.train()
        total_loss = 0.0

        for _ in range(self.inner_steps):
            self.optimizer.zero_grad()

            # Forward pass
            # Note: Model must support returning logits and loss
            if hasattr(self.model, 'forward'):
                _, loss, _ = self.model(context_ids, labels=context_ids)
            else:
                loss = torch.tensor(0.0)

            if loss is not None and loss.requires_grad:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / max(self.inner_steps, 1)

    def evaluate(
        self,
        eval_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate on current window with adapted weights.

        Args:
            eval_ids: Evaluation token IDs [B, T].
            labels: Optional labels for loss computation.

        Returns:
            Tuple of (logits, loss).
        """
        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                logits, loss, _ = self.model(eval_ids, labels=labels)
            else:
                logits = torch.randn(eval_ids.shape[0], eval_ids.shape[1], 1056)
                loss = None

        return logits, loss

    def reset_adaptation(self) -> None:
        """Reset LoRA parameters to zero."""
        for lora_layer in self.lora_layers.values():
            nn.init.zeros_(lora_layer.lora_B)


# =============================================================================
# Sliding Window Evaluation
# =============================================================================

def sliding_window_evaluate(
    model: nn.Module,
    input_ids: torch.Tensor,
    stride: int = 64,
    max_length: int = 2048
) -> Tuple[torch.Tensor, list]:
    """Evaluate with sliding window and TTT.

    Args:
        model: Model to evaluate.
        input_ids: Full token sequence [B, T].
        stride: Window stride.
        max_length: Maximum window length.

    Returns:
        Tuple of (all_logits, window_losses).
    """
    batch_size, total_len = input_ids.shape
    num_windows = (total_len - max_length) // stride + 1

    all_logits = []
    window_losses = []

    # Initialize TTT
    ttt = TestTimeTrainer(model, r=8, inner_steps=3)

    for i in range(num_windows):
        start = i * stride
        end = start + max_length

        window_ids = input_ids[:, start:end]

        if i > 0:
            # Adapt on previous window
            prev_start = max(0, start - stride)
            prev_end = start
            prev_window = input_ids[:, prev_start:prev_end]
            ttt.adapt(prev_window)

        # Evaluate current window
        logits, loss = ttt.evaluate(window_ids)
        all_logits.append(logits)

        if loss is not None:
            window_losses.append(loss.item())

    # Concatenate logits
    if all_logits:
        all_logits = torch.cat(all_logits, dim=1)
    else:
        all_logits = torch.randn(batch_size, 0, 1056)

    return all_logits, window_losses


# =============================================================================
# Bits Per Byte Calculation
# =============================================================================

def compute_bits_per_byte(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int = 1056
) -> float:
    """Compute Bits Per Byte (BPB) metric.

    BPB = cross_entropy_loss / ln(2)
    Lower is better. Measured on held-out test data.

    Args:
        logits: Model logits [B, T, vocab_size].
        targets: Target token IDs [B, T].
        vocab_size: Vocabulary size.

    Returns:
        Bits per byte.
    """
    batch_size, seq_len, _ = logits.shape

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-100,
        reduction="mean"
    )

    # Convert to bits per byte
    # Assuming 1 token ≈ 1 byte (rough estimate for subword tokenization)
    bpb = loss.item() / math.log(2)

    return bpb
