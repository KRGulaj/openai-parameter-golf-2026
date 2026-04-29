"""Unit tests for model architectures."""

import torch
import pytest


def test_ouroboros6_forward():
    """Test Ouroboros6 forward pass."""
    from src.models.ouroboros import Ouroboros6ForCausalLM
    from src.config.ouroboros_config import OuroborosConfig

    config = OuroborosConfig()
    model = Ouroboros6ForCausalLM(config)

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, labels=input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0  # Loss should be positive


def test_ssm8_forward():
    """Test SSM8 forward pass (simplified if mamba not available)."""
    from src.models.ssm import SSM8ForCausalLM
    from src.config.ssm_config import SSMConfig

    config = SSMConfig()
    try:
        model = SSM8ForCausalLM(config)

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, loss, _ = model(input_ids, labels=input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    except RuntimeError as e:
        if "mamba_ssm" in str(e):
            pytest.skip("mamba_ssm not installed")
        raise


def test_liteformer_forward():
    """Test LiteFormer forward pass."""
    from src.models.liteformer import LiteFormerForCausalLM
    from src.config.liteformer_config import LiteFormerConfig

    config = LiteFormerConfig()
    model = LiteFormerForCausalLM(config)

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, labels=input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_parameter_counts():
    """Test that parameter counts are reasonable."""
    from src.models.ouroboros import Ouroboros6ForCausalLM
    from src.models.liteformer import LiteFormerForCausalLM
    from src.config.ouroboros_config import OuroborosConfig
    from src.config.liteformer_config import LiteFormerConfig

    # Ouroboros6 (~8M params)
    ouro_config = OuroborosConfig()
    ouro_model = Ouroboros6ForCausalLM(ouro_config)
    ouro_params = ouro_model.get_parameter_count()
    assert 6_000_000 < ouro_params < 10_000_000, f"Ouroboros6 has {ouro_params} params"

    # LiteFormer (~35M params)
    lite_config = LiteFormerConfig()
    lite_model = LiteFormerForCausalLM(lite_config)
    lite_params = lite_model.get_parameter_count()
    assert 30_000_000 < lite_params < 40_000_000, f"LiteFormer has {lite_params} params"


def test_gqa_attention():
    """Test Grouped Query Attention."""
    from src.models.liteformer import GroupedQueryAttention
    from src.config.liteformer_config import LiteFormerConfig

    config = LiteFormerConfig()
    config.max_seq_len = 32

    attn = GroupedQueryAttention(config)

    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, config.d_model)
    freqs_cos, freqs_sin = torch.randn(seq_len, config.partial_rope_dims), torch.randn(seq_len, config.partial_rope_dims)

    out = attn(x, freqs_cos, freqs_sin, config.partial_rope_dims, use_xsa=False)
    assert out.shape == x.shape


if __name__ == "__main__":
    test_ouroboros6_forward()
    test_liteformer_forward()
    test_parameter_counts()
    test_gqa_attention()
    print("All model tests passed!")
